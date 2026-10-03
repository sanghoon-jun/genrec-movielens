"""GenRec: history -> text -> LLM -> h (last token) -> score = h @ E.T

Same head and loss as sasrec.py; only the encoder differs.
"""
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from common import evaluate, fmt, load, log_history, machine, pick_device, save_metrics, start_run


def verbalize(d, items, ratings, n, prompt="v2"):
    # v2: title + rating; v3: + genres
    def movie(i, r):
        genres = f" [{d.movies.genres[i]}]" if prompt == "v3" else ""
        return f"{d.movies.title[i]}{genres} {r}"
    movies = "; ".join(movie(i, r) for i, r in zip(items[-n:], ratings[-n:]))
    return f"Watched (rating 1-5): {movies}\nNext:"


class GenRec(nn.Module):
    def __init__(self, n_items, backbone, lora_r=16, dtype=torch.float32, grad_ckpt=False):
        super().__init__()
        llm = AutoModel.from_pretrained(backbone, dtype=dtype)
        lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        if grad_ckpt:  # recompute activations in backward: ~30% slower, ~GBs less memory
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()  # needed for checkpointing with a frozen base
        self.llm = get_peft_model(llm, lora)
        self.norm = nn.LayerNorm(llm.config.hidden_size)  # LLM h has norm ~300 -> huge logits without this
        self.item = nn.Embedding(n_items + 1, llm.config.hidden_size)  # scoring head E
        nn.init.normal_(self.item.weight, std=0.02)

    def encode(self, tokens):  # left-padded -> (B, d), last token
        return self.norm(self.llm(**tokens).last_hidden_state[:, -1].float())

    def score(self, h):
        return h @ self.item.weight.T


class Encoder:
    """histories -> tokens on device."""
    def __init__(self, d, tok, n, device, prompt):
        self.d, self.tok, self.n, self.device, self.prompt = d, tok, n, device, prompt

    def __call__(self, users, hist):
        texts = [verbalize(self.d, h, self.d.ratings[u], self.n, self.prompt) for u, h in zip(users, hist)]
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)


def score_fn(model, enc):
    def f(users, hist):
        model.eval()
        return model.score(model.encode(enc(users, hist)))
    return f


def train_samples(d, per_user):
    """(user, history, target, target rating) with random cut points in each train seq."""
    out = []
    for u, s in enumerate(d.split("train")[0]):
        for t in random.sample(range(1, len(s)), min(per_user, len(s) - 1)):
            out.append((u, s[:t], s[t], d.ratings[u][t]))
    random.shuffle(out)
    return out


def train(args):
    run = start_run("genrec", args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    d = load()
    if args.users:
        d.seqs, d.ratings = d.seqs[:args.users], d.ratings[:args.users]

    tok = AutoTokenizer.from_pretrained(args.backbone, padding_side="left")
    enc = Encoder(d, tok, args.history, device, args.prompt)
    model = GenRec(d.n_items, args.backbone, args.lora_r, dtype, args.grad_ckpt).to(device)
    model.llm.print_trainable_parameters()
    if args.resume:  # continue from a saved checkpoint (LoRA + head); a fresh LR schedule starts
        missing = model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
        print(f"resumed from {args.resume} ({len(missing.unexpected_keys)} unexpected keys)")
    print("example prompt:\n" + verbalize(d, d.seqs[0][:-2], d.ratings[0], args.history, args.prompt))

    opt = torch.optim.AdamW([
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [*model.norm.parameters(), *model.item.parameters()], "lr": args.head_lr},
    ])
    # linear warmup -> linear decay to 0 (standard for LLM fine-tuning; without it training can collapse to popularity)
    n_samples = sum(min(args.samples_per_user, len(s) - 1) for s in d.split("train")[0])
    total_steps = args.epochs * -(-n_samples // args.batch_size)
    warmup = max(1, int(args.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min((t + 1) / warmup, max(0.0, (total_steps - t) / (total_steps - warmup))))
    params = [p for p in model.parameters() if p.requires_grad]
    best, best_state = -1.0, None

    t0, global_step = time.time(), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        samples = train_samples(d, args.samples_per_user)
        total = window = 0.0
        for step, i in enumerate(range(0, len(samples), args.batch_size), 1):
            users, hist, tgt, rating = zip(*samples[i:i + args.batch_size])
            logits = model.score(model.encode(enc(users, hist)))
            loss = F.cross_entropy(logits, torch.tensor(tgt, device=device), reduction="none")
            if args.rating_weight:  # paper's reward weighting: weight by rating, mean 1 per batch
                w = torch.tensor(rating, device=device, dtype=torch.float)
                loss = loss * w / w.mean()
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            total += loss.item()
            window += loss.item()
            global_step += 1
            if step % args.log_every == 0:
                print(f"  step {step}  loss={total / step:.4f}  {(time.time() - t0) / 60:.1f} min")
                log_history(run, global_step, loss=window / args.log_every)
                window = 0.0

        m = evaluate(score_fn(model, enc), d, "valid", batch_size=args.eval_batch_size)
        print(f"epoch {epoch}  loss={total / step:.4f}  valid {fmt(m)}")
        log_history(run, global_step, **{f"valid_{k}": v for k, v in m.items()})
        trainable = {k for k, p in model.named_parameters() if p.requires_grad}
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}
        torch.save(state, run / "last.pt")  # trainable params only: LoRA + norm + item embeddings
        if m["NDCG@10"] > best:
            best, best_state = m["NDCG@10"], state
            torch.save(state, run / "best.pt")

    minutes = (time.time() - t0) / 60
    model.load_state_dict(best_state, strict=False)
    m = evaluate(score_fn(model, enc), d, "test", batch_size=args.eval_batch_size)
    print("GenRec test ", fmt(m))
    print(f"train time {minutes:.1f} min on {machine(device)} ({device})")
    save_metrics(run, args, m, minutes, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"])
    p.add_argument("--users", type=int, help="subset for quick debugging")
    p.add_argument("--history", type=int, default=20, help="movies in prompt")
    p.add_argument("--prompt", choices=["v2", "v3"], default="v2", help="v2: title+rating, v3: +genres")
    p.add_argument("--samples_per_user", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA")
    p.add_argument("--head_lr", type=float, default=1e-3, help="item embeddings")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--grad_ckpt", action="store_true", help="save memory, e.g. 16GB Mac")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    p.add_argument("--clip", type=float, default=1.0, help="grad norm clip")
    p.add_argument("--rating_weight", action="store_true", help="weight loss by target rating")
    p.add_argument("--resume", help="checkpoint (.pt) to start from, e.g. runs/<run>/best.pt")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
