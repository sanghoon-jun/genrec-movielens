"""GenRec: history -> text -> LLM -> h (last token) -> score = h @ E.T

Same head and loss as sasrec.py; only the encoder differs.
--all_positions (GenRec-AP, not in the paper): rank the next movie after every movie in the prompt, as SASRec does.
"""
import argparse
import bisect
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from common import add_synopses, evaluate, fmt, load, log_history, machine, pick_device, save_metrics, start_run


PROMPTS = {"rating": "title + rating", "genres": "+ genres", "profile": "genres + user profile",
           "synopsis": "genres + synopsis for the last 5 movies"}
SYNOPSIS_LAST = 5  # synopsis prompt: only for the most recent movies (token budget)


def verbalize(d, items, ratings, n, prompt="rating", user=None, ends=False):
    """ratings aligned with items. ends=True also returns the char index of the ';' after each movie but the last."""
    items, ratings = items[-n:], ratings[-n:]

    def movie(k, i, r):
        genres = f" [{d.movies.genres[i]}]" if prompt != "rating" else ""
        syn = d.movies.synopsis[i] if prompt == "synopsis" and k >= len(items) - SYNOPSIS_LAST else ""
        return f"{d.movies.title[i]}{genres} {r}" + (f" ({syn})" if syn else "")
    parts = [movie(k, i, r) for k, (i, r) in enumerate(zip(items, ratings))]
    user_line = f"User: {d.profiles[user]}\n" if prompt == "profile" else ""
    head = f"{user_line}Watched (rating 1-5): "
    text = head + "; ".join(parts) + "\nNext:"
    if not ends:
        return text
    seps, c = [], len(head)
    for part in parts[:-1]:
        c += len(part)
        seps.append(c)
        c += 2  # "; "
    return text, seps


class GenRec(nn.Module):
    def __init__(self, n_items, backbone, lora_r=16, dtype=torch.float32, grad_ckpt=False, lm=False):
        super().__init__()
        # lm=True keeps the LM head for the paper's language loss; the ranking path is identical either way
        llm = (AutoModelForCausalLM if lm else AutoModel).from_pretrained(backbone, dtype=dtype)
        lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        if grad_ckpt:  # recompute activations in backward: ~30% slower, ~GBs less memory
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()  # needed for checkpointing with a frozen base
        self.llm = get_peft_model(llm, lora)
        self.norm = nn.LayerNorm(llm.config.hidden_size)  # LLM h has norm ~300 -> huge logits without this
        self.item = nn.Embedding(n_items + 1, llm.config.hidden_size)  # scoring head E
        nn.init.normal_(self.item.weight, std=0.02)

    def hidden(self, tokens):  # transformer body (with LoRA) -> (B, L, d)
        base = self.llm.get_base_model()
        return getattr(base, "model", base)(**tokens).last_hidden_state

    def encode(self, tokens):  # left-padded -> (B, d), last token
        return self.norm(self.hidden(tokens)[:, -1].float())

    def encode_at(self, tokens, rows, pos):  # GenRec-AP without LM loss: h at the given (row, position) pairs
        return self.norm(self.hidden(tokens)[rows, pos].float())

    def encode_lm(self, tokens, title_len, at=None):
        """Prompt + target title in one pass: h at the last prompt token (before the title), or at the
        (rows, positions) in `at` (GenRec-AP); LM loss on the title tokens."""
        H = self.hidden(tokens)
        L = H.size(1)
        rows, pos = at or (torch.arange(H.size(0), device=H.device), L - title_len - 1)
        h = self.norm(H[rows, pos].float())
        b, pos = zip(*[(i, L - n - 1 + k) for i, n in enumerate(title_len.tolist()) for k in range(n)])
        b, pos = torch.tensor(b, device=H.device), torch.tensor(pos, device=H.device)
        logits = self.llm.get_base_model().lm_head(H[b, pos]).float()  # position pos predicts token pos+1
        lm_loss = F.cross_entropy(logits, tokens["input_ids"][b, pos + 1])
        return h, lm_loss

    def score(self, h):
        return h @ self.item.weight.T


class Encoder:
    """histories -> tokens on device."""
    def __init__(self, d, tok, n, device, prompt):
        self.d, self.tok, self.n, self.device, self.prompt = d, tok, n, device, prompt

    def texts(self, users, hist):  # hist is a prefix of the user's sequence -> its ratings are the first len(h)
        return [verbalize(self.d, h, self.d.ratings[u][:len(h)], self.n, self.prompt, u) for u, h in zip(users, hist)]

    def __call__(self, users, hist):
        return self.tok(self.texts(users, hist), return_tensors="pt", padding=True).to(self.device)

    def with_titles(self, users, hist, tgt):
        """Prompt followed by the target title (training only); left-padded. Returns tokens, title lengths."""
        prompts = self.tok(self.texts(users, hist))["input_ids"]
        titles = self.tok([" " + self.d.movies.title[t] for t in tgt], add_special_tokens=False)["input_ids"]
        seqs = [p + t for p, t in zip(prompts, titles)]
        L, pad = max(map(len, seqs)), self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in seqs])
        mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs])
        tokens = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
        return tokens, torch.tensor([len(t) for t in titles], device=self.device)

    def all_positions(self, samples, titles):
        """GenRec-AP batch from all_position_samples. Each movie s[j], j in [first, end), is a query for s[j + 1]:
        h at the ';' after it (causal: sees movies up to s[j] only), or at the last prompt token for the final movie,
        as in evaluation. titles=True appends the title of s[end] for the LM loss.
        Returns tokens (left-padded), title lengths, query (rows, positions), targets."""
        seqs, title_lens, queries = [], [], []
        for u, end, first in samples:
            s = self.d.seqs[u]
            text, seps = verbalize(self.d, s[:end], self.d.ratings[u][:end], self.n, self.prompt, u, ends=True)
            enc = self.tok(text, return_offsets_mapping=True)
            ids, tok_end = enc["input_ids"], [z for _, z in enc["offset_mapping"]]
            q = [bisect.bisect_right(tok_end, c) for c in seps]  # token holding the ';' at char c
            assert all(not text[c + 1:tok_end[t]].strip() for c, t in zip(seps, q)), "';' token runs into the next movie"
            q.append(len(ids) - 1)
            start = max(0, end - self.n)  # s[start] is the first movie in the prompt
            queries.append([(q[j - start], s[j + 1]) for j in range(first, end)])
            title = self.tok(" " + self.d.movies.title[s[end]], add_special_tokens=False)["input_ids"] if titles else []
            seqs.append(ids + title)
            title_lens.append(len(title))
        L, pad = max(map(len, seqs)), self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in seqs])
        mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in seqs])
        rows, pos, tgt = zip(*[(b, L - len(x) + p, t) for b, (x, qs) in enumerate(zip(seqs, queries)) for p, t in qs])
        tokens = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
        dev = lambda x: torch.tensor(x, device=self.device)
        return tokens, dev(title_lens), (dev(rows), dev(pos)), dev(tgt)


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


def all_position_samples(d, n):
    """GenRec-AP: (user, end, first) chunks that use every train target once per epoch (random offset per user).
    The prompt holds the n movies before s[end]; movies s[first..end-1] each predict the next one."""
    out = []
    for u, s in enumerate(d.split("train")[0]):
        ends = list(range(random.randint(1, n), len(s) - 1, n)) + [len(s) - 1]
        out += [(u, end, first) for first, end in zip([0] + ends, ends)]
    random.shuffle(out)
    return out


def train(args):
    if args.all_positions and args.rating_weight:
        raise ValueError("--rating_weight is not supported with --all_positions")
    run = start_run("genrec", args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":  # cuDNN attention (used on H100-class GPUs) made training NaN; A100 is unaffected
        torch.backends.cuda.enable_cudnn_sdp(False)
    d = load()
    if args.prompt == "synopsis":
        add_synopses(d)
    if args.users:
        d.seqs, d.ratings = d.seqs[:args.users], d.ratings[:args.users]

    tok = AutoTokenizer.from_pretrained(args.backbone, padding_side="left")
    enc = Encoder(d, tok, args.history, device, args.prompt)
    model = GenRec(d.n_items, args.backbone, args.lora_r, dtype, args.grad_ckpt, lm=args.lm_weight > 0).to(device)
    model.llm.print_trainable_parameters()
    if args.resume:  # continue from a saved checkpoint (LoRA + head); a fresh LR schedule starts
        missing = model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
        print(f"resumed from {args.resume} ({len(missing.unexpected_keys)} unexpected keys)")
    print("example prompt:\n" + verbalize(d, d.seqs[0][:-2], d.ratings[0], args.history, args.prompt, 0))

    opt = torch.optim.AdamW([
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [*model.norm.parameters(), *model.item.parameters()], "lr": args.head_lr},
    ])
    # linear warmup -> linear decay to 0 (standard for LLM fine-tuning; without it training can collapse to popularity)
    if args.all_positions:  # chunks drawn up front so the schedule knows the step count
        ap_epochs = [all_position_samples(d, args.history) for _ in range(args.epochs)]
        total_steps = sum(-(-len(s) // args.batch_size) for s in ap_epochs)
    else:
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
        samples = ap_epochs[epoch - 1] if args.all_positions else train_samples(d, args.samples_per_user)
        total = window = 0.0
        for step, i in enumerate(range(0, len(samples), args.batch_size), 1):
            if args.all_positions:  # every movie in the prompt predicts the next one
                tokens, title_len, at, tgt = enc.all_positions(samples[i:i + args.batch_size], args.lm_weight > 0)
                h, lm_loss = model.encode_lm(tokens, title_len, at) if args.lm_weight > 0 \
                    else (model.encode_at(tokens, *at), 0.0)
            else:
                users, hist, tgt, rating = zip(*samples[i:i + args.batch_size])
                tgt = torch.tensor(tgt, device=device)
                if args.lm_weight > 0:  # paper: L = α·L_rank + β·L_language, α + β = 1
                    h, lm_loss = model.encode_lm(*enc.with_titles(users, hist, tgt.tolist()))
                else:
                    h, lm_loss = model.encode(enc(users, hist)), 0.0
            logits = model.score(h)
            loss = F.cross_entropy(logits, tgt, reduction="none")
            if args.rating_weight:  # paper's reward weighting: weight by rating, mean 1 per batch
                w = torch.tensor(rating, device=device, dtype=torch.float)
                loss = loss * w / w.mean()
            loss = (1 - args.lm_weight) * loss.mean() + args.lm_weight * lm_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
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
    p.add_argument("--prompt", choices=list(PROMPTS), default="rating",
                   help=", ".join(f"{k}: {v}" for k, v in PROMPTS.items()))
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
    p.add_argument("--lm_weight", type=float, default=0.0, help="β for the paper's language loss on the target title")
    p.add_argument("--all_positions", action="store_true",
                   help="GenRec-AP (not in the paper): rank the next movie after every movie in the prompt")
    p.add_argument("--resume", help="checkpoint (.pt) to start from, e.g. runs/<run>/best.pt")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
