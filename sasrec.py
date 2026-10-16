"""SASRec with full-catalog cross-entropy.

Same shape as GenRec:  encoder(history) -> h -> score = h @ E.T
SASRec encoder = small transformer over item ids; GenRec encoder = LLM over text.
"""
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import evaluate, fmt, load, log_history, machine, pick_device, save_metrics, start_run


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, d=64, layers=2, heads=1, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)  # also the scoring head E
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        for emb in (self.item, self.pos):  # small init: N(0,1) gives huge logits
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def encode(self, seq):  # (B, L) left-padded -> (B, L, d)
        L = seq.size(1)
        x = self.item(seq) * self.item.embedding_dim ** 0.5 + self.pos.weight[:L]
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=seq.device)
        return self.norm(self.encoder(x, mask=causal, is_causal=True))

    def score(self, h):  # (..., d) -> (..., n_items + 1)
        return h @ self.item.weight.T


def pad_left(seqs, L):
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[-L:]
        if s:
            out[i, -len(s):] = torch.tensor(s)
    return out


def chunks(seqs, L):
    """SASRec (chunked): windows of L + 1 items with a random offset, so every train target is used once per epoch
    (the default trains on each user's last L + 1 items only)."""
    out = []
    for s in seqs:
        o = random.randint(1, L)
        out.append(s[:o + 1])                                    # targets 1..o
        out += [s[i:i + L + 1] for i in range(o, len(s) - 1, L)]  # targets i+1..i+L
    return out


def score_fn(model, device):
    def f(users, hist):
        model.eval()
        h = model.encode(pad_left(hist, model.max_len).to(device))[:, -1]
        return model.score(h)
    return f


def train(args):
    run = start_run("sasrec", args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)
    d = load()
    if args.users:
        d.seqs, d.ratings = d.seqs[:args.users], d.ratings[:args.users]

    model = SASRec(d.n_items, args.max_len, args.dim, args.layers, args.heads, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # input = s[:-1], label = s[1:] (shifted by one, L+1 window)
    train_seqs = [s for s in d.split("train")[0] if len(s) > 1]
    best, best_state = -1.0, None

    t0, global_step = time.time(), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seqs = chunks(train_seqs, args.max_len) if args.chunk else train_seqs
        random.shuffle(seqs)
        total = 0.0
        for i in range(0, len(seqs), args.batch_size):
            batch = pad_left(seqs[i:i + args.batch_size], args.max_len + 1).to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model.score(model.encode(x))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            global_step += 1
        log_history(run, global_step, loss=total / (i // args.batch_size + 1))

        if epoch % args.eval_every == 0:
            m = evaluate(score_fn(model, device), d, "valid")
            print(f"epoch {epoch:3d}  loss={total / (i // args.batch_size + 1):.4f}  valid {fmt(m)}")
            log_history(run, global_step, **{f"valid_{k}": v for k, v in m.items()})
            if m["NDCG@10"] > best:
                best, best_state = m["NDCG@10"], {k: v.clone() for k, v in model.state_dict().items()}

    minutes = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    m = evaluate(score_fn(model, device), d, "test")
    print("SASRec test ", fmt(m))
    print(f"train time {minutes:.1f} min on {machine(device)} ({device})")
    save_metrics(run, args, m, minutes, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps", "cuda"])
    p.add_argument("--users", type=int, help="subset for quick debugging")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--chunk", action="store_true", help="train on every position via max_len + 1 windows")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
