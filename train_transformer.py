"""Small Transformer trained from scratch; no pretrained weights are used."""
import argparse
import json
import math
import random
import re
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import DATA, RESULTS, read_jsonl, write_jsonl

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(s):
    return TOKEN_RE.findall(s.lower())


def detokenize(tokens):
    return re.sub(r"\s+([,.!?;:%)\]}])", r"\1", " ".join(tokens)).replace("( ", "(").replace("' ", "'")


def vocab_from(rows, field, size):
    counts = Counter(t for row in rows for t in tokenize(row[field]))
    return SPECIAL + [t for t, _ in counts.most_common(size - len(SPECIAL))]


def encode(s, vocab, length):
    ids = {t: i for i, t in enumerate(vocab)}
    out = [BOS] + [ids.get(t, UNK) for t in tokenize(s)[:length - 2]] + [EOS]
    return out + [PAD] * (length - len(out))


class SmallTransformer(nn.Module):
    def __init__(self, src_size, tgt_size, dim=128, layers=2, heads=4, max_len=48):
        super().__init__()
        self.src = nn.Embedding(src_size, dim, padding_idx=PAD)
        self.tgt = nn.Embedding(tgt_size, dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, dim)
        self.net = nn.Transformer(d_model=dim, nhead=heads, num_encoder_layers=layers,
                                  num_decoder_layers=layers, dim_feedforward=dim * 4,
                                  dropout=0.1, batch_first=True, norm_first=True)
        self.out = nn.Linear(dim, tgt_size)

    def forward(self, source, target):
        sp = torch.arange(source.size(1), device=source.device)
        tp = torch.arange(target.size(1), device=target.device)
        src = self.src(source) + self.pos(sp)
        tgt = self.tgt(target) + self.pos(tp)
        causal = nn.Transformer.generate_square_subsequent_mask(target.size(1), device=target.device)
        hidden = self.net(src, tgt, tgt_mask=causal,
                          src_key_padding_mask=source.eq(PAD),
                          tgt_key_padding_mask=target.eq(PAD),
                          memory_key_padding_mask=source.eq(PAD))
        return self.out(hidden)


@torch.no_grad()
def translate(model, source, vocab, max_len):
    target = torch.tensor([[BOS]], device=source.device)
    for _ in range(max_len - 1):
        logits = model(source, target)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        target = torch.cat([target, next_id], dim=1)
        if next_id.item() == EOS:
            break
    return detokenize([vocab[i] for i in target[0].tolist()[1:] if i not in (PAD, EOS, BOS)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=40)
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--max-test", type=int, default=0, help="0 means full test set")
    args = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    train = read_jsonl(DATA / "train.jsonl")
    validation = read_jsonl(DATA / "validation.jsonl")
    test = read_jsonl(DATA / "test.jsonl")[:args.max_test or None]
    src_vocab = vocab_from(train, "source", args.vocab_size)
    tgt_vocab = vocab_from(train, "reference", args.vocab_size)
    def batches(rows, shuffle=False):
        pairs = [(torch.tensor(encode(r["source"], src_vocab, args.max_length)),
                  torch.tensor(encode(r["reference"], tgt_vocab, args.max_length))) for r in rows]
        return DataLoader(pairs, batch_size=args.batch_size, shuffle=shuffle)
    model = SmallTransformer(len(src_vocab), len(tgt_vocab), max_len=args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    best = math.inf
    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for source, target in batches(train, True):
            optimizer.zero_grad()
            logits = model(source, target[:, :-1])
            loss = loss_fn(logits.reshape(-1, len(tgt_vocab)), target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for source, target in batches(validation):
                logits = model(source, target[:, :-1])
                val_loss += loss_fn(logits.reshape(-1, len(tgt_vocab)), target[:, 1:].reshape(-1)).item()
        val_loss /= max(1, len(batches(validation)))
        history.append({"epoch": epoch + 1, "training_loss_sum": running, "validation_loss": val_loss})
        print(f"Epoch {epoch + 1}: validation loss {val_loss:.4f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                        "max_length": args.max_length}, RESULTS / "transformer.pt")
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(RESULTS / "transformer.pt", weights_only=False)["model"])
    model.eval()
    rows = []
    inference_start = time.perf_counter()
    for i, row in enumerate(test):
        source = torch.tensor([encode(row["source"], src_vocab, args.max_length)])
        rows.append({**row, "prediction": translate(model, source, tgt_vocab, args.max_length)})
        if (i + 1) % 10 == 0:
            print(f"Translated {i + 1}/{len(test)}", flush=True)
    inference_seconds = time.perf_counter() - inference_start
    write_jsonl(RESULTS / "transformer_predictions.jsonl", rows)
    metrics = {"model": "SmallTransformer", "method": "trained from scratch", "train_seconds": train_seconds,
               "inference_seconds": inference_seconds, "parameters": sum(p.numel() for p in model.parameters()),
               "test_count": len(rows), "history": history}
    (RESULTS / "transformer_run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
