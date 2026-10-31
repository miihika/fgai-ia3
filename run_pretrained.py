"""Run mT5 fine-tuning or prompt a compact LLaMA/Gemma architecture model."""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from common import DATA, RESULTS, read_jsonl, write_jsonl

MODELS = {
    "mt5": "google/mt5-small",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma": "google/gemma-3-270m-it",
}


def run_mt5(args, test):
    train = read_jsonl(DATA / "train.jsonl")
    model_id = MODELS["mt5"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def collate(rows):
        sources = ["translate English to French: " + r["source"] for r in rows]
        inputs = tokenizer(sources, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        targets = tokenizer(text_target=[r["reference"] for r in rows], padding=True,
                            truncation=True, max_length=args.max_length, return_tensors="pt")
        labels = targets["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return inputs, labels

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    started = time.perf_counter()
    losses = []
    for step, (inputs, labels) in enumerate(loader, start=1):
        optimizer.zero_grad()
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"mT5 step {step}/{args.max_steps}: loss {loss.item():.4f}", flush=True)
        if step >= args.max_steps:
            break
    train_seconds = time.perf_counter() - started
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(test, 1):
            inputs = tokenizer("translate English to French: " + row["source"],
                               truncation=True, max_length=args.max_length, return_tensors="pt")
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            predictions.append({**row, "prediction": tokenizer.decode(ids[0], skip_special_tokens=True).strip()})
            if i % 10 == 0:
                print(f"mT5 translated {i}/{len(test)}", flush=True)
    inference_seconds = time.perf_counter() - started
    return model, predictions, train_seconds, inference_seconds, losses


def generate_causal(model, tokenizer, source, args, temperature=None):
    instruction = f"Translate this English text into French. Return only the French translation.\nEnglish: {source}\nFrench:"
    messages = [{"role": "user", "content": instruction}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_tensors="pt")
        if not isinstance(encoded, torch.Tensor):
            encoded = encoded["input_ids"]
    else:
        encoded = tokenizer(instruction, return_tensors="pt")["input_ids"]
    options = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if temperature is None:
        options["do_sample"] = False
    else:
        options.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        output = model.generate(encoded, **options)
    answer = tokenizer.decode(output[0, encoded.shape[1]:], skip_special_tokens=True).strip()
    return answer.split("\n")[0].strip()


def run_causal(args, test):
    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", low_cpu_mem_usage=True)
    model.eval()
    predictions = []
    started = time.perf_counter()
    for i, row in enumerate(test, 1):
        predictions.append({**row, "prediction": generate_causal(model, tokenizer, row["source"], args)})
        if i % 10 == 0:
            print(f"{args.model} translated {i}/{len(test)}", flush=True)
    inference_seconds = time.perf_counter() - started
    # Temperature is examined separately; deterministic outputs above are used for BLEU/chrF.
    samples = []
    for row in test[:args.temperature_examples]:
        for temperature in (0.3, 0.7, 1.0):
            samples.append({**row, "temperature": temperature,
                            "prediction": generate_causal(model, tokenizer, row["source"], args, temperature)})
    write_jsonl(RESULTS / f"{args.model}_temperature.jsonl", samples)
    return model, predictions, None, inference_seconds, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--max-steps", type=int, default=200, help="mT5 gradient steps")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature-examples", type=int, default=3)
    ap.add_argument("--max-test", type=int, default=0, help="0 means full test set")
    args = ap.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    test = read_jsonl(DATA / "test.jsonl")[:args.max_test or None]
    if args.model == "mt5":
        model, predictions, train_seconds, inference_seconds, losses = run_mt5(args, test)
        method = f"full fine-tuning for {len(losses)} gradient steps"
    else:
        model, predictions, train_seconds, inference_seconds, losses = run_causal(args, test)
        method = "zero-shot prompting; no fine-tuning"
    write_jsonl(RESULTS / f"{args.model}_predictions.jsonl", predictions)
    metadata = {"model": MODELS[args.model], "method": method, "train_seconds": train_seconds,
                "inference_seconds": inference_seconds, "parameters": sum(p.numel() for p in model.parameters()),
                "test_count": len(predictions), "losses": losses}
    (RESULTS / f"{args.model}_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
