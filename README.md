# Compact English to French Machine Translation Comparison

## Objective

Compare encoder-decoder and decoder-only machine translation on one held-out English-to-French test set. The four systems are a Transformer trained from scratch, fine-tuned mT5-small, a prompted LLaMA-architecture model, and a prompted Gemma model. The deliberately small experiment fits a CPU-only, 16 GB RAM computer; its scores are an educational comparison, not a production benchmark.

## Dataset

[OPUS Books English-French](https://huggingface.co/datasets/Helsinki-NLP/opus_books) contains aligned book text. The source page lists about 127,000 English-French rows. `prepare.py` keeps pairs with 3-30 English words and 3-35 French words, normalizes whitespace, removes exact duplicate pairs without regard to case, shuffles with seed 42, and selects 2,000 training, 100 validation, and 50 test pairs. The three splits are stored in `data/`. The selected text is literary and sometimes dated, which limits generalization to current, conversational language. Please acknowledge the OPUS source if you reuse the data.

## Models and adaptation

| System | Architecture | Adaptation |
|---|---|---|
| `SmallTransformer` | Encoder-decoder, 2 layers per side, 128-dimensional embeddings, 4 attention heads | Trained from scratch on the 2,000 training pairs for 10 epochs; validation loss selects the checkpoint |
| [`google/mt5-small`](https://huggingface.co/google/mt5-small) | Encoder-decoder | Full fine-tuning for 200 gradient steps with batch size 2, consuming 400 shuffled pairs from the training split |
| [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | LLaMA decoder-only architecture | Zero-shot translation prompt; no fine-tuning |
| [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) | Gemma decoder-only architecture | Zero-shot translation prompt; no fine-tuning |

TinyLlama is a smaller LLaMA-architecture model, rather than an official Meta LLaMA checkpoint. The official Gemma model requires accepting its terms and authenticating with Hugging Face before running it. Model parameter counts, training and inference times, BLEU, and chrF are measured by the scripts and saved in `results/summary.csv`. Prompted models have no task-specific training time. Model sizes, pretraining data, tokenization, and adaptation differ, so score differences cannot be attributed to architecture alone.

## Reproduce the experiment

Python 3.11 or newer and about 16 GB of RAM are recommended. A GPU will speed up training and inference. Install dependencies with `python -m pip install -r requirements.txt`, then run from this folder:

```powershell
python prepare.py --train 2000 --validation 100 --test 50
python train_transformer.py
python run_pretrained.py mt5
python run_pretrained.py llama
python run_pretrained.py gemma
python evaluate.py
```

The first three model downloads use the Hugging Face cache and are not part of this repository. To run Gemma, visit its linked model page, accept the terms, then authenticate locally with `hf auth login`. The scripts save one prediction for each test pair. Main BLEU and chrF use deterministic greedy decoding; `run_pretrained.py` also creates a separate temperature sample file for each prompted model using temperatures 0.3, 0.7, and 1.0 on the first three test examples. These sampled outputs are analyzed separately because temperature changes decoding rather than model training.

## Results and comparison

| Model | BLEU | chrF | Training time | CPU seconds per test example | Parameters |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.32 | 9.97 | 337.5 s | 0.058 | 2.47M |
| mT5-small | 0.07 | 4.37 | 399.0 s | 1.838 | 300.18M |
| TinyLlama 1.1B | **9.90** | **34.40** | Not fine-tuned | 12.012 | 1,100.05M |
| Gemma 3 270M | 4.44 | 22.92 | Not fine-tuned | 2.188 | 268.10M |

**TinyLlama performed best on this test set by both scores**, with the slowest CPU inference. The basic Transformer was fastest but scored poorly; mT5's 200-update run was too short for useful translation. The architecture comparison is confounded by different model sizes, pretraining, tokenizers, and adaptation. See `results/analysis.md` for the error review, ethical concerns, and temperature analysis. Raw measurements are in `results/summary.csv`, plots in `results/scores.png` and `results/compute.png`, and generated translations in `results/samples.md`. The 50-pair test set makes score estimates noisy. Several model runs overlapped, so times are approximate local costs. The first mT5 run generated 100 predictions before the final test size was set to 50; its per-example time is the measured 100-example average and its BLEU/chrF use only the common first 50. Re-running the commands above times exactly 50 examples.

## Sampling temperature analysis

For three examples, prompted models were sampled at temperatures 0.3, 0.7, and 1.0. Higher temperature increased wording variation but did not reliably improve fidelity: TinyLlama changed the source animals into “reptiles” in one 1.0 output, while Gemma copied an English sentence at all three temperatures. These are qualitative observations from only three examples. The main BLEU and chrF results use greedy decoding, not these samples; see the two `results/*_temperature.jsonl` files.

## Responsible use

The literary corpus can contain dated views and uneven cultural representation. Translation systems may change names or facts, invent details, mistranslate sensitive wording, or amplify gender stereotypes. Inspect examples from each system, especially names, numbers, identity terms, and sensitive passages. For consequential translation, use a human reviewer fluent in both languages. Do not rely on BLEU or chrF alone; they compare wording with one reference and do not establish semantic safety. Keep source text visible, flag low-confidence outputs, and evaluate on representative current-domain material before deployment.

## Submission

This project's GitHub repository is https://github.com/miihika/fgai-ia3. The `data/`, source files, `results/`, and this README are included. The scratch model checkpoint (`results/transformer.pt`) can also be reproduced by the training script. The completed Word submission sheet and its PDF are in `submission/final/`; upload that PDF to the LMS. The older sheet in `submission/` is a draft.
