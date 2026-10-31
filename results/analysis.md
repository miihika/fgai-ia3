# Evaluation and responsible use analysis

## Experimental scope

The experiment used 2,000 OPUS Books English-French training pairs, 100 validation pairs, and the same 50 held-out test pairs for all four models. The basic Transformer was trained from scratch for 10 epochs and selected by validation loss (best at epoch 9). mT5-small received 200 gradient updates with batch size 2, so it saw 400 shuffled training pairs. TinyLlama and Gemma used zero-shot prompts. Main predictions used greedy decoding with an output cap. BLEU and chrF were computed over the complete 50-example test set with SacreBLEU.

## Measured results

| Model | BLEU | chrF | Training time | CPU inference per example | Parameters |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.32 | 9.97 | 337.5 s | 0.058 s | 2,471,328 |
| mT5-small | 0.07 | 4.37 | 399.0 s | 1.838 s | 300,176,768 |
| TinyLlama 1.1B | **9.90** | **34.40** | Not fine-tuned | 12.012 s | 1,100,048,384 |
| Gemma 3 270M | 4.44 | 22.92 | Not fine-tuned | 2.188 s | 268,098,176 |

All times were measured on a Windows computer with 16 GB RAM and CPU-only PyTorch. Several runs overlapped, so wall-clock timings are useful as rough local costs rather than controlled performance benchmarks. mT5 inference was timed on 100 examples before the final test set was set to 50; its per-example figure is the measured 100-example average. BLEU and chrF for every model use the same 50 examples. More precise values and raw timings are in `summary.csv`.

TinyLlama scored highest on both automatic metrics, but it also had the largest parameter count and slowest CPU inference. Gemma was smaller and faster, though its score was lower. The scratch Transformer was extremely small and fast but could not learn robust translation from 2,000 literary pairs. The limited mT5 run often repeated the same phrase; the 200-update training budget was insufficient. These results do not establish that decoder-only models are inherently better at translation. The models differ in scale, pretraining, tokenization, and adaptation. TinyLlama and Gemma started with instruction tuning, while the scratch model started with random weights. TinyLlama is a LLaMA-architecture model, rather than an official Meta checkpoint.

## Translation errors and ethical concerns

The saved samples show concrete failure modes. For the sentence about a light curricle and two mares, TinyLlama translated the vehicle as a “vélo de lumière” and changed the horses' description. For the sentence about two people silently re-entering a house, it produced an unrelated cheerful statement. Such invented details can mislead a reader even when a model's overall BLEU is the highest. Gemma often copied the English source instead of translating it. The scratch model emitted `<unk>` tokens, and mT5 repeated “l'un”; these are visible signs that their outputs should not be used without review.

The literary dataset contains dated wording and culture-specific expressions. One test sentence uses a historically derogatory term for a group of people. A translation system might repeat a slur, change its severity, or erase context. Gendered terms such as “young girl” also merit inspection for stereotyping or incorrect agreement. The system should retain the source beside the translation, mark uncertain or untranslated output, check names, numbers, identity terms, and sensitive passages, and require a fluent human reviewer for consequential use. Future evaluation should include current-domain text and examples covering different groups and dialects. BLEU and chrF alone cannot verify factual accuracy, cultural appropriateness, or safety.

## Sampling temperature

The main scores use deterministic decoding. Separate files `llama_temperature.jsonl` and `gemma_temperature.jsonl` contain three held-out examples sampled at temperatures 0.3, 0.7, and 1.0 with top-p 0.9. On the animal sentence, TinyLlama's higher-temperature output changed the animals to “reptiles” and added unsupported details. On the curricle sentence, its sampled versions varied widely in the vehicle and horse descriptions. Gemma copied the English sentence at all three temperatures for two examples, while its third example changed wording without reliably preserving meaning. A higher temperature increased variation but did not consistently improve translation. This qualitative comparison uses only three examples and is not a general performance estimate.

## Conclusion

**TinyLlama was the best-performing model in this specific small experiment by BLEU and chrF.** Its errors and 12-second average CPU latency mean it is not ready for unsupervised use. The comparison mainly shows how strongly pretraining and the adaptation budget affect this assignment's results.
