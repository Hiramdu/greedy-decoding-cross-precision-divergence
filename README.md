# Greedy Decoding Is Not Precision-Invariant

Official code for the accepted TMLR paper **“Greedy Decoding Is Not
Precision-Invariant: Cross-Precision Output Divergence in LLM Inference.”**

This repository studies numerical precision in large language model inference:
the same model, prompt, hardware, and greedy decoding algorithm can produce
different token sequences in BF16 and FP16. The paper connects these flips to
small top-two logit margins at the output head and evaluates selective FP32
`lm_head` recomputation as a targeted mitigation.

> Gaoyuan Du, Anam Nawaz Khan, Rex Zhou, Xiaoyang Liu, Deepayan Chakrabarti,
> Fnu Suya, and Xueping Li. Transactions on Machine Learning Research, 2026.

## Paper

- [OpenReview discussion and paper](https://openreview.net/forum?id=QDOKyg7a5e)
- [PDF](https://openreview.net/pdf?id=QDOKyg7a5e)
- [TMLR accepted-papers index](https://jmlr.org/tmlr/papers/)
- [BibTeX from TMLR](https://jmlr.org/tmlr/papers/bib/QDOKyg7a5e.bib)

## Main result

Greedy decoding is deterministic only after the numerical execution path is
fixed. Changing inference precision from BF16 to FP16 can alter the winning
token when the leading logits are close. Once one token changes, autoregressive
feedback can make the remaining generation diverge.

The paper reports cross-precision divergence on 59–82% of prompts across six
models, four model families, and three benchmarks. Its selective intervention
recomputes the full output head in FP32 only when the native top-two logit margin
falls below a threshold.

## Repository contents

- [`run_core_experiment.py`](run_core_experiment.py) reproduces the central
  BF16-versus-FP16 comparison and Intervention C.
- [`requirements.txt`](requirements.txt) lists the Python dependencies.
- [`paper.bib`](paper.bib), [`CITATION.cff`](CITATION.cff), and
  [`codemeta.json`](codemeta.json) provide machine-readable citation metadata.
- [`llms.txt`](llms.txt) gives a compact index for web agents and retrieval
  systems.

This compact release intentionally excludes experiment outputs, checkpoints,
caches, logs, paper sources, extended causal analyses, cross-model sweeps, FP8
experiments, task-quality tests, and auxiliary ablations.

## Requirements

- Linux and Python 3.10 or newer
- A CUDA GPU with native BF16 support
- Enough GPU memory for the selected model

The paper's primary TinyLlama experiments used an NVIDIA A10G. Model weights
and the public GSM8K test split are downloaded from Hugging Face on first use.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run a two-prompt smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python run_core_experiment.py \
  --n-prompts 2 \
  --max-new-tokens 32
```

Run the core paper configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python run_core_experiment.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --n-prompts 100 \
  --max-new-tokens 256 \
  --threshold 1e-3 \
  --seed 42
```

Progress is written to standard error. The script prints one JSON summary to
standard output and does not create output files. The main fields are:

- `baseline.exact_agreement_rate`
- `intervention_c.exact_agreement_rate`
- `absolute_ear_lift`
- `intervention_c.trigger_rate`

To test another compatible Hugging Face causal language model, change
`--model-id`. Models requiring custom loading code can be enabled with
`--trust-remote-code`.

## Method implemented

At decoding step `t`, the script computes the native-precision top-two margin:

```text
margin = largest_logit - second_largest_logit
```

When `margin < tau`, the final hidden state and output-head weights are cast to
FP32 for one full-vocabulary `lm_head` recomputation. The next token is selected
from the FP32 logits. Large-margin steps use the native logits unchanged.

BF16 and FP16 models are loaded and evaluated sequentially so the comparison
can run on one GPU. Exact agreement requires the complete generated token
sequences to match.

## Reproducibility note

Numerical outcomes can vary with the model, GPU architecture, CUDA version,
PyTorch version, and Transformers version. These variations are part of the
phenomenon studied in the paper, so record the complete software and hardware
environment when reporting results.

## Citation

```bibtex
@article{du2026greedy,
  title   = {Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in {LLM} Inference},
  author  = {Gaoyuan Du and Anam Nawaz Khan and Rex Zhou and Xiaoyang Liu and Deepayan Chakrabarti and Fnu Suya and Xueping Li},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=QDOKyg7a5e}
}
```

## Contact

For questions about the code or paper, open a GitHub issue.
