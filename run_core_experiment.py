#!/usr/bin/env python3
"""Minimal reproduction of the paper's core cross-precision experiment.

The script compares BF16 and FP16 greedy decoding under:

1. the native-precision baseline; and
2. Intervention C: recompute the full ``lm_head`` in FP32 when the native
   top-two logit margin is below ``tau``.

It downloads public GSM8K prompts, runs both precisions sequentially on one
CUDA GPU, and prints aggregate metrics to stdout. It does not write files.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class GenerationResult:
    """Token sequence and intervention statistics for one prompt."""

    token_ids: list[int]
    triggered_steps: int
    total_steps: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BF16 and FP16 exact agreement before and after selective "
            "FP32 lm_head recomputation."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model and tokenizer.",
    )
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    """Configure repeatable execution within each numerical precision."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_gsm8k_prompts(n_prompts: int) -> list[str]:
    """Load the first ``n_prompts`` public GSM8K test questions."""

    if n_prompts <= 0:
        raise ValueError("--n-prompts must be positive")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    n_selected = min(n_prompts, len(dataset))
    return [
        f"Solve step by step:\n{dataset[index]['question']}\nAnswer:"
        for index in range(n_selected)
    ]


def load_model(
    model_id: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool,
):
    """Load one low-precision arm on a single device."""

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def eos_token_ids(model) -> set[int]:
    """Return all configured end-of-sequence token IDs."""

    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


def fp32_lm_head_logits(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply the model output head in FP32 without storing a persistent copy."""

    output_head = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise RuntimeError("The model does not expose a linear output embedding.")

    weight = output_head.weight.float()
    bias = getattr(output_head, "bias", None)
    bias_fp32 = bias.float() if bias is not None else None
    return F.linear(hidden_state.float(), weight, bias_fp32)


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    threshold: float,
    intervention_c: bool,
    device: str,
) -> GenerationResult:
    """Generate one sequence with the baseline or Intervention C."""

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    current_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    past_key_values = None
    generated: list[int] = []
    triggered_steps = 0
    stop_ids = eos_token_ids(model)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=intervention_c,
            return_dict=True,
        )
        native_logits = outputs.logits[:, -1, :]
        top2_values = torch.topk(native_logits[0], k=2).values
        margin = float((top2_values[0] - top2_values[1]).item())

        should_trigger = intervention_c and (
            threshold == 0.0 or margin < threshold
        )
        if should_trigger:
            final_hidden_state = outputs.hidden_states[-1][:, -1, :]
            repaired_logits = fp32_lm_head_logits(model, final_hidden_state)
            next_token = int(repaired_logits.argmax(dim=-1).item())
            triggered_steps += 1
        else:
            next_token = int(native_logits.argmax(dim=-1).item())

        generated.append(next_token)
        if next_token in stop_ids:
            break

        past_key_values = outputs.past_key_values
        current_ids = torch.tensor(
            [[next_token]], dtype=torch.long, device=device
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return GenerationResult(
        token_ids=generated,
        triggered_steps=triggered_steps,
        total_steps=len(generated),
        elapsed_seconds=elapsed_seconds,
    )


def run_precision(
    model,
    tokenizer,
    prompts: Iterable[str],
    *,
    precision_name: str,
    max_new_tokens: int,
    max_input_tokens: int,
    threshold: float,
    device: str,
) -> dict[str, list[GenerationResult]]:
    """Run both decoding modes for one numerical precision."""

    prompt_list = list(prompts)
    results: dict[str, list[GenerationResult]] = {
        "baseline": [],
        "intervention_c": [],
    }

    for mode_name, intervention_c in (
        ("baseline", False),
        ("intervention_c", True),
    ):
        for index, prompt in enumerate(prompt_list, start=1):
            result = greedy_generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                threshold=threshold,
                intervention_c=intervention_c,
                device=device,
            )
            results[mode_name].append(result)
            print(
                f"[{precision_name}/{mode_name}] "
                f"{index}/{len(prompt_list)}",
                file=sys.stderr,
            )

    return results


def exact_agreement(
    left: list[GenerationResult],
    right: list[GenerationResult],
) -> tuple[int, float]:
    """Compute whole-sequence exact agreement."""

    if len(left) != len(right):
        raise ValueError("Precision arms contain different numbers of prompts")
    agreed = sum(
        first.token_ids == second.token_ids
        for first, second in zip(left, right)
    )
    return agreed, agreed / len(left)


def mode_summary(
    bf16: list[GenerationResult],
    fp16: list[GenerationResult],
) -> dict[str, float | int]:
    """Aggregate one mode across the two precision arms."""

    agreed, rate = exact_agreement(bf16, fp16)
    all_results = bf16 + fp16
    triggered = sum(item.triggered_steps for item in all_results)
    steps = sum(item.total_steps for item in all_results)
    return {
        "agreed_prompts": agreed,
        "exact_agreement_rate": round(rate, 4),
        "triggered_steps": triggered,
        "total_decoding_steps": steps,
        "trigger_rate": round(triggered / steps, 6) if steps else 0.0,
        "mean_seconds_per_prompt": round(
            statistics.fmean(item.elapsed_seconds for item in all_results), 4
        ),
    }


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no CUDA device is available")
    if args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support native BF16")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")

    prompts = load_gsm8k_prompts(args.n_prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    results_by_precision: dict[str, dict[str, list[GenerationResult]]] = {}
    for precision_name, dtype in PRECISIONS.items():
        print(
            f"Loading {args.model_id} in {precision_name}...",
            file=sys.stderr,
        )
        model = load_model(
            args.model_id,
            dtype,
            args.device,
            args.trust_remote_code,
        )
        results_by_precision[precision_name] = run_precision(
            model,
            tokenizer,
            prompts,
            precision_name=precision_name,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            threshold=args.threshold,
            device=args.device,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = mode_summary(
        results_by_precision["bf16"]["baseline"],
        results_by_precision["fp16"]["baseline"],
    )
    intervention = mode_summary(
        results_by_precision["bf16"]["intervention_c"],
        results_by_precision["fp16"]["intervention_c"],
    )

    summary = {
        "model": args.model_id,
        "dataset": "GSM8K test",
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "seed": args.seed,
        "baseline": baseline,
        "intervention_c": intervention,
        "absolute_ear_lift": round(
            intervention["exact_agreement_rate"]
            - baseline["exact_agreement_rate"],
            4,
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
