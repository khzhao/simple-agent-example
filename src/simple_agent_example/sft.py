"""Supervised fine-tuning script for the 2048 agent.

This mirrors the SFT setup used in the old repo: it loads a JSONL dataset of
prompt/completion pairs, tokenizes them with the Tinker tokenizer, and runs a
few LoRA updates via the official Tinker SDK.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from dotenv import load_dotenv

import tinker
from tinker import ServiceClient
from tinker.types import AdamParams


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_datum(
    tokenizer,
    prompt: str,
    completion: str,
) -> tinker.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(
        completion,
        add_special_tokens=False,
    )

    tokens = prompt_tokens + completion_tokens

    model_input = tinker.types.ModelInput.from_ints(tokens[:-1])
    target_tokens = tinker.TensorData.from_torch(
        torch.tensor(tokens[1:], dtype=torch.long)
    )
    weights = tinker.TensorData.from_torch(
        torch.tensor(
            [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens),
            dtype=torch.float32,
        )
    )

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="SFT for the 2048 agent.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sft/2048_sft_dataset.jsonl"),
        help="Path to JSONL file containing prompt/completion pairs.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-name", default="2048-sft-initializer")
    parser.add_argument("--tinker-api-key", default=None)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=3,
        help="Number of dataset prompts to preview after training (0 disables).",
    )
    parser.add_argument(
        "--preview-max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per preview completion.",
    )
    parser.add_argument(
        "--preview-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for preview generations.",
    )
    parser.add_argument(
        "--preview-top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value for preview generations.",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    else:
        root_logger.setLevel(log_level)

    records = list(iter_jsonl(args.dataset))
    if not records:
        raise ValueError(f"No samples found in {args.dataset}")

    logging.info("Loaded %s SFT samples from %s", len(records), args.dataset)

    logging.info(
        "Creating LoRA training client for %s (rank=%d)",
        args.base_model,
        args.rank,
    )
    service_client = ServiceClient(api_key=args.tinker_api_key)
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
    )
    tokenizer = training_client.get_tokenizer()

    datums = [build_datum(tokenizer, r["prompt"], r["completion"]) for r in records]
    datums = [datum for datum in datums if datum is not None]
    logging.info("Prepared %s datum objects", len(datums))
    prompt_lengths = [len(tokenizer.encode(r["prompt"], add_special_tokens=False)) for r in records]
    completion_lengths = [len(tokenizer.encode(r["completion"], add_special_tokens=False)) for r in records]
    logging.info(
        "Prompt tokens: mean=%.1f max=%d | Completion tokens: mean=%.1f max=%d",
        float(np.mean(prompt_lengths)),
        int(np.max(prompt_lengths)),
        float(np.mean(completion_lengths)),
        int(np.max(completion_lengths)),
    )

    adam_params = AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )

    steps = args.steps
    batch_size = args.batch_size
    losses = []

    total = len(datums)
    for step in range(steps):
        start = (step * batch_size) % total
        end = start + batch_size
        if end <= total:
            batch = datums[start:end]
        else:
            batch = datums[start:] + datums[: end % total]

        logging.debug(
            "Step %04d/%04d - running forward/backward on batch size %d",
            step + 1,
            steps,
            len(batch),
        )
        fwdbwd = training_client.forward_backward(batch, "cross_entropy").result()
        training_client.optim_step(adam_params).result()

        logprobs = np.concatenate(
            [
                out["logprobs"].tolist()
                for out in fwdbwd.loss_fn_outputs
                if "logprobs" in out
            ]
        )
        weights = np.concatenate(
            [d.loss_fn_inputs["weights"].tolist() for d in batch]
        )
        loss = -np.dot(logprobs, weights) / np.clip(weights.sum(), 1e-6, None)
        losses.append(float(loss))
        logging.debug(
            "Step %04d - batch loss %.4f (tokens=%d)",
            step + 1,
            loss,
            len(batch),
        )

        if (step + 1) % args.log_every == 0:
            logging.info(
                "Step %04d/%04d - loss %.4f",
                step + 1,
                steps,
                np.mean(losses[-args.log_every :]),
            )

    preview_count = min(max(args.preview_samples, 0), len(records))
    if preview_count:
        logging.info("Generating %d preview completions using latest weights", preview_count)
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"{args.save_name}-preview"
        )
        sampling_params = tinker.types.SamplingParams(
            temperature=args.preview_temperature,
            top_p=args.preview_top_p,
            max_tokens=args.preview_max_tokens,
            logprobs=False,
        )
        bos_token_id = tokenizer.bos_token_id
        for idx, record in enumerate(records[:preview_count], start=1):
            prompt = record["prompt"]
            completion = record.get("completion", "")
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if bos_token_id is not None and (
                not prompt_tokens or prompt_tokens[0] != bos_token_id
            ):
                prompt_tokens = [bos_token_id] + prompt_tokens
            model_input = tinker.types.ModelInput.from_ints(prompt_tokens)
            result = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()
            if not getattr(result, "sequences", None):
                logging.warning("Preview %d failed: no sequences returned", idx)
                continue
            sequence = result.sequences[0]
            generated = tokenizer.decode(
                getattr(sequence, "tokens", []),
                skip_special_tokens=True,
            ).strip()
            logging.info("Preview %d prompt: %s", idx, prompt)
            logging.info("Preview %d target completion: %s", idx, completion)
            logging.info("Preview %d model completion: %s", idx, generated)

    checkpoint = training_client.save_state(name=args.save_name).result()
    logging.info(
        "Saved adapter '%s' to %s",
        args.save_name,
        getattr(checkpoint, "path", "(unknown path)"),
    )


if __name__ == "__main__":
    main()
