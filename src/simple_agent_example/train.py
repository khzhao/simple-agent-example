"""Async training entrypoint inspired by the OpenPipe ART 2048 example."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from .rollout import rollout
from .tinker_client import (
    TinkerTrainableModel,
    gather_trajectory_groups,
    environment_seed,
)

DEFAULT_TRAIN_STEPS = 40
DEFAULT_SIMULTANEOUS_GAMES = 18
DEFAULT_LEARNING_RATE = 1e-5


async def train(args: argparse.Namespace) -> None:
    """Run PPO-style online LoRA updates."""
    load_dotenv()
    environment_seed(args.seed)

    model = TinkerTrainableModel(
        name=args.model_name,
        project=args.project,
        base_model=args.base_model,
        api_key=args.api_key,
        rank=args.rank,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
    )

    if args.resume_from:
        logging.info("Loading checkpoint %s", args.resume_from)
        await model.load_checkpoint(args.resume_from)
    else:
        await model.refresh_sampling_client()

    for step in range(args.start_step, args.train_steps):
        logging.info("Collecting trajectories for step %s", step)

        groups = await gather_trajectory_groups(
            [
                [
                    rollout(
                        model,
                        step,
                        is_validation=False,
                        verbose=args.verbose_rollouts,
                    )
                    for _ in range(args.simultaneous_games)
                ]
            ],
            after_each=model.score_group if args.enable_ruler else None,
        )

        if not groups:
            logging.warning("No trajectories gathered for step %s; skipping update", step)
            continue

        logging.info("Training on %s trajectory group(s)", len(groups))
        await model.train(groups, learning_rate=args.learning_rate)

        if args.save_checkpoints:
            await model.save_checkpoint(name=f"step-{step:04d}")

    if args.save_checkpoints:
        logging.info("Final checkpoint save")
        await model.save_checkpoint(name="final")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tinker LoRA policy on 2048.")
    parser.add_argument("--model-name", default=os.getenv("TINKER_MODEL_NAME", "tutorial-001"))
    parser.add_argument("--project", default=os.getenv("TINKER_PROJECT_NAME", "2048"))
    parser.add_argument(
        "--base-model",
        default=os.getenv("TINKER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--simultaneous-games", type=int, default=DEFAULT_SIMULTANEOUS_GAMES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--enable-ruler", action="store_true")
    parser.add_argument("--verbose-rollouts", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
