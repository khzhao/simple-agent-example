"""Async training entrypoint inspired by the OpenPipe ART 2048 example."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

import numpy as np
from dotenv import load_dotenv

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from .rollout import rollout
from .tinker_client import (TinkerTrainableModel, environment_seed,
                            gather_trajectory_groups)

DEFAULT_TRAIN_STEPS = 40
DEFAULT_SIMULTANEOUS_GAMES = 18
DEFAULT_LEARNING_RATE = 1e-4


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

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` or rerun with --no-wandb."
            )
        wandb_project = args.wandb_project or args.project
        wandb_config = vars(args).copy()
        wandb_run = wandb.init(
            project=wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    if args.resume_from:
        logging.info("Loading checkpoint %s", args.resume_from)
        resume_path = await model.load_checkpoint(args.resume_from)
        logging.info("Loaded weights from %s", resume_path)
    else:
        await model.refresh_sampling_client()

    try:
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
                logging.warning(
                    "No trajectories gathered for step %s; skipping update", step
                )
                continue

            if args.dump_trajectories and step == args.start_step:
                dumped = _dump_sample_trajectories(
                    groups,
                    args.dump_trajectories,
                    limit=args.dump_trajectories_count,
                )
            logging.info(
                "Dumped %d sample trajectories to %s",
                dumped,
                args.dump_trajectories,
            )

            if args.preview_samples and step == args.start_step:
                _log_sample_outputs(groups, args.preview_samples)

            logging.info("Training on %s trajectory group(s)", len(groups))
            train_stats = await model.train(groups, learning_rate=args.learning_rate)

            if wandb_run is not None:
                summary = _summarize_trajectories(groups)
                summary.update(
                    {
                        "training/num_datums": train_stats.get("num_datums", 0.0),
                        "training/trainable_groups": train_stats.get(
                            "trainable_groups", 0.0
                        ),
                        "training/submitted_groups": train_stats.get(
                            "submitted_groups", 0.0
                        ),
                        "training/learning_rate": args.learning_rate,
                    }
                )
                wandb_run.log(summary, step=step)

            if args.save_checkpoints:
                path = await model.save_checkpoint(name=f"step-{step:04d}")
                logging.info("Saved checkpoint to %s", path)

        if args.save_checkpoints:
            logging.info("Final checkpoint save")
            path = await model.save_checkpoint(name="final")
            logging.info("Final checkpoint stored at %s", path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _summarize_trajectories(groups: list) -> dict[str, float]:
    trajectories = [trajectory for group in groups for trajectory in group]
    if not trajectories:
        return {}

    rewards = np.array([trajectory.reward for trajectory in trajectories])
    win_flags = np.array(
        [
            float(bool(trajectory.metrics.get("win", False)))
            for trajectory in trajectories
        ]
    )
    moves = [
        trajectory.metrics.get("num_moves")
        for trajectory in trajectories
        if trajectory.metrics.get("num_moves") is not None
    ]
    max_tiles = [
        trajectory.metrics.get("max_value")
        for trajectory in trajectories
        if trajectory.metrics.get("max_value") is not None
    ]
    board_values = [
        trajectory.metrics.get("board_value")
        for trajectory in trajectories
        if trajectory.metrics.get("board_value") is not None
    ]
    invalids = [
        trajectory.metrics.get("invalid_move", 0) for trajectory in trajectories
    ]

    summary: dict[str, float] = {
        "reward/mean": float(rewards.mean()),
        "reward/std": float(rewards.std()),
        "reward/min": float(rewards.min()),
        "reward/max": float(rewards.max()),
        "rollouts/count": float(len(trajectories)),
        "win/rate": float(win_flags.mean() if win_flags.size else 0.0),
        "invalid/frequency": (
            float(np.array(invalids, dtype=float).mean()) if invalids else 0.0
        ),
    }

    if moves:
        moves_array = np.array(moves, dtype=float)
        summary["moves/mean"] = float(moves_array.mean())
        summary["moves/std"] = float(moves_array.std())
    if max_tiles:
        tiles_array = np.array(max_tiles, dtype=float)
        summary["max_tile/mean"] = float(tiles_array.mean())
        summary["max_tile/max"] = float(tiles_array.max())
    if board_values:
        board_array = np.array(board_values, dtype=float)
        summary["board_value/mean"] = float(board_array.mean())
        summary["board_value/max"] = float(board_array.max())

    return summary


def _dump_sample_trajectories(groups, path: str, limit: int = 3) -> int:
    samples = []
    for group in groups:
        for trajectory in group:
            samples.append(_trajectory_to_dict(trajectory))
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as f:
        json.dump(samples, f, indent=2)

    return len(samples)


def _trajectory_to_dict(trajectory) -> dict:
    return {
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "metadata": trajectory.metadata,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in trajectory.messages
        ],
        "steps": [
            {
                "prompt_text": step.prompt_text,
                "response_text": step.response_text,
                "reward": step.reward,
                "done": step.done,
            }
            for step in trajectory.steps
        ],
    }


def _log_sample_outputs(groups, limit: int) -> None:
    count = 0
    for group in groups:
        for trajectory in group:
            first_board = next(
                (m.content for m in trajectory.messages if m.role == "user"),
                "",
            )
            last_reply = next(
                (
                    m.content
                    for m in reversed(trajectory.messages)
                    if m.role == "assistant"
                ),
                "",
            )
            logging.info(
                "Preview trajectory %d | reward=%.3f | invalid=%s",
                count + 1,
                trajectory.reward,
                bool(trajectory.metrics.get("invalid_move")),
            )
            if first_board:
                logging.info("Board:\n%s", first_board)
            if last_reply:
                logging.info("Assistant:\n%s", last_reply)
            count += 1
            if count >= limit:
                return

    logging.info("Only %d trajectories available for preview", count)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tinker LoRA policy on 2048.")
    parser.add_argument(
        "--model-name", default=os.getenv("TINKER_MODEL_NAME", "tutorial-001")
    )
    parser.add_argument("--project", default=os.getenv("TINKER_PROJECT_NAME", "2048"))
    parser.add_argument(
        "--base-model",
        default=os.getenv("TINKER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--simultaneous-games", type=int, default=DEFAULT_SIMULTANEOUS_GAMES
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--dump-trajectories")
    parser.add_argument("--dump-trajectories-count", type=int, default=3)
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=0,
        help="Log up to this many trajectory responses after the first gather.",
    )
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
