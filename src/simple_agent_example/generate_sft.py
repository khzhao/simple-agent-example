"""Generate 2048 SFT examples using an OpenAI teacher policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from dotenv import load_dotenv

from .env import apply_agent_move, check_game_finished, generate_game, render_board
from .openai_client import OpenAIChatModel
from .rollout import (RESPONSE_PATTERN, SYSTEM_PROMPT, build_prompt,
                      extract_move_xml, _teacher_prompt)
from .tinker_client import ChatMessage

load_dotenv()


def _write_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


async def _generate_examples_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    teacher = OpenAIChatModel(
        model=args.teacher_model,
        api_key=args.openai_api_key,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    game_counter = 0

    with output_path.open("w", encoding="utf-8") as handle:
        while total_examples < args.examples:
            print(f"Generating game {game_counter} of {args.examples}")
            game = generate_game()
            game_counter += 1
            step = 0

            while total_examples < args.examples:
                if step % 10 == 0:
                    print(f"Generating step {step} of game {game_counter} of {args.examples} (total examples: {total_examples})")
                board_view = render_board(game)
                prompt = build_prompt(board_view)
                teacher_prompt = _teacher_prompt(board_view)
                attempts = 0
                move_applied = False
                game_finished = False

                while total_examples < args.examples:
                    attempts += 1
                    messages = [
                        ChatMessage(role="system", content=SYSTEM_PROMPT),
                        ChatMessage(role="user", content=teacher_prompt),
                    ]

                    try:
                        reply = await teacher.sample_action(messages)
                    except Exception as exc:  # pragma: no cover - network failure path
                        logging.warning(
                            "Teacher call failed on game %d step %d attempt %d: %s",
                            game_counter,
                            step,
                            attempts,
                            exc,
                        )
                        continue

                    content = reply.content.strip()
                    if not RESPONSE_PATTERN.fullmatch(content):
                        logging.debug(
                            "Malformed teacher output on game %d step %d attempt %d: %r",
                            game_counter,
                            step,
                            attempts,
                            content,
                        )
                        teacher_prompt = (
                            f"{teacher_prompt}\n\nThe previous response was invalid: {content!r}"
                        )
                        continue

                    move_xml = extract_move_xml(content)

                    rng_state = random.getstate()
                    preview_game = {
                        "id": game["id"],
                        "board": [row[:] for row in game["board"]],
                    }
                    try:
                        apply_agent_move(preview_game, move_xml)
                    except ValueError as exc:
                        random.setstate(rng_state)
                        message = str(exc).lower()
                        if "did not change board" in message:
                            logging.debug(
                                "Teacher produced no-op move on game %d step %d attempt %d",
                                game_counter,
                                step,
                                attempts,
                            )
                            teacher_prompt = (
                                f"{teacher_prompt}\n\nYour last move did not change the board. Try a different direction."
                            )
                            continue
                        logging.warning(
                            "Invalid teacher move on game %d step %d attempt %d: %s",
                            game_counter,
                            step,
                            attempts,
                            exc,
                        )
                        teacher_prompt = (
                            f"{teacher_prompt}\n\nThe previous move was invalid: {exc}"
                        )
                        continue
                    random.setstate(rng_state)

                    record = {"prompt": prompt, "completion": content}
                    _write_record(handle, record)
                    total_examples += 1
                    step += 1

                    if total_examples % args.log_every == 0:
                        logging.info(
                            "Collected %d / %d SFT examples",
                            total_examples,
                            args.examples,
                        )

                    apply_agent_move(game, move_xml)
                    move_applied = True

                    if check_game_finished(game):
                        game_finished = True
                        break

                    break  # Proceed to next board state
                else:
                    # Exiting attempts loop because quota reached
                    break

                if not move_applied:
                    # Could not obtain a valid move; try a new game.
                    break

                if game_finished:
                    break

        logging.info(
            "Finished SFT generation: %d examples across %d games",
            total_examples,
            game_counter,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SFT data for 2048 using an OpenAI teacher."
    )
    parser.add_argument(
        "--examples", type=int, default=1000, help="Number of SFT pairs to produce."
    )
    parser.add_argument(
        "--output",
        default="data/sft/gpt4o_generated_2048_sft.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--teacher-model",
        default="gpt-4o-mini",
        help="OpenAI model to use for teacher rollouts.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Explicit OpenAI API key (falls back to OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for teacher completions.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens per teacher completion.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N examples.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_generate_examples_async(args))


if __name__ == "__main__":
    main()
