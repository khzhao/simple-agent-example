"""Generate 2048 SFT examples using an OpenAI teacher policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .env import apply_agent_move, check_game_finished, generate_game, render_board
from .openai_client import OpenAIChatModel
from .rollout import RESPONSE_PATTERN, SYSTEM_PROMPT, build_prompt, extract_move_xml
from .tinker_client import ChatMessage

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Track statistics during SFT generation."""

    total_examples: int = 0
    total_games: int = 0
    total_api_calls: int = 0
    total_attempts: int = 0
    format_failures: int = 0
    noop_failures: int = 0
    invalid_move_failures: int = 0
    api_errors: int = 0
    abandoned_games: int = 0
    completed_games: int = 0

    def log_summary(self) -> None:
        """Log comprehensive statistics summary."""
        logger.info("=" * 80)
        logger.info("SFT Generation Complete")
        logger.info("=" * 80)
        logger.info("Examples collected: %d", self.total_examples)
        logger.info("Games played: %d (completed: %d, abandoned: %d)",
                    self.total_games, self.completed_games, self.abandoned_games)
        logger.info("Total API calls: %d", self.total_api_calls)
        
        if self.total_examples > 0:
            avg_attempts = self.total_attempts / self.total_examples
            logger.info("Average attempts per example: %.2f", avg_attempts)
        
        logger.info("Failures - format: %d | noop: %d | invalid_move: %d | api_error: %d",
                    self.format_failures, self.noop_failures, 
                    self.invalid_move_failures, self.api_errors)
        logger.info("=" * 80)


@dataclass
class MoveAttemptResult:
    """Result of attempting to get a valid move from the teacher."""

    success: bool
    content: Optional[str] = None
    move_xml: Optional[str] = None
    attempts: int = 0
    format_failures: int = 0
    noop_failures: int = 0
    invalid_move_failures: int = 0
    api_errors: int = 0


def _write_record(handle, record: dict) -> None:
    """Write a single JSONL record to file."""
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def _build_teacher_prompt(board_view: str) -> str:
    """Build prompt specifically for teacher (without system prompt prefix)."""
    return (
        "Current board:\n"
        f"{board_view}\n"
        "Return your answer using exactly:\n"
        "<think>...</think>\n"
        "<move>direction</move>"
    )


async def _attempt_valid_move(
    teacher: OpenAIChatModel,
    game: dict,
    board_view: str,
    max_attempts: int = 10,
) -> MoveAttemptResult:
    """
    Attempt to get a valid move from the teacher with retry logic.
    
    This function encapsulates all retry logic for format errors, no-ops, and invalid moves.
    It preserves the random state when testing moves to ensure deterministic gameplay.
    """
    result = MoveAttemptResult(success=False)
    teacher_prompt = _build_teacher_prompt(board_view)
    
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        result.api_errors += 0  # Reset for this attempt
        
        # Build messages with system prompt
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=teacher_prompt),
        ]
        
        # Call teacher API
        try:
            reply = await teacher.sample_action(messages)
        except Exception as exc:
            result.api_errors += 1
            logger.warning("API call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            await asyncio.sleep(0.5 * attempt)  # Exponential backoff
            continue
        
        content = reply.content.strip()
        
        # Validate response format
        if not RESPONSE_PATTERN.fullmatch(content):
            result.format_failures += 1
            logger.debug("Invalid format (attempt %d/%d): %r", attempt, max_attempts, content[:100])
            teacher_prompt = (
                f"{teacher_prompt}\n\n"
                f"Previous response had invalid format. Must be: <think>...</think><move>direction</move>"
            )
            continue
        
        # Extract move XML
        try:
            move_xml = extract_move_xml(content)
        except ValueError as exc:
            result.invalid_move_failures += 1
            logger.debug("Failed to extract move (attempt %d/%d): %s", attempt, max_attempts, exc)
            teacher_prompt = f"{teacher_prompt}\n\nPrevious response missing valid <move> tag."
            continue
        
        # Test if move changes the board (preserve random state)
        rng_state = random.getstate()
        preview_game = {
            "id": game["id"],
            "board": [row[:] for row in game["board"]],
        }
        
        try:
            apply_agent_move(preview_game, move_xml)
        except ValueError as exc:
            random.setstate(rng_state)  # Restore state on failure
            error_message = str(exc).lower()
            
            if "did not change board" in error_message:
                result.noop_failures += 1
                logger.debug("No-op move (attempt %d/%d): %s", attempt, max_attempts, move_xml)
                teacher_prompt = (
                    f"{teacher_prompt}\n\n"
                    f"Your move {move_xml} did not change the board. Choose a different direction."
                )
            else:
                result.invalid_move_failures += 1
                logger.debug("Invalid move (attempt %d/%d): %s", attempt, max_attempts, exc)
                teacher_prompt = f"{teacher_prompt}\n\nInvalid move: {exc}"
            
            continue
        
        # Success! Restore random state (will be used when actually applying move)
        random.setstate(rng_state)
        result.success = True
        result.content = content
        result.move_xml = move_xml
        return result
    
    # Exhausted all attempts
    logger.warning("Failed to obtain valid move after %d attempts", max_attempts)
    return result


async def _play_game_for_sft(
    teacher: OpenAIChatModel,
    stats: GenerationStats,
    target_examples: int,
    file_handle,
    log_every: int,
) -> int:
    """
    Play a single game and collect SFT examples.
    
    Returns the number of examples collected from this game.
    """
    game = generate_game()
    stats.total_games += 1
    game_id = game["id"]
    examples_from_game = 0
    
    logger.debug("Starting game %d (id=%s)", stats.total_games, game_id)
    
    while stats.total_examples < target_examples:
        board_view = render_board(game)
        
        # Get a valid move from teacher
        move_result = await _attempt_valid_move(teacher, game, board_view)
        
        # Update statistics
        stats.total_api_calls += move_result.attempts
        stats.total_attempts += move_result.attempts
        stats.format_failures += move_result.format_failures
        stats.noop_failures += move_result.noop_failures
        stats.invalid_move_failures += move_result.invalid_move_failures
        stats.api_errors += move_result.api_errors
        
        # Check if we got a valid move
        if not move_result.success:
            logger.warning("Abandoning game %d (id=%s) after %d steps - no valid move",
                          stats.total_games, game_id, examples_from_game)
            stats.abandoned_games += 1
            return examples_from_game
        
        # Create and write SFT record
        # Use build_prompt for the student-facing prompt (includes system prompt)
        student_prompt = build_prompt(board_view)
        record = {
            "prompt": student_prompt,
            "completion": move_result.content,
        }
        _write_record(file_handle, record)
        
        stats.total_examples += 1
        examples_from_game += 1
        
        # Log progress
        if stats.total_examples % log_every == 0:
            logger.info("Collected %d / %d examples (game %d, step %d)",
                       stats.total_examples, target_examples,
                       stats.total_games, examples_from_game)
        
        logger.debug(
            "Example %d accepted (game %d step %d) | attempts=%d | format_fail=%d | noop_fail=%d | invalid_fail=%d",
            stats.total_examples, stats.total_games, examples_from_game,
            move_result.attempts, move_result.format_failures,
            move_result.noop_failures, move_result.invalid_move_failures,
        )
        
        # Apply the move to advance game state
        apply_agent_move(game, move_result.move_xml)
        
        # Check if game is finished
        if check_game_finished(game):
            logger.debug("Game %d (id=%s) completed after %d steps",
                        stats.total_games, game_id, examples_from_game)
            stats.completed_games += 1
            return examples_from_game
    
    # Reached target examples mid-game
    logger.debug("Game %d (id=%s) interrupted at %d steps (target reached)",
                stats.total_games, game_id, examples_from_game)
    stats.completed_games += 1
    return examples_from_game


async def _generate_examples_async(args: argparse.Namespace) -> None:
    """Main async function to generate SFT dataset."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Initialize teacher model
    teacher = OpenAIChatModel(
        model=args.teacher_model,
        api_key=args.openai_api_key,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )
    
    # Prepare output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = GenerationStats()
    
    logger.info("Starting SFT generation: target=%d examples, model=%s",
                args.examples, args.teacher_model)
    
    # Generate examples
    with output_path.open("w", encoding="utf-8") as handle:
        while stats.total_examples < args.examples:
            await _play_game_for_sft(
                teacher=teacher,
                stats=stats,
                target_examples=args.examples,
                file_handle=handle,
                log_every=args.log_every,
            )
    
    # Log final statistics
    stats.log_summary()


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
