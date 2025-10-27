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
class WorkerStats:
    """Statistics for a single worker."""
    
    worker_id: int
    examples: int = 0
    games: int = 0
    api_calls: int = 0
    attempts: int = 0
    format_failures: int = 0
    noop_failures: int = 0
    invalid_move_failures: int = 0
    api_errors: int = 0
    abandoned_games: int = 0
    completed_games: int = 0


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

    @classmethod
    def from_worker_stats(cls, worker_stats: list[WorkerStats]) -> GenerationStats:
        """Aggregate statistics from multiple workers."""
        return cls(
            total_examples=sum(w.examples for w in worker_stats),
            total_games=sum(w.games for w in worker_stats),
            total_api_calls=sum(w.api_calls for w in worker_stats),
            total_attempts=sum(w.attempts for w in worker_stats),
            format_failures=sum(w.format_failures for w in worker_stats),
            noop_failures=sum(w.noop_failures for w in worker_stats),
            invalid_move_failures=sum(w.invalid_move_failures for w in worker_stats),
            api_errors=sum(w.api_errors for w in worker_stats),
            abandoned_games=sum(w.abandoned_games for w in worker_stats),
            completed_games=sum(w.completed_games for w in worker_stats),
        )

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
    Play a complete game and collect all SFT examples from it.
    
    Returns the number of examples collected from this game.
    Note: This always plays the complete game regardless of target.
    """
    game = generate_game()
    stats.total_games += 1
    game_id = game["id"]
    examples_from_game = 0
    
    logger.debug("Starting game %d (id=%s)", stats.total_games, game_id)
    
    # Play until game finishes (no mid-game stopping)
    while True:
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
            logger.info(
                "Game %d (id=%s) completed after %d steps | total examples: %d",
                stats.total_games,
                game_id,
                examples_from_game,
                stats.total_examples,
            )
            stats.completed_games += 1
            return examples_from_game


async def _play_single_game(
    teacher: OpenAIChatModel,
    file_handle,
) -> tuple[int, WorkerStats]:
    """
    Play a single complete game and collect SFT examples.
    
    Returns tuple of (examples_collected, game_stats).
    """
    game = generate_game()
    game_id = game["id"]
    examples_from_game = 0
    
    # Local stats for this game
    game_stats = WorkerStats(worker_id=0)
    game_stats.games = 1
    
    logger.debug("Starting game (id=%s)", game_id)
    
    while True:
        board_view = render_board(game)
        
        # Get a valid move from teacher
        move_result = await _attempt_valid_move(teacher, game, board_view)
        
        # Update statistics
        game_stats.api_calls += move_result.attempts
        game_stats.attempts += move_result.attempts
        game_stats.format_failures += move_result.format_failures
        game_stats.noop_failures += move_result.noop_failures
        game_stats.invalid_move_failures += move_result.invalid_move_failures
        game_stats.api_errors += move_result.api_errors
        
        # Check if we got a valid move
        if not move_result.success:
            logger.warning("Abandoning game (id=%s) after %d steps - no valid move",
                          game_id, examples_from_game)
            game_stats.abandoned_games = 1
            game_stats.examples = examples_from_game
            return examples_from_game, game_stats
        
        # Create and write SFT record
        student_prompt = build_prompt(board_view)
        record = {
            "prompt": student_prompt,
            "completion": move_result.content,
        }
        _write_record(file_handle, record)
        
        examples_from_game += 1
        
        logger.debug(
            "Example collected (game %s step %d) | attempts=%d | format_fail=%d | noop_fail=%d | invalid_fail=%d",
            game_id, examples_from_game,
            move_result.attempts, move_result.format_failures,
            move_result.noop_failures, move_result.invalid_move_failures,
        )
        
        # Apply the move to advance game state
        apply_agent_move(game, move_result.move_xml)
        
        # Check if game is finished
        if check_game_finished(game):
            logger.debug("Game (id=%s) completed after %d steps", game_id, examples_from_game)
            game_stats.completed_games = 1
            game_stats.examples = examples_from_game
            return examples_from_game, game_stats


async def _worker_generate_sft(
    worker_id: int,
    teacher: OpenAIChatModel,
    target_examples: int,
    output_path: Path,
    log_every: int,
) -> WorkerStats:
    """
    Worker function that generates SFT examples and writes to its own file.
    
    Each worker writes to a separate file: output_path.worker_{worker_id}.jsonl
    """
    worker_file = output_path.parent / f"{output_path.stem}.worker_{worker_id}.jsonl"
    stats = WorkerStats(worker_id=worker_id)
    
    logger.info("Worker %d starting | target=%d examples | output=%s",
                worker_id, target_examples, worker_file.name)
    
    with worker_file.open("w", encoding="utf-8") as handle:
        while stats.examples < target_examples:
            examples_collected, game_stats = await _play_single_game(teacher, handle)
            
            # Aggregate game stats into worker stats
            stats.examples += game_stats.examples
            stats.games += game_stats.games
            stats.api_calls += game_stats.api_calls
            stats.attempts += game_stats.attempts
            stats.format_failures += game_stats.format_failures
            stats.noop_failures += game_stats.noop_failures
            stats.invalid_move_failures += game_stats.invalid_move_failures
            stats.api_errors += game_stats.api_errors
            stats.abandoned_games += game_stats.abandoned_games
            stats.completed_games += game_stats.completed_games
            
            # Log progress
            if stats.examples > 0 and stats.examples % log_every == 0:
                logger.info("Worker %d: %d / %d examples collected",
                           worker_id, stats.examples, target_examples)
    
    logger.info("Worker %d complete: %d examples in %d games",
                worker_id, stats.examples, stats.games)
    return stats


async def _generate_examples_parallel(args: argparse.Namespace) -> None:
    """Generate SFT examples by continuously spawning game tasks until target is reached."""
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    
    logging.getLogger().setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    
    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    max_concurrent = args.workers
    target_examples = args.examples
    
    logger.info("Starting parallel SFT generation: target=%d examples, max_concurrent=%d, model=%s",
                target_examples, max_concurrent, args.teacher_model)
    logger.info("Strategy: Continuously spawn game tasks until target reached")
    
    # Shared state
    total_examples = 0
    game_counter = 0
    all_stats = []
    active_tasks = set()
    output_files = []
    
    async def play_and_track(game_id: int) -> tuple[int, WorkerStats]:
        """Play one complete game and return results."""
        teacher = OpenAIChatModel(
            model=args.teacher_model,
            api_key=args.openai_api_key,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
        )
        
        game_file = output_path.parent / f"{output_path.stem}.game_{game_id}.jsonl"
        output_files.append(game_file)
        
        with game_file.open("w", encoding="utf-8") as handle:
            examples, stats = await _play_single_game(teacher, handle)
        
        logger.info("Game %d complete: %d examples collected", game_id, examples)
        return examples, stats
    
    # Keep spawning games until we reach target
    while total_examples < target_examples:
        # Fill up to max_concurrent workers
        while len(active_tasks) < max_concurrent and total_examples < target_examples:
            game_counter += 1
            task = asyncio.create_task(play_and_track(game_counter))
            active_tasks.add(task)
            logger.debug("Spawned game task %d | active=%d | total_examples=%d/%d",
                        game_counter, len(active_tasks), total_examples, target_examples)
        
        # Wait for at least one game to complete
        if active_tasks:
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Process completed games
            for task in done:
                examples, stats = await task
                total_examples += examples
                all_stats.append(stats)
                active_tasks.remove(task)
                
                if total_examples % args.log_every == 0 or total_examples >= target_examples:
                    logger.info("Progress: %d / %d examples collected from %d games",
                               total_examples, target_examples, len(all_stats))
    
    # Wait for any remaining active tasks to complete
    if active_tasks:
        logger.info("Waiting for %d remaining game(s) to complete...", len(active_tasks))
        remaining_results = await asyncio.gather(*active_tasks)
        for examples, stats in remaining_results:
            total_examples += examples
            all_stats.append(stats)
    
    # Aggregate statistics
    total_stats = GenerationStats.from_worker_stats(all_stats)
    
    # Merge all game files into final output
    logger.info("Merging %d game files into %s", len(output_files), output_path.name)
    with output_path.open("w", encoding="utf-8") as outfile:
        for game_file in output_files:
            if game_file.exists():
                with game_file.open("r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                # Optionally delete game file after merging
                if not args.keep_worker_files:
                    game_file.unlink()
                    logger.debug("Deleted game file: %s", game_file.name)
    
    # Log final statistics
    logger.info("Target reached: %d / %d examples from %d games",
                total_stats.total_examples, target_examples, total_stats.total_games)
    total_stats.log_summary()


async def _generate_examples_async(args: argparse.Namespace) -> None:
    """Main async function to generate SFT dataset."""
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    
    # Set all loggers to DEBUG to capture everything
    logging.getLogger().setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    
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
    logger.info("Strategy: Play complete games sequentially until target reached")
    
    # Generate examples by playing complete games
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
    logger.info("Target reached: %d / %d examples from %d games",
                stats.total_examples, args.examples, stats.total_games)
    stats.log_summary()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SFT data for 2048 using an OpenAI teacher."
    )
    parser.add_argument(
        "--examples", 
        type=int, 
        default=1000, 
        help="Number of SFT examples to generate (spawns complete games until target reached)."
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
        help="Sampling temperature for teacher completions (use 1.0 for o1/o1-mini models).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens per teacher response.",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent games to run in parallel (1 = sequential).",
    )
    parser.add_argument(
        "--keep-worker-files",
        action="store_true",
        help="Keep individual game files after merging (useful for debugging).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Choose parallel or sequential generation based on workers count
    if args.workers > 1:
        asyncio.run(_generate_examples_parallel(args))
    else:
        asyncio.run(_generate_examples_async(args))


if __name__ == "__main__":
    main()
