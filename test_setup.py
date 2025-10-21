#!/usr/bin/env python3
"""
Test script to verify the 2048 RL setup is working correctly.

This script tests all components without requiring Tinker API credentials.
"""

import sys
import numpy as np


def test_environment():
    """Test the 2048 game environment."""
    print("Testing 2048 game environment...")

    try:
        from simple_agent_example.envs import Game2048Env

        env = Game2048Env()
        obs, info = env.reset()

        assert obs.shape == (4, 4), "Observation shape should be 4x4"
        assert info["score"] == 0, "Initial score should be 0"
        assert np.sum(obs > 0) == 2, "Should have 2 initial tiles"

        # Test a move
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (4, 4), "Observation shape should remain 4x4"

        print("  ✓ Environment test passed")
        return True

    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def test_text_encoder():
    """Test the text state encoder."""
    print("Testing text encoder...")

    try:
        from simple_agent_example.models import TextStateEncoder, ActionParser

        encoder = TextStateEncoder()
        parser = ActionParser()

        # Test encoding
        grid = np.array([[2, 4, 0, 0],
                        [0, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        text = encoder.encode_state(grid, score=4, move_count=1)

        assert isinstance(text, str), "Encoded state should be a string"
        assert "2048" in text.lower(), "Text should mention 2048"
        assert "score" in text.lower(), "Text should mention score"

        # Test action parsing
        assert parser.parse_action("up") == 0, "Should parse 'up' correctly"
        assert parser.parse_action("down") == 1, "Should parse 'down' correctly"
        assert parser.parse_action("left") == 2, "Should parse 'left' correctly"
        assert parser.parse_action("right") == 3, "Should parse 'right' correctly"
        assert parser.parse_action("invalid") == -1, "Should return -1 for invalid"

        print("  ✓ Text encoder test passed")
        return True

    except Exception as e:
        print(f"  ✗ Text encoder test failed: {e}")
        return False


def test_config():
    """Test the training configuration."""
    print("Testing configuration...")

    try:
        from simple_agent_example.training import TrainingConfig

        config = TrainingConfig()

        assert config.num_episodes > 0, "num_episodes should be positive"
        assert config.batch_size > 0, "batch_size should be positive"
        assert 0 <= config.gamma <= 1, "gamma should be between 0 and 1"
        assert config.learning_rate > 0, "learning_rate should be positive"

        print("  ✓ Configuration test passed")
        return True

    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_game_play():
    """Test playing a few moves in the game."""
    print("Testing gameplay...")

    try:
        from simple_agent_example.envs import Game2048Env

        env = Game2048Env()
        obs, info = env.reset()

        moves_made = 0
        for _ in range(10):
            # Try all actions until one works
            for action in range(4):
                prev_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)

                if not np.array_equal(prev_obs, obs):
                    moves_made += 1
                    break

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        assert moves_made > 0, "Should be able to make at least one move"

        print(f"  ✓ Gameplay test passed (made {moves_made} moves)")
        return True

    except Exception as e:
        print(f"  ✗ Gameplay test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("Testing dependencies...")

    dependencies = [
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ]

    all_available = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} available")
        except ImportError:
            print(f"  ✗ {name} not available")
            all_available = False

    # Check optional dependencies
    try:
        import wandb
        print(f"  ✓ wandb available (optional)")
    except ImportError:
        print(f"  ⚠ wandb not available (optional, for logging)")

    try:
        import tinker
        print(f"  ✓ tinker available")
    except ImportError:
        print(f"  ⚠ tinker not available (required for training)")
        print(f"    Install with: pip install tinker")

    return all_available


def test_complete_episode():
    """Test playing a complete episode."""
    print("Testing complete episode...")

    try:
        from simple_agent_example.envs import Game2048Env
        from simple_agent_example.models import TextStateEncoder

        env = Game2048Env()
        encoder = TextStateEncoder()

        obs, info = env.reset()
        done = False
        moves = 0
        max_moves = 100

        while not done and moves < max_moves:
            # Get text representation
            text = encoder.encode_state(obs, info["score"], moves)
            assert len(text) > 0, "Text representation should not be empty"

            # Random action (since we don't have a trained model)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            moves += 1

        print(f"  ✓ Episode test passed")
        print(f"    Final score: {info['score']}")
        print(f"    Max tile: {info['max_tile']}")
        print(f"    Moves: {moves}")
        return True

    except Exception as e:
        print(f"  ✗ Episode test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("2048 RL SETUP TEST")
    print("=" * 60 + "\n")

    tests = [
        test_dependencies,
        test_environment,
        test_text_encoder,
        test_config,
        test_game_play,
        test_complete_episode,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nYour setup is ready! To start training:")
        print("  1. Set your Tinker API key: export TINKER_API_KEY='your-key'")
        print("  2. Run: python train.py --episodes 100")
        print("=" * 60 + "\n")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nPlease fix the failing tests before training.")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
