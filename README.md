# 2048 RL Agent with Tinker LORA API

Train a reinforcement learning agent to play the game 2048 using Tinker's LORA fine-tuning API.

## Overview

This project implements a complete RL pipeline for training a language model to play 2048:

- **Game Environment**: Gymnasium-compatible 2048 environment
- **Text-based State Representation**: Game state encoded as natural language
- **LORA Training**: Efficient fine-tuning using Tinker's API
- **Reinforcement Learning**: Policy gradient approach with self-play
- **Logging**: WandB integration for metrics tracking
- **Inference**: Play and visualize trained models

## Installation

1. Clone the repository:
```bash
cd /Users/kzhao/github/simple-agent-example
```

2. Install dependencies (already in pyproject.toml):
```bash
uv sync
```

Dependencies include:
- `tinker>=0.2.2` - LORA training API
- `gymnasium>=1.2.1` - RL environment interface
- `torch>=2.9.0` - Deep learning framework
- `transformers>=4.57.1` - Model loading
- `wandb>=0.22.2` - Experiment tracking
- `numpy>=2.3.4` - Numerical operations

## Setup

### 1. Get Tinker API Key

Sign up for Tinker API access at [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker/)

Set your API key:
```bash
export TINKER_API_KEY="your-api-key-here"
```

### 2. (Optional) Setup WandB

For experiment tracking:
```bash
wandb login
```

## Usage

### Training

Basic training:
```bash
python train.py --episodes 1000 --batch-size 8
```

With custom parameters:
```bash
python train.py \
    --episodes 2000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
    --lora-rank 16 \
    --experiment-name "2048-experiment-1" \
    --eval-interval 50 \
    --save-interval 100
```

Using a config file:
```bash
python train.py --config example_config.json
```

Disable WandB:
```bash
python train.py --no-wandb --episodes 500
```

### Playing with Trained Model

Play a single game with a trained checkpoint:
```bash
python play.py --checkpoint checkpoint-ep000100
```

Play multiple games for evaluation:
```bash
python play.py \
    --checkpoint checkpoint-ep000500 \
    --num-games 10 \
    --delay 0.2
```

Fast evaluation (no rendering):
```bash
python play.py \
    --checkpoint checkpoint-ep000500 \
    --num-games 100 \
    --no-render
```

**Note**: The `--checkpoint` argument expects a checkpoint **name/label**, not a file path. Tinker stores checkpoints in the cloud with simple alphanumeric labels.

## Configuration

Key configuration parameters in `TrainingConfig`:

### Tinker API Settings
- `base_model`: Base model for LORA (default: "Qwen/Qwen2.5-0.5B-Instruct")
- `rank`: LoRA rank (default: 8)

### Training Hyperparameters
- `learning_rate`: Learning rate (default: 1e-4)
- `num_episodes`: Number of training episodes (default: 1000)
- `batch_size`: Episodes before update (default: 8)
- `gamma`: Discount factor (default: 0.99)

### Reward Shaping
- `score_reward_scale`: Scale for score increases (default: 1.0)
- `invalid_move_penalty`: Penalty for invalid moves (default: -10.0)
- `game_over_penalty`: Penalty for game over (default: -50.0)
- `valid_move_bonus`: Bonus for valid moves (default: 1.0)

### Evaluation
- `eval_interval`: Evaluate every N episodes (default: 25)
- `eval_episodes`: Number of eval episodes (default: 5)
- `save_interval`: Save checkpoint every N episodes (default: 50)

See `example_config.json` for a complete configuration example.

## How It Works

### 1. Game Environment

The 2048 game is implemented as a Gymnasium environment (`Game2048Env`):
- **Action Space**: Discrete(4) - up, down, left, right
- **Observation Space**: 4x4 grid of tile values
- **Rewards**: Score increases, valid move bonuses, penalties for invalid moves/game over

### 2. Text Encoding

Game states are converted to natural language for the LLM:
```
You are playing the 2048 game.
Current score: 128
Moves made: 15

Board state:
  Row 1: [2, 4, 8, 16]
  Row 2: [empty, 2, 4, 8]
  Row 3: [empty, empty, 2, 4]
  Row 4: [empty, empty, empty, 2]

Largest tile: 16
Empty cells: 7

Choose your next move from: up, down, left, right
```

The model responds with an action (e.g., "left"), which is parsed and executed.

### 3. RL Training Loop

The training process:

1. **Episode Collection**: Model plays games, collecting state-action-reward trajectories
2. **Return Computation**: Calculate discounted returns for each action
3. **Batch Update**: When batch is full, update LORA weights using Tinker API
4. **Evaluation**: Periodically evaluate performance on test games
5. **Checkpointing**: Save model checkpoints at intervals

The trainer uses a policy gradient approach where the model learns from reward feedback.

### 4. LORA Fine-tuning with Tinker

Tinker's API handles:
- Distributed training infrastructure
- Efficient LORA parameter updates
- Model checkpointing and state management
- Sampling from the fine-tuned model via SamplingClient

The implementation uses Tinker's async API pattern:
- `create_lora_training_client_async()` - Initialize training client
- `forward_backward_async()` - Compute gradients with loss function
- `optim_step_async()` - Apply optimizer step
- `save_state_async()` / `load_state_async()` - Checkpoint management
- `create_sampling_client()` - Create client for inference
- `sample_async()` - Generate completions from the model

Note: The trainer uses async/await throughout and wraps the async training loop in `asyncio.run()` for the synchronous `train()` method.

## Monitoring Training

### WandB Dashboard

When WandB is enabled, you can track:
- Episode rewards and scores
- Max tile achieved
- Win rate (reaching 2048)
- Average moves per episode
- Evaluation metrics

### Console Output

Training progress is printed to console:
```
Episode 50/1000
  Score: 512
  Max Tile: 64
  Reward: 45.00
  Moves: 87
  Avg Score (last 100): 324.50
  Win Rate: 0.00%
```

## Tips for Better Performance

1. **Reward Shaping**: Adjust reward parameters to encourage desired behaviors
2. **Longer Training**: The agent improves significantly with more episodes (2000+)
3. **Batch Size**: Larger batches can stabilize training but slow it down
4. **Model Size**: Larger base models may perform better but cost more to train
5. **Temperature**: Lower temperature during evaluation for more deterministic play

## Troubleshooting

### Tinker API Errors
- Ensure `TINKER_API_KEY` is set correctly
- Check you have API credits available
- Verify base model name is supported (Qwen3/Llama3 series)

### Training Issues
- Start with fewer episodes to test the pipeline
- Check WandB logs for unusual patterns
- Reduce batch size if running into memory issues

### Poor Performance
- The model may need more training episodes
- Try adjusting reward shaping parameters
- Consider using a larger base model
- Verify the text encoding is clear and consistent

## Example Results

After training for 1000+ episodes, the model should:
- Achieve scores of 1000+ regularly
- Reach 128/256 tiles consistently
- With extended training, reach 512/1024 tiles
- Eventually learn to reach 2048 (win condition)

## Architecture Details

### Text State Encoder (`TextStateEncoder`)
- Converts grid to readable text format
- Provides clear action prompts
- Includes game statistics

### Action Parser (`ActionParser`)
- Extracts actions from model output
- Handles various response formats
- Fallback to random on parse failure

### RL Trainer (`RLTrainer`)
- Manages episode collection
- Computes discounted returns
- Interfaces with Tinker API
- Handles checkpointing and evaluation

### Game Player (`GamePlayer`)
- Loads trained checkpoints
- Plays games with visualization
- Collects evaluation statistics

## Advanced Usage

### Custom Reward Functions

Modify reward shaping in `game_2048.py` step function:
```python
# Example: bonus for creating large tiles
if max_tile >= 512:
    reward += 100
```

### Different Base Models

Use different Tinker-supported models:
```bash
python train.py --base-model "meta-llama/Llama-3.1-8B"
```

### Extended Training

For best results, train for longer:
```bash
python train.py --episodes 5000 --batch-size 16 --save-interval 250
```

## Contributing

Feel free to:
- Improve reward shaping
- Add new state representations
- Implement other RL algorithms
- Optimize the training loop
- Add visualization features

## License

This project is provided as-is for educational and research purposes.

## References

- [Tinker API Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [2048 Game](https://play2048.co/)

## Citation

If you use this code in your research, please cite:
```
@software{2048_rl_tinker,
  title = {Simple Agent Example},
  author = {Kevin Zhao},
  year = {2025},
}
```
