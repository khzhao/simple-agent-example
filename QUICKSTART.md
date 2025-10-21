# Quick Start Guide

Get your 2048 RL agent training in 5 minutes!

## Prerequisites

- Python 3.12+
- Tinker API key (sign up at [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker/))

## Step 1: Verify Setup

Run the test suite to ensure everything is installed correctly:

```bash
uv run python test_setup.py
```

You should see all tests passing.

## Step 2: Set Tinker API Key

Export your Tinker API key:

```bash
export TINKER_API_KEY="your-api-key-here"
```

Or add it to your `.env` file:

```bash
echo "TINKER_API_KEY=your-api-key-here" >> .env
```

## Step 3: Run a Quick Test Training

Start with a short training run to verify everything works:

```bash
uv run python train.py --episodes 10 --batch-size 4 --no-wandb
```

This will train for just 10 episodes without WandB logging. You should see output like:

```
Creating LORA training client with base model: Qwen/Qwen2.5-0.5B-Instruct
Starting training...

Episode 1/10
  Score: 256
  Max Tile: 32
  Reward: 45.00
  ...
```

## Step 4: Full Training Run

Once verified, start a full training run:

```bash
uv run python train.py --episodes 1000 --batch-size 8
```

With WandB enabled (recommended):

```bash
# First login to WandB
uv run wandb login

# Then train
uv run python train.py --episodes 1000 --batch-size 8
```

## Step 5: Play with Trained Model

After training completes, play games with your trained model:

```bash
uv run python play.py --checkpoint checkpoints/checkpoint_ep1000.pt
```

Watch the model play:

```bash
uv run python play.py \
  --checkpoint checkpoints/checkpoint_ep1000.pt \
  --num-games 5 \
  --delay 0.3
```

## Common Commands

### Training Commands

Quick test (10 episodes):
```bash
uv run python train.py --episodes 10 --no-wandb
```

Standard training:
```bash
uv run python train.py --episodes 1000 --batch-size 8
```

Extended training for better results:
```bash
uv run python train.py --episodes 2000 --batch-size 16
```

Custom configuration:
```bash
uv run python train.py --config example_config.json
```

### Inference Commands

Single game:
```bash
uv run python play.py --checkpoint checkpoints/checkpoint_ep500.pt
```

Evaluate with 10 games:
```bash
uv run python play.py \
  --checkpoint checkpoints/checkpoint_ep500.pt \
  --num-games 10 \
  --delay 0.2
```

Fast evaluation (no visualization):
```bash
uv run python play.py \
  --checkpoint checkpoints/checkpoint_ep500.pt \
  --num-games 100 \
  --no-render
```

## Project Structure

```
simple-agent-example/
├── src/simple_agent_example/
│   ├── envs/          # 2048 game environment
│   ├── models/        # Text encoding/parsing
│   ├── training/      # RL training loop
│   ├── inference/     # Model inference
│   └── utils/         # Logging utilities
├── train.py           # Main training script
├── play.py            # Inference script
├── test_setup.py      # Test suite
├── example_config.json # Example configuration
└── README.md          # Full documentation
```

## Key Configuration Options

Edit `example_config.json` or pass command-line arguments:

- `--episodes`: Number of training episodes (default: 1000)
- `--batch-size`: Episodes before weight update (default: 8)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--base-model`: Tinker base model (default: Qwen/Qwen2.5-0.5B-Instruct)
- `--lora-rank`: LORA rank for efficiency (default: 8)
- `--eval-interval`: Evaluate every N episodes (default: 25)
- `--save-interval`: Save checkpoint every N episodes (default: 50)

## Monitoring Training

### Console Output

Training progress is printed to the console:

```
Episode 50/1000
  Score: 512
  Max Tile: 64
  Reward: 45.00
  Moves: 87
  Avg Score (last 100): 324.50
  Win Rate: 0.00%
```

### WandB Dashboard

If WandB is enabled, monitor:
- Real-time training metrics
- Episode rewards and scores
- Max tiles achieved
- Win rate over time
- Model evaluation results

Access your dashboard at: https://wandb.ai/your-username/2048-tinker-rl

## Expected Results

After various training durations:

**100 episodes:**
- Average score: 200-400
- Max tile: 32-64
- Win rate: 0%

**500 episodes:**
- Average score: 500-800
- Max tile: 64-128
- Win rate: 0-1%

**1000+ episodes:**
- Average score: 800-1500
- Max tile: 128-256
- Win rate: 1-5%

**2000+ episodes:**
- Average score: 1500+
- Max tile: 256-512
- Win rate: 5-15%

Note: Results vary based on hyperparameters and random initialization.

## Troubleshooting

### Tinker API Issues

```
ValueError: TINKER_API_KEY not found
```

**Solution:** Set your API key:
```bash
export TINKER_API_KEY="your-key"
```

### Import Errors

```
ModuleNotFoundError: No module named 'tinker'
```

**Solution:** Run with `uv`:
```bash
uv run python train.py
```

Or activate the virtual environment:
```bash
source .venv/bin/activate
python train.py
```

### Poor Performance

If the model isn't learning well:

1. **Train longer:** Try 2000+ episodes
2. **Adjust rewards:** Edit reward values in `config.py`
3. **Try larger model:** Use a bigger base model
4. **Increase batch size:** More stable learning

### Memory Issues

If you run out of memory:

1. **Reduce batch size:** Use `--batch-size 4`
2. **Smaller model:** Use Qwen2.5-0.5B instead of larger models
3. **Lower LORA rank:** Use `--lora-rank 4`

## Next Steps

1. **Experiment with hyperparameters:** Adjust learning rate, batch size, rewards
2. **Try different base models:** Test Llama or larger Qwen models
3. **Implement improvements:** Add better reward shaping, curriculum learning
4. **Analyze performance:** Use WandB to track what works best

## Resources

- [Full README](README.md) - Complete documentation
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Gymnasium Docs](https://gymnasium.farama.org/)

## Getting Help

If you encounter issues:

1. Run `uv run python test_setup.py` to verify setup
2. Check the [README](README.md) for detailed information
3. Review Tinker API documentation
4. Check WandB logs for training issues

Happy training!
