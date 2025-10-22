import numpy as np
import tinker
from tinker import types

from simple_agent_example.envs.game_2048 import Game2048Env
from simple_agent_example.models.text_encoder import TextStateEncoder


def generate_sft_episode(max_steps=500):
    game = Game2048Env()
    obs, info = game.reset()
    done = False
    dataset = []
    move_count = 0
    while not done and move_count < max_steps:
        state_text = TextStateEncoder.encode_state(obs, info["score"], move_count)
        valid_actions = info["valid_actions"]

        action_int = np.random.choice(valid_actions)
        action_text = None
        if action_int == 0:
            action_text = "up"
        elif action_int == 1:
            action_text = "down"
        elif action_int == 2:
            action_text = "left"
        elif action_int == 3:
            action_text = "right"
        assert action_text is not None

        next_obs, _, terminated, truncated, next_info = game.step(action_int)
        done = terminated or truncated
        move_count += 1
        obs = next_obs
        info = next_info

        dataset.append(
            {
                "state": state_text,
                "action": action_text,
            }
        )
    return dataset


def process_example(example: dict, tokenizer) -> types.Datum:
    prompt = tokenizer.encode(example["state"], add_special_tokens=False)
    prompt_weights = [0] * len(prompt)

    completion = tokenizer.encode(example["action"] + "\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion)

    tokens = prompt + completion
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)

base_model = "Qwen/Qwen3-30B-A3B-Base"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

import json

with open("datasets.json", "r") as f:
    training_data = json.load(f)

processed_training_data = [
    process_example(example, tokenizer) for example in training_data
]
import numpy as np

for _ in range(100):
    fwdbwd_future = training_client.forward_backward(
        processed_training_data, "cross_entropy"
    )
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate(
        [output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [
            example.loss_fn_inputs["weights"].tolist()
            for example in processed_training_data
        ]
    )
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")

resume_path = training_client.save_state(name="post-sft-2048-nl-train-v0").result().path
print(resume_path)
