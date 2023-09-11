"""
Main function
"""

import argparse

from jax import random

import numpy as np

from nn.transformer import init_transformer

def get_config():
    """
    Get config
    """

    parser = argparse.ArgumentParser('JAX Implementation of a Transformer')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--d_model', type=int, default=768, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads')
    parser.add_argument('--num_encoders', type=int, default=1,
                        help='Number of Transformer encoders')
    parser.add_argument('--num_decoders', type=int, default=1,
                        help='Number of Transformer decoders')

    config = parser.parse_args()

    return config

def main():
    """
    Main function
    """

    config = get_config()

    main_key = random.PRNGKey(config.seed)
    main_key, *subkeys = random.split(main_key, num=3)
    model_init_key, data_key = subkeys

    # Input shape: [batch_size, encoder_sequence, d_model]
    input_data = random.uniform(data_key,
                                (config.batch_size, 60, config.d_model))
    input_mask = None

    print(f"Before init dense: {model_init_key}")
    _, model = init_transformer(model_init_key, config)
    print(f"After init dense: {model_init_key}")

    input_mask = np.ones((config.batch_size, 60, 1))
    for i in range(config.batch_size):
        data_key, subkey = random.split(data_key, num=2)
        index = random.randint(subkey, (1, 1), 0, input_mask.shape[1])[0][0]
        for j in range(index, input_mask.shape[1]):
            input_mask[i, j, :] = 0

    print("Training model...")
    # train(model, (input_data, input_data, input_mask), config)

    print("Training complete! Saving model!")
    # TODO: Save model


if __name__ == "__main__":
    main()
