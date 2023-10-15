"""
Main function
"""

import argparse

import jax
from jax import random
import jax.numpy as jnp

import optax

import numpy as np

from nn.transformer import init_transformer
from nn.transformer import transformer_forward

def get_config():
    """
    Get config

    :returns: Config
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

    parser.add_argument('--num_epochs', type=int, default=16, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='AdamW decay rate for first moment of past gradients')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='AdamW decay rate for second moment of past gradients')
    config = parser.parse_args()

    return config

def get_batch(dataset, batch_size):
    """
    Batch data

    :param dataset: Input, output, mask tuple, each of size
        [num_instances, seq_len, dims]
    :returns: Batched data of size [batch_size, seq_len, dims]
    """

    remainder = dataset[0].shape[0] % batch_size
    for i in range(0, dataset[0].shape[0] - remainder, batch_size):
        # print("Getting batch: {}:{}".format(i, i+batch_size))
        mask_data = None
        if dataset[2] is not None:
            mask_data = dataset[2][i:i+batch_size]
        yield dataset[0][i:i+batch_size], dataset[1][i:i+batch_size], mask_data

    # print("Getting final batch: {}:{}".format(data[0].shape[0]-remainder, data[0].shape[0]))
    if remainder != 0:
        mask_data = None
        if dataset[2] is not None:
            mask_data = dataset[2][-remainder:]
        yield dataset[0][-remainder:], dataset[1][-remainder:], mask_data

def compute_loss(model, config: dict, input_data: jnp.array,
                 target_data: jnp.array, input_mask: jnp.array):
    """
    Compute loss
    
    :param model: Model parameters
    :param config: Config
    :param input_data: Input data with dims [batch_size, seq_len, dims]
    :param target_data: Target data with dims [batch_size, seq_len, dims]
    :param input_mask: Mask data with dims [batch_size, seq_len, 1]

    :returns: loss
    """

    pred_data = transformer_forward(model, config, input_data, mask=input_mask)

    loss = optax.l2_loss(pred_data, target_data)
    return jnp.average(loss)

def train(model, dataset, config: argparse.Namespace):
    """
    Train model

    :param model: Transformer model
    :param dataset: Input, output, mask tuple, each of size
        [num_instances, seq_len, dims]
    :param config: Config

    :returns: Trained model
    """

    optimizer = optax.adamw(
        config.lr, config.b1, config.b2)
    opt_state = optimizer.init(model)

    for epoch in range(config.num_epochs):
        print(f"=========== Epoch: {epoch} ===========")
        for batch_idx, batch in enumerate(get_batch(dataset, config.batch_size)):
            print(f"-------- Batch: {batch_idx} --------")
            input_data, output_data, mask = batch
            value, grads = jax.value_and_grad(compute_loss)(
                model, config, input_data, output_data, mask)

            print(f"Loss: {value}")
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = optax.apply_updates(model, updates)
    return model

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
                                (32, 60, config.d_model))
    input_mask = None

    print(f"Before init dense: {model_init_key}")
    _, model = init_transformer(model_init_key, config)
    print(f"After init dense: {model_init_key}")

    input_mask = np.ones((32, 60, 1))
    for i in range(config.batch_size):
        data_key, subkey = random.split(data_key, num=2)
        index = random.randint(subkey, (1, 1), 0, input_mask.shape[1])[0][0]
        for j in range(index, input_mask.shape[1]):
            input_mask[i, j, :] = 0

    print("Training model...")
    model = train(model, (input_data, input_data, input_mask), config)

    print("Training complete! Saving model!")
    # TODO: Save model


if __name__ == "__main__":
    main()
