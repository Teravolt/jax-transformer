"""
Multi-headed Self-Attention
"""

import argparse
import jax.numpy as jnp

from jax import random
from jax.nn import softmax

from nn.utils import BATCH_FOWARD_DENSE
from nn.utils import init_dense

def init_self_attention(rng_key: random.PRNGKey,
                        input_dim: int,
                        config: argparse.Namespace):
    """
    Initialize self-attention parameters

    :param rng_key: Random generator key
    :param input_dim: Input dimension
    :param config: Config

    :returns: Self-attention parameters:
    """

    d_model = config.d_model
    num_heads = config.num_heads

    assert d_model % num_heads == 0

    head_size = d_model // num_heads
    config.head_size = head_size

    rng_key, *keys = random.split(rng_key, 5)

    _, wq = init_dense(keys[0], [input_dim, d_model])
    _, wk = init_dense(keys[1], [input_dim, d_model])
    _, wv = init_dense(keys[2], [input_dim, d_model])
    _, concat_layer = init_dense(keys[3], [d_model, input_dim])

    parameters = {
        'wq': wq,
        'wk': wk,
        'wv': wv,
        'concat': concat_layer
        }

    return rng_key, parameters

def __self_atttention(query: jnp.array, key: jnp.array, value: jnp.array,
                      mask, is_causal=False):
    """
    Core of self-attention

    :param query: Query with dims [batch_size, num_heads, seq_len, head_size]
    :param key: Key with dims [batch_size, num_heads, seq_len, head_size]
    :param value: Value with dims [batch_size, num_heads, seq_len, head_size]
    :param mask: Input mask of dimensions [batch_size, seq_len, 1]
    :param is_causal: Prevents current token from attending to future tokens (default=False)

    :returns: Results of self-attention with dimensions 
        [batch_size, num_heads, seq_length, head_size],
        attention weights of dimensions [batch_size, num_heads, seq_len, seq_len]  
    """

    # attention_matrix -> (batch_size, num_heads, seq_length, seq_length)
    attention_matrix = jnp.matmul(query, jnp.transpose(key, axes=[0, 1, 3, 2]))
    attention_matrix = attention_matrix / jnp.sqrt(key.shape[-1])

    mask_matrix = None

    if is_causal:
        mask_matrix = jnp.triu(jnp.ones(shape=attention_matrix.shape))

    if mask is not None:
        _mask = jnp.logical_not(mask).astype(dtype=jnp.float32)
        # seq_mask_matrix -> [batch_size, seq_len, seq_len]
        seq_mask_matrix = jnp.matmul(_mask, jnp.transpose(_mask, axes=[0, 2, 1]))

        # seq_mask_matrix -> [batch_size, num_heads, seq_len, seq_len]
        seq_mask_matrix = jnp.expand_dims(seq_mask_matrix, axis=1)
        seq_mask_matrix = jnp.tile(seq_mask_matrix, [1, query.shape[1], 1, 1])

        # Hadamard product
        mask_matrix = seq_mask_matrix if mask_matrix is None \
            else jnp.multiply(seq_mask_matrix, mask_matrix)

    if mask_matrix is not None:
        # We multiply the mask_matrix by -1e8 so that softmax
        # turns those that are masked to 0
        attention_matrix += (mask_matrix * -1e8)

    attention_weights = softmax(attention_matrix, axis=-1)
    scaled_attention = jnp.matmul(attention_matrix, value)

    print(f"Attention weight shape: {attention_weights.shape}")
    print(f"Attention scaling shape: {scaled_attention.shape}")

    return scaled_attention, attention_weights

def __split_heads(data: jnp.array, num_heads: int, head_size: int):
    """
    Split feature dimensions by `num_heads`

    :param data: Input data with dimensions [batch_size, seq_len, dims]
    :param num_heads: Number of self-attention heads
    :param head_size: Size of self-attention heads

    :returns: Input split into `num_heads` with dimensions
        [batch_size, num_heads, seq_len, head_size]
    """

    _data = jnp.reshape(data, (data.shape[0], -1, num_heads, head_size))
    return jnp.transpose(_data, axes=[0, 2, 1, 3])

def __project_and_split_qkv(parameters, query: jnp.array, key: jnp.array,
                            value: jnp.array, num_heads: int,
                            head_size: int):
    """
    Project query, key, and value and split by `num_heads` number of heads

    :param parameters: Query, key, value projection parameters
    :param query: Query with dimensions [batch_size, seq_len, dims]
    :param key: Key with dimensions [batch_size, seq_len, dims]
    :param value: Value with dimensions [batch_size, seq_len, dims]
    :param num_heads: Number of self-attention heads
    :param head_size: Size of self-attention heads

    :returns: Query, key, value split by heads with dimensions
         [batch_size, num_heads, seq_len, head_size]
    """

    _query = BATCH_FOWARD_DENSE(parameters['wq'], query)
    _key = BATCH_FOWARD_DENSE(parameters['wk'], key)
    _value = BATCH_FOWARD_DENSE(parameters['wv'], value)

    __query = __split_heads(_query, num_heads, head_size)
    __key = __split_heads(_key, num_heads, head_size)
    __value = __split_heads(_value, num_heads, head_size)

    return __query, __key, __value

def attention_forward(parameters, query: jnp.array, key: jnp.array,
                      value: jnp.array, config: argparse.Namespace,
                      mask=None, is_causal=False):
    """
    Self-attention forward pass

    :param parameters: Self-attention parameters
    :param query: Query of dimensions [batch_size, seq_len, dims]
    :param key: Key of dimensions [batch_size, seq_len, dims]
    :param value: Value of dimensions [batch_size, seq_len, dims]
    :param config: Config
    :param mask: Input mask of dimensions [batch_size, seq_len, 1]
    :param is_causal: Prevents current token from attending to future tokens (default=False)

    :returns: Results of self-attention with dimensions [batch_size, seq_len, dims],
        attention weights of dimensions [batch_size, num_heads, seq_len, seq_len]    
    """

    d_model = config.d_model
    num_heads = config.num_heads

    if d_model % num_heads != 0:
        raise ValueError(f"Model dimensions {d_model} "
                         f"should be divisible by the number of heads {num_heads}")

    head_size = d_model // num_heads

    _query, _key, _value = __project_and_split_qkv(parameters,
                                                   query, key, value,
                                                   num_heads, head_size)

    scaled_attention, attention_weights = __self_atttention(
        _query, _key, _value, mask, is_causal)

    # scaled_attention -> (batch_size, num_heads, seq_length, head_size)
    scaled_attention = jnp.transpose(scaled_attention, axes=[0, 2, 1, 3])
    scaled_attention = jnp.reshape(scaled_attention, (query.shape[0], -1, d_model))

    scaled_attention = BATCH_FOWARD_DENSE(parameters['concat'], scaled_attention)

    print(f"Attention results shape: {scaled_attention.shape}")

    return scaled_attention, attention_weights
