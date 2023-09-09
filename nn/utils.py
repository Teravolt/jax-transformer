"""
Assorted utilities
"""

import jax.numpy as jnp

from jax import random
from jax import vmap

EPSILON = 1e-8

def init_dense(rng_key: random.PRNGKey, layer_sizes: list[int]):
    """
    Initialize an N-layer dense network

    :param rng_key: Random generator key
    :param layer_sizes: 

    :returns layer parameters:
    """

    parameters = {}

    rng_key, *layer_keys = random.split(rng_key, num=len(layer_sizes))

    for i, layer_key in enumerate(layer_keys):
        # print(f"i: {i} - Size: ({layer_sizes[i]}, {layer_sizes[i+1]})")
        _, *_keys = random.split(layer_key, num=3)
        weight_key, bias_key = _keys

        weights = random.normal(weight_key, (layer_sizes[i], layer_sizes[i+1]))
        biases = random.normal(bias_key, (layer_sizes[i+1], ))
        parameters[f'layer-{i}'] = (weights, biases)

    return rng_key, parameters

def init_layer_norm(rng_key: random.PRNGKey, dimensions: int):
    """
    Initialize layer normalization bias and gain

    :param rng_key: Random generator key
    :param input_dim: Input dimensions

    :returns layer parameters:
    """

    rng_key, *_keys = random.split(rng_key, num=3)
    weight_key, bias_key = _keys

    scale = random.normal(weight_key, (dimensions, ))
    offset = random.normal(bias_key, (dimensions, ))

    parameters = {'layer-norm': (scale, offset)}

    return rng_key, parameters

def __forward_dense(parameters: dict, input_data):
    """
    Forward pass over dense layer

    :param input_data: Input data
    :returns: output of dense layer
    """

    _x = input_data

    for _, param in parameters.items():
        _x = jnp.dot(_x, param[0]) + param[1]

    return _x

def __forward_layer_norm(parameters: dict, input_data):
    """
    Layer normalization

    :param parameters: Layer norm parameters
    :param input_data: Input data

    :returns layer norm output:
    """

    mean = jnp.mean(input_data, axis=-1, keepdims=True)
    variance = jnp.var(input_data, axis=-1, keepdims=True)
    output = (input_data - mean) / jnp.sqrt(variance - EPSILON)

    # print(params["layer-norm"][0].shape,  params["layer-norm"][1].shape)
    # print("Layer norm output shape: ", layer_norm_output.shape)

    output = jnp.dot(output, parameters['layer-norm'][0]) \
        + parameters['layer-norm'][1]
    # print("Final layer norm output shape: ", layer_norm_output.shape)

    return output

# def __seq_forward_layer_norm(parameters: dict, input_data):
#     """
#     Layer normalization

#     Notes:
#         This does not implement bias and gain

#     Args:
#         params (dict[str, (jax.array, jax.array)]): Layer norm parameters
#         input_data (jax.array): Input data with dims (batch_size, seq_len, dims)

#     Returns:
#         layer_norm_output (jax.array): Layer norm output with dims (batch_size, seq_len, dims)
#     """
#     return SEQUENTIAL_LAYER_NORM(params, input_data)

# BATCH VMAP FUNCTIONS
BATCH_FOWARD_DENSE = vmap(__forward_dense, in_axes=(None, 0), out_axes=0)
BATCH_LAYER_NORM = vmap(__forward_layer_norm, in_axes=(None, 0), out_axes=0)

# SEQUENTIAL_LAYER_NORM = vmap(__forward_layer_norm, in_axes=(None, 0), out_axes=0)
