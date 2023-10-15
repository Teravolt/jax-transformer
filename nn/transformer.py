"""
Transformer Neural Network
"""

import argparse

from jax import random
import jax.numpy as jnp

from nn.self_attention import init_self_attention
from nn.self_attention import attention_forward

from nn.utils import init_dense
from nn.utils import init_layer_norm

from nn.utils import BATCH_FOWARD_DENSE
from nn.utils import BATCH_LAYER_NORM

def init_transformer_encoder(rng_key: random.PRNGKey,
                             config: argparse.Namespace):
    """
    Initialize Transformer Encoder

    :param rng_key: Random generator key
    :param config: Config

    :returns: encoder parameters
    """

    num_encoders = config.num_encoders
    d_model = config.d_model

    rng_key, *encoder_keys = random.split(rng_key,
                                          num=num_encoders+1)

    parameters = []
    for i in range(num_encoders):
        rng_key, *layer_keys = random.split(encoder_keys[i], num=5)

        _, attention = init_self_attention(layer_keys[0], d_model, config)

        _, linear = init_dense(layer_keys[1], [d_model, d_model])

        _, layer_norm_1 = init_layer_norm(layer_keys[2], d_model)
        _, layer_norm_2 = init_layer_norm(layer_keys[3], d_model)

        params = {
            'attention': attention,
            'linear': linear,
            'norm-1': layer_norm_1,
            'norm-2': layer_norm_2
            }

        parameters.append(params)

    return rng_key, parameters

def init_transformer_decoder(rng_key: random.PRNGKey,
                             config: argparse.Namespace):
    """
    Initialize Transformer Decoder

    :param rng_key: Random generator key
    :param config: Config

    :returns: decoder parameters
    """

    num_decoders = config.num_decoders
    d_model = config.d_model

    rng_key, *encoder_keys = random.split(rng_key,
                                          num=num_decoders+1)

    parameters = []
    for i in range(num_decoders):
        rng_key, *layer_keys = random.split(encoder_keys[i], num=6)

        _, attention = init_self_attention(layer_keys[0], d_model, config)
        _, masked_attention = init_self_attention(layer_keys[1], d_model, config)

        _, linear = init_dense(layer_keys[2], [d_model, d_model])

        _, layer_norm_1 = init_layer_norm(layer_keys[3], d_model)
        _, layer_norm_2 = init_layer_norm(layer_keys[4], d_model)

        params = {
            'attention': attention,
            'masked-attention': masked_attention,
            'linear': linear,
            'norm-1': layer_norm_1,
            'norm-2': layer_norm_2
            }

        parameters.append(params)

    return rng_key, parameters

def init_transformer(rng_key: random.PRNGKey,
                     config: argparse.Namespace):
    """
    Initialize Transformer

    :param rng_key: Random generator key
    :param config: Config

    :returns: transformer parameters
    """

    rng_key, encoder = init_transformer_encoder(rng_key, config)
    rng_key, decoder = init_transformer_decoder(rng_key, config)

    return rng_key, {'encoder': encoder, 'decoder': decoder}

## -------------------------- Forward Pass Functions --------------------------

def __transformer_encoder_forward(parameters: dict,
                                  input_data,
                                  mask,
                                  config: argparse.Namespace):
    """
    Transformer Encoder Forward Pass Helper

    :param parameters: Encoder parameters
    :param config: Config
    :param input_data: Input of dimensions [batch_size, seq_len, dims]
    :param mask: Input mask [batch_size, seq_len, 1] (default=None)

    :returns: encoder output with dimensions [batch_size, seq_len, dims]
    """

    print(f"[Encoder] Layer data shape: {input_data.shape}")

    attention_output, _ = attention_forward(
        parameters['attention'], input_data, input_data, input_data,
        config, mask)

    add_1 = jnp.add(attention_output, input_data)
    print(f"[Encoder] Add layer 1: {add_1.shape}")

    layer_norm_1 = BATCH_LAYER_NORM(parameters['norm-1'], add_1)

    linear_out = BATCH_FOWARD_DENSE(parameters['linear'], layer_norm_1)
    print(f"[Encoder] Linear Output: {linear_out.shape}")

    add_2 = jnp.add(layer_norm_1, linear_out)
    print(f"[Encoder] Add layer 2: {add_2.shape}")

    output = BATCH_LAYER_NORM(parameters['norm-2'], add_2)
    print(f"[Encoder] Layer Norm 2: {output.shape}")

    return output

def transformer_encoder_forward(model: dict,
                                input_data,
                                mask,
                                config: argparse.Namespace):
    """
    Transformer Encoder Forward Pass

    :param model: Model parameters
    :param input_data: Input of dimensions [batch_size, seq_len, dims]
    :param mask: Input mask [batch_size, seq_len, 1]
    :param config: Config

    :returns: output with dimensions [batch_size, seq_len, dims]
    """

    if mask is not None:
        print(f"[Transformer] Input data shape: {input_data.shape}, {mask.shape}")
    else:
        print(f"[Transformer] Input data shape: {input_data.shape}")

    output = input_data

    for layer in model:
        output = \
            __transformer_encoder_forward(layer, output, mask, config)

    return output

# def transformer_decoder_forward(params: dict, encoder_input: jax.numpy,
#                                 input_data: jax.numpy, mask=None):
#     """Transformer decoder forward pass"""

#     print("Input data shape: {}".format(input_data.shape))

#     cur_layer_input = input_data
#     for i in range(len(params)):
#         print("[Decoder] Input data shape: {}".format(cur_layer_input.shape))

#         attn_output, _ = attention_forward(
#             params[f"masked-attn-{i}"], input_data, input_data,
#             input_data, mask, True)
#         add_1_output = np.add(attn_output, input_data)
#         print("[Decoder] Add layer 1: {}".format(add_1_output.shape))

#         layer_norm_1_output = jax.nn.normalize(add_1_output, axis=2)
#         print("[Decoder] Layer Norm 1: {}".format(layer_norm_1_output.shape))

#         attn_output_2, _ = attention_forward(
#             params[f"attn-{i}"], encoder_input,
#             encoder_input, layer_norm_1_output, mask)
#         add_2_output = np.add(layer_norm_1_output, attn_output_2)
#         print("[Decoder] Add layer 2: {}".format(add_2_output.shape))
#         layer_norm_2_output = jax.nn.normalize(add_2_output, axis=2)
#         print("[Decoder] Layer norm 2: {}".format(layer_norm_2_output.shape))

#         linear_output = BATCH_FOWARD_DENSE(net_params[f"linear-{i}"], layer_norm_1_output)
#         print("[Decoder] Linear Output: {}".format(linear_output.shape))
#         add_3_output = np.add(layer_norm_2_output, linear_output)
#         print("[Decoder] Add layer 3: {}".format(add_3_output.shape))
#         layer_norm_3_output = jax.nn.normalize(add_3_output)
#         print("[Decoder] Add layer 2: {}".format(layer_norm_3_output.shape))
#         return layer_norm_3_output

def transformer_forward(model: dict,
                        config: argparse.Namespace,
                        input_data,
                        mask=None):
    """
    Transformer Forward Pass

    :param parameters: Transformer model (i.e., parameters)
    :param config: Config
    :param input_data: Input of dimensions [batch_size, seq_len, dims]
    :param mask: Input mask [batch_size, seq_len, 1] (default=None)

    :returns: output with dimensions [batch_size, seq_len, dims]
    """

    output = transformer_encoder_forward(
        model['encoder'], input_data, mask, config)

    # TODO: Impelement Transformer decoder forward function
    # output = transformer_decoder_forward(
    #     params["decoder"], config, output, input_data, mask)
    return output

# BATCH_TRANSFORMER_FORWARD = vmap(transformer_forward, in_axes=(None, None, 0, 0), out_axes=0)

# def create_positional_embeddings(tensor):
#     """Constructs positional embeddings given a tensor.

#     Keywords:
#         tensor (jax.numpy): Input tensor for a transformer
#     """

#     d_model = tensor.shape[-1]

#     dim_indices = tf.range(0, d_model, dtype=tf.float32)
#     sequence_indices = tf.range(0, tensor.shape[1], dtype=tf.float32)

#     even_indices = tf.where(sequence_indices % 2 == 0)
#     odd_indices = tf.where(sequence_indices % 2 != 0)

#     even_indices = tf.cast(even_indices, dtype=tf.float32)
#     odd_indices = tf.cast(odd_indices, dtype=tf.float32)

#     even_pos_embeddings = tf.math.sin(even_indices/tf.math.pow(10000.0, (2*dim_indices)/d_model))
#     odd_pos_embeddings = tf.math.cos(odd_indices/tf.math.pow(10000.0, (2*dim_indices)/d_model))

#     even_indices = tf.cast(even_indices, dtype=tf.int32)
#     odd_indices = tf.cast(odd_indices, dtype=tf.int32)

#     positional_embedding = tf.scatter_nd(
#         even_indices,
#         even_pos_embeddings,
#         shape=tf.constant([tensor.shape[1], tensor.shape[2]]))
#     positional_embedding = tf.tensor_scatter_nd_update(
#         positional_embedding,
#         odd_indices,
#         odd_pos_embeddings)
#     positional_embedding = tf.reshape(
#         positional_embedding,
#         [1, positional_embedding.shape[0], positional_embedding.shape[1]])
#     positional_embedding = tf.tile(
#         positional_embedding,
#         tf.constant([tensor.shape[0], 1, 1], dtype=tf.int32))
#     return positional_embedding
