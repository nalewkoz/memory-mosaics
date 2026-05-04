import functools as ft
import itertools as it
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from utils import dot_dict
from gpt_config import Config
from masks import causal_mask
from nn import layer_norm, gelu_new, softmax

'''
Transformer (based on GPT2)
'''

def attention(Q, K, V):
    # Q, K, V: "bskh"
    h = K.shape[-1]
    s = K.shape[1]
    QK = jnp.einsum("bskh,btkh->bkst", Q, K) 
    M = causal_mask(s)
    # make sure that M is broadcasted properly when added below (this should not be needed, check)
    M = M[None, None, :, :] 
    weights = softmax( QK / jnp.sqrt(h) + M ) 
    return jnp.einsum("bkst,btkh->bskh", weights, V)
    
def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
    '''
    Following JAX's training cookbook, no dropout. 
    I switched to pre-norm with more standard LayerNorm.
    '''
    x = params.embeddings.at[tokens].get()
    x += params.pos_embed
    del tokens

    for layer_ind in range(config.num_layers):
        block = params.layers[layer_ind]
        z = layer_norm(x, block.attention.gamma, block.attention.beta) # pre-norm
        ## self-attention
        qkv = jnp.einsum("bsd,3dkh->bs3kh", z, block.attention.w_qkv)
        z = attention(qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :])
        z = jnp.einsum("bskh,khd->bsd", z, block.attention.w_out)
        x += z
        z = layer_norm(x, block.mlp.gamma, block.mlp.beta)
        ## ff net
        z = jnp.einsum("bsd,dh->bsh", z, block.mlp.w_in)
        z = gelu_new(z)
        z = jnp.einsum("bsh,hd->bsd", z, block.mlp.w_out)
        x += z

    logits = jnp.einsum("bsd,dl->bsl", x, params.linear_out.w)
    return logits

def init_param_state(config: Config) -> dot_dict:
    root_key = jax.random.key(config.param_seed)
    key = map(ft.partial(jax.random.fold_in, root_key), it.count()) # this is cool
    zero_init = jax.nn.initializers.constant(0.0)
    he_init = jax.nn.initializers.he_normal(1, 1)  # If I have time, try to initialize with my own initializer
    dtype = config.dtype
    
    params = dot_dict(
    pos_embed=zero_init(next(key), (config.seq_length, config.embed_dim), dtype),
    layers=dot_dict(),
    )
    params.embedding = he_init(next(key), (config.vocab_size, config.embed_dim), dtype)
    ## linear_in is not used in model_apply O_O
    #params.linear_in = dot_dict(
    #kernel=he_init(next(key), (1, config.embed_dim), dtype),
    #bias=zero_init(next(key), (config.embed_dim,), dtype),
    #)
    params.linear_out = dot_dict(
    w=he_init(next(key), (config.embed_dim, config.vocab_size), dtype),
    )
    for layer in range(config.num_layers):
        qkv_shape = (3, config.embed_dim, config.num_heads, config.head_dim)
        out_shape = (config.num_heads, config.head_dim, config.embed_dim)
        params.layers[layer] = dot_dict(
          attention=dot_dict(
            w_qkv=he_init(next(key), qkv_shape, dtype, config.att_qkv),
            w_out=he_init(next(key), out_shape, dtype, config.att_out),
          ),
          mlp=dot_dict(
            w_in=he_init(next(key), (config.embed_dim, config.mlp_dim), dtype),
            w_out=he_init(next(key), (config.mlp_dim, config.embed_dim), dtype),
          ),
        )
    return params 
    