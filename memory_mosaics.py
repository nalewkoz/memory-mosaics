import functools as ft
import itertools as it
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from utils import dot_dict
from mm_config import Config
from masks import causal_mask
from nn import layer_norm, gelu_new, softmax

'''
Memory mosaics 
'''

def c_associative_memory(K, V, beta):
    # K, V: "bskh"
    h = K.shape[-1]
    s = K.shape[1]
    KK = jnp.einsum("bskh,btkh->bkst", K, K) 
    M = causal_mask(s, k=-1) # shifted by one to account for acausal v's
    # make sure that M is broadcasted properly when added below (this should not be needed, check)
    M = M[None, None, :, :] 
    weights = softmax( beta*KK + M ) 
    return jnp.einsum("bkst,btkh->bskh", weights, V)

def p_associative_memory(Q, K, V, beta): 
    '''
    Since v is fixed as a parameter, this is effectively a simple feedforward 
    neural net. This implementation is likely unnecessary compute-heavy. 
    TODO: Test with a simple ff network like in the standard transformer block.
    '''
    # Q: "bskh" 
    # K, V: "bnkh" (n fixed and smaller than seq_len)
    h = K.shape[-1]
    QK = jnp.einsum("bskh,bnkh->bksn", Q, K) 
    weights = softmax( beta*QK ) # no causal mask needed!
    return jnp.einsum("bksn,bnkh->bskh", weights, V)
    
def feature_extractor(x, W, lamb, shift=0, normalize=True):
    '''
    Leaky feature extractor. 
    In larger networks/longer contexts this may be somewhat slow (check!),
    so I would try short convolutions instead, as they may be enough.
    At first it wasn't really clear to me if the leak coefficient is supposed to be trained, 
    but they do state that "we use a single scalar leaky average coefficient per head", 
    so it lookes like it is!
    '''
    # x: 'bsd'
    # W: 'dkh' (each head gets its own weight matrix)
    # lamb: 'k' (each head gets its own scalar decay parameter)
    # out: 'bskh'
    def step(carry, z):
        y = jnp.einsum("k,bkh->bkh", lamb, carry) + z
        return y, y
        
    z = jnp.einsum("bsd,dkh->sbkh", x, W) # notice swapped axes. We want to scan along the leading axis!
    _, out = jax.lax.scan(step, jnp.zeros_like(z[0,...]), z)
    out = jnp.swapaxes(out, 0, 1) 
    # Normalize (within each head, for each time step)
    if normalize:
        out *= jax.lax.rsqrt(jnp.sum(out**2, axis=-1, keepdims=True) + 1e-6)
    
    return jnp.roll(out, shift, axis=1) # roll needed for v
    
def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
    '''
    Simple memory mosaics implementation. 
    No Dropout. Only leaky feature extractor implemented at the moment.
    '''
    x = params.embeddings.at[tokens].get()
    del tokens
        
    for layer_ind in range(config.num_layers):
        block = params.layers[layer_ind]
        # pre-norm
        z = layer_norm(x, block.cmems.gamma, block.cmems.beta) 
        ## contextual memory units
        # slightly suboptimal to keep wk and wv separate, rather than in a single tensor. ok for simplicity.
        k = feature_extractor(z, block.cmems.wk, block.cmems.lambda_coef) 
        v = feature_extractor(z, block.cmems.wv, block.cmems.lambda_coef, shift=-1)
        z = c_associative_memory(k, v, block.cmems.beta)
        z = jnp.einsum("bskh,khd->bsd", z, block.cmems.w_out)
        x += z
        z = layer_norm(x, block.mlp.gamma, block.mlp.beta)
        ## persistent memory units
        # key, value pairs are predefined at training.
        # The current obs. is compared to stored keys. 
        # What they call key is actually more like a query then. 
        q = feature_extractor(z, block.pmems.wk, block.pmems.lambda_coef)
        z = associative_memory(q, block.pmems.k, block.pmems.v, block.pmems.beta)
        z = jnp.einsum("bskh,khd->bsd", z, block.pmems.w_out)
        x += z

    logits = jnp.einsum("bsd,dl->bsl", x, params.linear_out.w)
    return logits
