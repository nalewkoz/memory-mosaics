import jax.numpy as jnp

_MASK_CACHE = {}

def causal_mask(size, k=0):
    '''
    Use k=0 for standard causal mask, and k=-1 for retarded causal mask used in vanilla memory mosaics. 
    '''
    key = (size, k)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = jnp.where(
            jnp.tril(jnp.ones((size, size)), k=k) == 1,
            0.0,
            -1e10
        )
    return _MASK_CACHE[key]