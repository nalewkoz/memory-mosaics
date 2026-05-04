import jax.numpy as jnp

def layer_norm(z, gamma, beta, eps=1e-6):
    mean = jnp.mean(z, axis=-1, keepdims=True)
    var  = jnp.mean( (z - mean)**2,  axis=-1, keepdims=True) # biased, but that should be fine
    z = (z - mean)/jnp.sqrt(var + eps)
    return gamma*z + beta
    
def gelu_new(z):
    #  "gelu new" (approximately similar to gelu)
    return z * (1 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (z + 0.044715 * z**3) ) )/2

def softmax(z, axis=-1):
    z_max = jnp.max(z, axis=axis, keepdims=True)
    expz = jnp.exp(z - z_max)
    return expz / jnp.sum(expz, axis=axis, keepdims=True)