import jax.numpy as jnp

from masks import causal_mask
from memory_mosaics import feature_extractor, c_associative_memory, p_associative_memory

print("===== Masks: =====")
print(causal_mask(3))
print(causal_mask(3,-1))
print(causal_mask(5))
print(causal_mask(5,-1))
print("memoization:")
a = causal_mask(3)
b = causal_mask(3)
c = causal_mask(3,-1)
d = causal_mask(4)
# if memoization works, a is the same as b but not c or d
assert a is b
assert not (a is c)
assert not (a is d)

print("===== Feature extractor: =====")
lambs = jnp.array( (0.0, 0.5, 1.0) )
b = 1
s = 7
d = 1
k = lambs.shape[0]
h = 1
x = jnp.ones( shape=(b,s,d) )
W = jnp.ones( shape=(d,k,h) )
print(x)
print(W)
keys = feature_extractor(x, W, lambs, normalize=False)
vals = feature_extractor(x, W, lambs, shift=-1, normalize=False)
# keys: because inputs are all 1's, we should get 1's, a sequence converging to 2, and a counter (simple sum).
# vals: same but shifted so that the same values appear one earlier.
print(keys)
print(vals)



