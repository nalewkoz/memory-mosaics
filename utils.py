import jax

@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

  def tree_flatten_with_keys(self):
    keys = tuple(sorted(self))
    return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

  @classmethod
  def tree_unflatten(cls, keys, values):
    return cls(zip(keys, values))