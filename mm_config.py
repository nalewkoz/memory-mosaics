import jax
from dataclasses import dataclass

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    seq_length: int = 128
    
    num_train_steps: int = 10**6
    host_batch_size: int = 16
    learning_rate: float = 1e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    
    param_seed: int = 12738
    num_layers: int = 4
    embed_dim: int = 512
    #mlp_dim: int = 512 * 4
    vocab_size: int = 2**8  # uint8 ascii encoding
    num_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"