import jax
import jax.numpy as jnp

from utils import dot_dict
'''
    This code is taken from JAX's training cookbook.
    For now I use Adam. In Memory Mosaics paper they used AdamW.
    If this does not work similar to their implementation I may need to 
    switch to AdamW. 
'''
def init_adam_state(param: jax.Array) -> dot_dict:
  adam_state = dot_dict(mu=jnp.zeros_like(param), nu=jnp.zeros_like(param), count=jnp.array(0))
  return adam_state 

def adam_update(config, param: jax.Ref, grad: jax.Array, adam_state: dot_dict):
  adam_state.mu[...] = (1 - config.beta_1) * adam_state.mu[...] + config.beta_1 * grad
  adam_state.nu[...] = (1 - config.beta_2) * adam_state.nu[...] + config.beta_2 * grad**2
  adam_state.count[...] += 1

  mu_hat = adam_state.mu[...] / (1 - config.beta_1 ** adam_state.count[...])
  nu_hat = adam_state.nu[...] / (1 - config.beta_2 ** adam_state.count[...])
  param[...] -= config.learning_rate * mu_hat / (jnp.sqrt(nu_hat + config.eps_root) + config.eps)

@jax.jit
def init_train_state(config) -> dot_dict:
  train_state = dot_dict()
  train_state.params = init_param_state(config)
  train_state.opt = jax.tree.map(init_adam_state, train_state.params)
  return train_state 

@jax.jit
def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
    def loss_fn(params):
        logits = model_apply(config, params, batch["observed_ids"])
        labels = jax.nn.one_hot(batch["target_ids"], config.vocab_size)
        return -(labels * jax.nn.log_softmax(logits)).mean()

    params = jax.tree.map(jax.ref.get, train_state.params)
    loss, grad = jax.value_and_grad(loss_fn)(params)
    jax.tree.map(ft.partial(adam_update, config), train_state.params, grad, train_state.opt)
    metrics = {"train_loss": loss}
    return metrics

class RecordWriter:
    prev_metrics = None
    
    def __call__(self, cur_metrics: dict):
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
        if log_metrics is None:
          return
        print(*it.starmap("{}: {}".format, log_metrics.items()), sep="\t")

def train_loop(config):
  record_writer = RecordWriter()
  train_state = init_train_state(config)
  train_state = jax.tree.map(jax.ref.new_ref, train_state)
  batch = iter(get_dataset_on_device(config))
  for step in range(config.num_train_steps):
    metrics = train_step(config, train_state, next(batch))
    record_writer({"step": step} | metrics)