import jax.numpy as jnp
import jax
import jax.random as random
from sampling import shared_corrector_update_fn, shared_predictor_update_fn
import functools

def get_pc_manifold_sampler(sde, loss, sched, model, shape, predictor,
    corrector, inverse_scaler, snr,
    n_steps=1, probability_flow=False, continuous=False,
    denoise=True, eps=1e-3):
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_manifold_sampler(rng, state, loss_params):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples, number of function evaluations
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
    g_loss = jax.value_and_grad(loss)
    init_loss = loss(x, loss_params)
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t

      rng, step_rng = random.split(rng)
      x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)

      goal_loss = sched(t / sde.T) * init_loss
      loss, slope = g_loss(x, loss_params)
      step = slope / jnp.sum(jnp.square(slope)) * (loss - goal_loss)
      x -= step

      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)

      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    # Denoising is equivalent to running one predictor step without adding noise.
    return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return jax.pmap(pc_manifold_sampler, axis_name='batch')
