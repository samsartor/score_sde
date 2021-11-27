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
  def scaled_loss(x, t, loss_params):
    return loss(inverse_scaler(x), loss_params)

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
    g_loss = jax.value_and_grad(scaled_loss)
    init_loss = scaled_loss(x, sde.T, loss_params)
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(val, i):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t

      rng, step_rng = random.split(rng)
      x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)

      goal_loss = sched(t / sde.T) * init_loss
      loss, slope = g_loss(x, t, loss_params)
      x -= slope / jnp.sum(jnp.square(slope)) * (loss - goal_loss)

      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)

      x_packed = jnp.int8((x - jnp.min(x)) / jnp.ptp(x) * 255 - 128)
      return (rng, x, x_mean), x_packed

    val, xs = jax.lax.scan(loop_body, (rng, x, x), jnp.arange(0, sde.N))
    _, x, x_mean = val
    # Denoising is equivalent to running one predictor step without adding noise.
    return inverse_scaler(x_mean if denoise else x), xs

  return jax.pmap(pc_manifold_sampler, axis_name='batch')
