import torch
from sampling import shared_corrector_update_fn, shared_predictor_update_fn
import functools

def get_pc_manifold_sampler(sde, loss, sched, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_manifold_sampler(model, loss_params, svae=None):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    # Initial sample
    with torch.no_grad():
      x = sde.prior_sampling(shape).to(device)
      if svae is not None:
        y = svae.decode(x)
      else:
        y = x
      init_loss = loss(y, loss_params)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    for i in range(sde.N):
      t = timesteps[i]
      vec_t = torch.ones(shape[0], device=t.device) * t
      with torch.no_grad():
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        pass

      x.requires_grad_(True)
      goal_loss = sched(t / sde.T) * init_loss
      if svae is not None:
        y = svae.decode(x)
      else:
        y = x
      this_loss = loss(y, loss_params)
      this_loss.backward()
      slope = x.grad
      x.requires_grad_(False)

      with torch.no_grad():
        mult = 1 / torch.sum(torch.square(slope)) * (this_loss - goal_loss)
        mult = max(min(mult, 0.01), 0)
        x -= slope * mult
        x, x_mean = predictor_update_fn(x, vec_t, model=model)

    x = x_mean if denoise else x
    if svae is not None:
      x = svae.decode(x)
    return inverse_scaler(x)

  return pc_manifold_sampler
