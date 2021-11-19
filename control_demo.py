import jax
import jax.random as random
import flax

import models
from models import utils as mutils
import controllable_generation
import datasets

from sampling import *
from sde_lib import *

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from configs.ve import church_ncsnpp_continuous as configs
ckpt_filename = "work/ve_church_ncsnpp_continuous"
config = configs.get_config()
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

random_seed = 0
rng = jax.random.PRNGKey(random_seed)
rng, run_rng = jax.random.split(rng)
rng, model_rng = jax.random.split(rng)
score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)

train_ds, eval_ds, _ = datasets.get_dataset(config)
eval_iter = iter(eval_ds)
bpds = []

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

predictor = ReverseDiffusionPredictor
corrector = AnnealedLangevinDynamics
snr = 0.16
n_steps = 16
probability_flow = False

pc_inpainter = controllable_generation.get_pc_inpainter(
    sde, score_model,
    predictor, corrector,
    inverse_scaler,
    snr=snr,
    n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=config.training.continuous,
    denoise=True)
batch = next(eval_iter)
img = batch['image']._numpy()
show_samples(img)
rng, step_rng = jax.random.split(rng)
img = scaler(img)
mask = np.ones(img.shape)
mask[..., :, 16:, :] = 0.
mask = jnp.asarray(mask)
show_samples(inverse_scaler(img * mask))

rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
step_rng = jnp.asarray(step_rng)
pstate = flax.jax_utils.replicate(state)
x = pc_inpainter(step_rng, pstate, img, mask)
show_samples(x)
