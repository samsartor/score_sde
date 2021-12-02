from configs.latent_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 128
  training.snapshot_freq = 20000

  # data
  data = config.data
  data.category = 'church_outdoor'

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # latent
  config.latent.checkpoint = 'v11/checkpoint_25.pth'

  # model
  model = config.model
  model.name = 'ddpm'
  model.num_scales = 1000
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (2, 2, 3, 4)
  model.num_res_blocks = 3
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
