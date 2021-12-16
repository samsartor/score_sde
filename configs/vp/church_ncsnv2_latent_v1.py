from configs.latent_lsun_configs import get_default_configs

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 128
  training.snapshot_freq = 5000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'none'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.centered = True
  data.category = 'church_outdoor'

  # latent
  config.latent.checkpoint = 'v12/checkpoint_10.pth'

  # model
  model = config.model
  model.name = 'ncsnv2_64'
  model.scale_by_sigma = False
  model.num_scales = 232
  model.ema_rate = 0.999
  model.normalization = 'InstanceNorm++'
  model.nonlinearity = 'elu'
  model.nf = 512
  model.interpolation = 'bilinear'

  return config
