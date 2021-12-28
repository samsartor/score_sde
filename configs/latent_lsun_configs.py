import ml_collections

import configs.default_lsun_configs

def get_default_configs():
  config = configs.default_lsun_configs.get_default_configs()

  config.latent = latent = ml_collections.ConfigDict()
  config.latent.checkpoint = 'v11/checkpoint_25.pth'
  latent.dims = [3, 64, 128, 256]
  latent.strides = [2, 2, 2]
  latent.dfc = False # set to true for training
  latent.vq = False
  latent.altpad = False
  latent.kld_weight = 0.0005

  return config
