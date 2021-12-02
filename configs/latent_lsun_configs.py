import ml_collections

from configs.default_lsun_configs import get_default_configs

def get_config():
  config = get_default_configs()

  config.latent = latent = ml_collections.ConfigDict()
  latent.checkpoint = 'v11/checkpoint_7.pth'
  latent.dims = [3, 32, 64, 128, 256, 256]
  latent.strides = [1, 2, 2, 2, 1]
  latent.dfc = False # set to true for training
  latent.vq = False
