# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

import ml_collections
from configs.default_lsun_configs import get_default_configs


def get_config():
    config = get_default_configs()

    training = config.training
    training.snapshot_sampling = True
    training.snapshot_freq = 5000

    # latent model
    config.latent = latent = ml_collections.ConfigDict()
    latent.dims = [3, 32, 128, 512]
    latent.strides = [2, 2, 2]
    latent.vq = True
    latent.num_embeddings = 512
    latent.embedding_dim = 64
    latent.dfc = False

    # data
    data = config.data
    data.category = "church_outdoor"
    data.center = True

    return config
