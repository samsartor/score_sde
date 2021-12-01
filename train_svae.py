import os
import sys
import torch
from torch.optim import Adam
from utils import save_checkpoint, restore_checkpoint
import tensorflow as tf
import datasets
import logging
import numpy as np
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from models import svae


def step_fn(state, batch, train):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state["model"]
    if train:
        optimizer = state["optimizer"]
        optimizer.zero_grad()
        recons, input, mu, log_var = model.forward(batch)
        loss = model.loss_function(recons, input, mu, log_var)
        loss["loss"].backward()
        optimizer.step()
        state["step"] += 1
    else:
        with torch.no_grad():
            recons, input, mu, log_var = model.forward(batch)
            loss = model.loss_function(recons, input, mu, log_var)

    return loss


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    model = svae.ShallowVAE(config).to(config.device)
    print(model)
    optimizer = Adam(model.parameters())
    state = dict(optimizer=optimizer, model=model, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization
    )
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Building sampling functions
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = (
            torch.from_numpy(next(train_iter)["image"]._numpy())
            .to(config.device)
            .float()
        )
        batch = batch.permute(0, 3, 1, 2)
        batch = batch[:, :, :, :]
        batch = scaler(batch)

        # Execute one training step
        loss = step_fn(state, batch, True)
        if step % config.training.log_freq == 0:
            logging.info(f"step: {step}, training_loss: {loss['loss']:.5}")
            writer.add_scalar("training_loss", loss["loss"], step)
            writer.add_scalar("training_recon_loss", loss["recon"], step)
            writer.add_scalar("training_kld_loss", loss["kld"], step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = (
                torch.from_numpy(next(eval_iter)["image"]._numpy())
                .to(config.device)
                .float()
            )
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = step_fn(state, eval_batch, False)
            logging.info(f"step: {step}, eval_loss: {eval_loss['loss']:.5}")
            writer.add_scalar("eval_loss", eval_loss["loss"], step)
            writer.add_scalar("eval_recon_loss", eval_loss["recon"], step)
            writer.add_scalar("eval_kld_loss", eval_loss["kld"], step)

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state
            )

            # Generate and save samples
            if config.training.snapshot_sampling:
                with torch.no_grad():
                    sample = model.generate(batch)[:16, :, :, :]
                    sample = torch.cat(
                        (
                            sample,
                            model.sample(
                                16,
                                config.data.image_size,
                                config.data.image_size,
                                config.device,
                            ),
                        )
                    )
                    sample = inverse_scaler(sample)
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(
                    sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255
                ).astype(np.uint8)
                with tf.io.gfile.GFile(
                    os.path.join(this_sample_dir, "sample.np"), "wb"
                ) as fout:
                    np.save(fout, sample)

                with tf.io.gfile.GFile(
                    os.path.join(this_sample_dir, "sample.png"), "wb"
                ) as fout:
                    save_image(image_grid, fout)


if __name__ == "__main__":
    from configs.svae import church_v5 as configs

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    dir = "/tmp/church_svae_v5"
    tf.io.gfile.makedirs(dir)
    gfile_stream = open(os.path.join(dir, "stdout.txt"), "w")
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")
    train(configs.get_config(), dir)
