#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torch.optim.lr_scheduler as lr_scheduler

import tqdm

from dataHandler.dataset import MHDProjDataset, LogNormalTransform
from ddpm.model import UNet, ResUNet
from ddpm.diffusion import DDPM, NCSN, SigmaDDPM, generate_image
from utils.scheduler import WarmUp

from absl import app, flags

## Constants and folders through flag parsing
####################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id",
    "MHD_DDPM_forget",
    help="ID of the model either trained, finetuned, evaluated....",
)

## Data & transforms
flags.DEFINE_string(
    "source_dir",
    "/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj",
    help="Source dir containing a list of npy files",
)
flags.DEFINE_bool("random_rotate", True, help="")
flags.DEFINE_bool(
    "no_lognorm", False, help="apply a lognormal transformation to the dataset"
)

## Network & diffusion parameters

flags.DEFINE_enum(
    "diffusion_mode",
    "ddpm",
    ["ddpm", "smld", "VE", "VD", "sub_VP", "SigmaDDPM"],
    help="Type of diffusion SDE used during training and inference",
)
## TODO if we want to fully use the SDE/ score based framework (and have custom/off the shelf SDE solvers, we will nedd to use runners/differentiate the training loop)
# For DDPM
flags.DEFINE_integer(
    "n_steps",
    1000,
    help="Diffusion total time see Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models",
)
flags.DEFINE_float(
    "beta_start",
    1e-4,
    help="Beta at time 0 see Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models",
)
flags.DEFINE_float(
    "beta_T",
    0.02,
    help="Beta at time T=n_steps Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models",
)

# Unet
flags.DEFINE_enum(
    "network", "unet", ["unet", "ResUNet"], help="DUNet not yet implemented"
)
flags.DEFINE_integer("size", 256, help="height and width of the images")
flags.DEFINE_integer(
    "in_channel", 1, help="number of channel on input and output images"
)
flags.DEFINE_enum(
    "normalization",
    "LN",
    [
        "LN",
        "default",
        "LN-D",
        "LN-F",
        "LN-F+",
        "LN-DF",
        "BN",
        "BN/LN",
        "BN/FLN",
        "BN/F+LN",
        "DBN/LN",
        "GN",
        "DN",
        "None",
    ],
    help="type of normalization applied",
)  ## TODO upcoming cleaning of these options based on elimination and perceived redundancies
flags.DEFINE_float(
    "eps_norm",
    1e-5,
    help="epsilon value added to the variance in the normalization layer to ensure numerical stability",
)
flags.DEFINE_integer("size_min", 32, help="size at the bottleneck")
flags.DEFINE_integer("num_blocks", 1, help="num of blocks per size on descent")
flags.DEFINE_enum(
    "padding_mode",
    "circular",
    ["zeros", "reflect", "replicate", "circular"],
    help="Conv2d padding mode",
)
flags.DEFINE_bool(
    "muP", False, help="Use mu Parametrisation for initialisation and training"
)  ## TODO
flags.DEFINE_float(
    "dropout",
    0,
    help="Probability for dropout, we did not find any impact because our models tend not to overfit",
)
flags.DEFINE_integer(
    "first_c_mult", 10, help="Multiplier between in_c and out_c for the first block"
)
flags.DEFINE_bool(
    "skip_rescale",
    False,
    help="Rescale skip connections (see Score Based Generative Modelling paper)",
)
flags.DEFINE_string("power_spectrum", '/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/power2.npy', help = "Power Spectrum file")

## Training parameters
flags.DEFINE_integer("batch_size", 64, help="Dataloader batch size")
flags.DEFINE_integer(
    "num_sample", 8, help="Number of sample for an epoch in the middle"
)
flags.DEFINE_integer(
    "num_result_sample", 256, help="Number of sample for an epoch in the middle"
)
flags.DEFINE_float("lr", 1e-3, help="Learning rate")
flags.DEFINE_enum(
    "lr_scheduler",
    "None",
    ["None", "stepLR"],
    help="scheduler, if any used in training",
)
flags.DEFINE_integer("warmup", 100, help="Length of warmup, if 0 then no warmup")
flags.DEFINE_integer("test_set_len", 95, help="")
flags.DEFINE_integer("num_epochs", 500, help="Number of epochs")
flags.DEFINE_enum(
    "optimizer",
    "Adam",
    ["AdamW", "Adam", "MoMo"],
    help="MoMo not implemented in particular for now",
)  ## TODO MoMo
flags.DEFINE_float("weight_decay", 0.0, help="Weight decay hyper parameter")
flags.DEFINE_float(
    "ema",
    0.0,
    help="Exponentially moving average momentum, if 0 then no ema applied NOT IMPLEMENTED yet ",
)  ## TODO

## Sampling and checkpointing
flags.DEFINE_integer(
    "save_step_epoch", 100, help="Period in nb of epochs for saving ckpt & losses"
)
flags.DEFINE_integer(
    "sample_step_epoch",
    100,
    help="Period in nb of epoch for generating a few npy samples",
)
flags.DEFINE_string(
    "sample_folder",
    "/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps",
    help="directory where generated samples (in the middle of training) or results are stored",
)
flags.DEFINE_string(
    "ckpt_folder",
    "/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps",
    help="Directory for ckpt & loss storage (as well as some training specs)",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Training
####################################################################################


def training_loop(
    model,
    dataloader,
    optimizer,
    num_epochs,
    num_timesteps,
    device=device,
    test_loss_batch=None,
    scheduler=None,
):
    """Training loop for DDPM"""
    try:
        os.mkdir(os.path.join(FLAGS.ckpt_folder, FLAGS.model_id))
    except:
        pass
    try:
        os.mkdir(os.path.join(FLAGS.sample_folder, FLAGS.model_id))
    except:
        pass

    global_step = 0
    losses = []
    test_losses_epoch = []
    steps_per_epoch = len(dataloader)

    for epoch in range(num_epochs):
        model.train()

        progress_bar = tqdm.tqdm(total=steps_per_epoch)
        progress_bar.set_description(f"Epoch {epoch}")

        for _, batch in enumerate(dataloader):
            if len(batch.shape)==3:
                batch = batch.to(device).unsqueeze(1)
            else:
                batch=batch.to(device)

            noise = torch.randn(batch.shape).to(device)
            timesteps = (
                torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)
            )
            loss = model.loss(timesteps, batch, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()

        if not (scheduler is None):
            scheduler.step()

        if (epoch % FLAGS.save_step_epoch == 0) or (epoch == num_epochs - 1):
            ckpt = {
                "ddpm_model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "test_set": dataloader.dataset.test_file_list,
            }
            torch.save(ckpt, os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "ckpt.pt"))
            np.save(
                os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "losses"),
                np.array(losses, dtype=float),
            )
            np.save(
                os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "test_losses"),
                np.array(test_losses_epoch, dtype=float),
            )
            np.save(
                os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "steps_per_epoch"),
                np.array(steps_per_epoch),
            )

        if (epoch % FLAGS.sample_step_epoch == 0) and (epoch != num_epochs - 1):
            generated, _ = generate_image(
                model, FLAGS.num_sample, FLAGS.in_channel, FLAGS.size
            )

            for i in range(FLAGS.num_sample):
                np.save(
                    os.path.join(
                        FLAGS.sample_folder,
                        FLAGS.model_id,
                        "sample_" + str(epoch).zfill(4) + "_" + str(i),
                    ),
                    generated[i].numpy(),
                )

        if epoch == num_epochs - 1:
            generated, _ = generate_image(
                model, FLAGS.num_result_sample, FLAGS.in_channel, FLAGS.size
            )

            for i in range(FLAGS.num_result_sample):
                np.save(
                    os.path.join(
                        FLAGS.sample_folder, FLAGS.model_id, "result_" + str(i).zfill(3)
                    ),
                    generated[i].numpy(),
                )

        if not (test_loss_batch is None):
            model.eval()
            with torch.no_grad():
                batch = test_loss_batch.to(device)
                noise = torch.randn(batch.shape).to(device)
                timesteps = (
                    torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)
                )
                loss = model.loss(timesteps, batch, noise).detach().cpu().item()

            test_losses_epoch.append(loss)
            model.train()
    return steps_per_epoch, losses, test_losses_epoch


def main(argv):
    try:
        os.mkdir(os.path.join(FLAGS.ckpt_folder, FLAGS.model_id))
        with open(
            os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "flagfile.txt"), "w"
        ) as f:
            f.write(FLAGS.flags_into_string())
    except:
        with open(
            os.path.join(FLAGS.ckpt_folder, FLAGS.model_id, "flagfile.txt"), "w"
        ) as f:
            f.write(FLAGS.flags_into_string())

    ## Dataloading & treatment
    ####################################################################################

    if FLAGS.no_lognorm:
        transforms = None
    else:
        transforms = LogNormalTransform()    

    dataset = MHDProjDataset(
        FLAGS.source_dir,
        test_batch_length=FLAGS.test_set_len,
        random_rotate=FLAGS.random_rotate,
        transforms=transforms,
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=8
    )

    ## Model
    ####################################################################################

    if FLAGS.network == "unet":
        log2sizes = list(
            range(int(np.log2(FLAGS.size_min)), int(np.log2(FLAGS.size)) + 1)
        )[::-1]
        sizes = [2**i for i in log2sizes]

        network = UNet(
            in_c=FLAGS.in_channel,
            out_c=FLAGS.in_channel,
            first_c=FLAGS.first_c_mult * FLAGS.in_channel,
            sizes=sizes,
            num_blocks=1,
            n_steps=FLAGS.n_steps,
            time_emb_dim=100,
            dropout=FLAGS.dropout,
            attention=[],
            normalisation=FLAGS.normalization,
            padding_mode=FLAGS.padding_mode,
            eps_norm=FLAGS.eps_norm,
        )
        network = network.to(device)
    if FLAGS.network == "ResUNet":
        log2sizes = list(
            range(int(np.log2(FLAGS.size_min)), int(np.log2(FLAGS.size)) + 1)
        )[::-1]
        sizes = [2**i for i in log2sizes]

        network = ResUNet(
            in_c=FLAGS.in_channel,
            out_c=FLAGS.in_channel,
            first_c=FLAGS.first_c_mult * FLAGS.in_channel,
            sizes=sizes,
            num_blocks=1,
            n_steps=FLAGS.n_steps,
            time_emb_dim=100,
            dropout=FLAGS.dropout,
            attention=[],
            normalisation=FLAGS.normalization,
            padding_mode=FLAGS.padding_mode,
            eps_norm=FLAGS.eps_norm,
            skiprescale=FLAGS.skip_rescale,
        )
        network = network.to(device)
    beta_T=FLAGS.beta_T*1000/FLAGS.n_steps
    if FLAGS.diffusion_mode == "ddpm":
        model = DDPM(
            network,
            FLAGS.n_steps,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    elif FLAGS.diffusion_mode == "smld":
        model = NCSN(
            network,
            FLAGS.n_steps,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    elif FLAGS.diffusion_mode == "SigmaDDPM":
        power_spectrum = torch.from_numpy(np.load(FLAGS.power_spectrum, allow_pickle=True).astype(np.float32))
        model  = SigmaDDPM(
            network,
            FLAGS.n_steps,
            power_spectrum=power_spectrum,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    if FLAGS.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            network.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
        )
    elif FLAGS.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
        )

    if FLAGS.lr_scheduler == "stepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=FLAGS.num_epochs // 5, gamma=0.5
        )
    else:
        scheduler = None
    if FLAGS.warmup > 0:
        scheduler = WarmUp(optimizer, scheduler, FLAGS.warmup, FLAGS.lr)

    losses = training_loop(
        model,
        dataloader,
        optimizer,
        FLAGS.num_epochs,
        FLAGS.n_steps,
        device=device,
        test_loss_batch=dataset.test_batch(),
        scheduler=scheduler,
    )


if __name__ == "__main__":
    app.run(main)
