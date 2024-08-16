#!/usr/bin/env python
# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/scripts/train.py

"""
Train a radiance field with nerfstudio using data streamed from ROS!

This is a stripped back version of the nerfstudio train.py script without
all of the distributed training code that is not support by the NSROS Bridge.

All of the tyro help functionality should still work, but instead of a CLI
just call this script directly:
    python ros_train.py ros_nerfacto --data /path/to/config.json [OPTIONS]
"""

import random
import signal
import sys
import traceback

import numpy as np
import torch
import tyro
from server_end.nsros.method_configs import AnnotatedBaseConfigUnion
from server_end.nsros.ros_datamanager import ROSDataManager
from server_end.nsros.ros_trainer import ROSTrainerConfig
from rich.console import Console

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils import profiler
import warnings

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore

warnings.simplefilter('ignore', UserWarning) # ignore annoying pytorch warnings

def sigint_handler(signal, frame):
    """Capture keyboard interrupts before they get caught by ROS."""
    CONSOLE.print(traceback.format_exc())
    sys.exit(0)


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: ROSTrainerConfig) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.dataparser.data")
        config.pipeline.datamanager.dataparser.data = config.data

    if config.max_num_iterations:
        CONSOLE.log("Using --max-num-iterations alias for --pipeline.datamanager.max_num_iterations")
        config.pipeline.datamanager.max_num_iterations = config.max_num_iterations

    # only non-splatfacto models need to set the camera_optimizer mode
    if config.method_name == "ros_gaussian_splatting":
        config.pipeline.datamanager.use_gaussian_splatting = True
    else:
        config.pipeline.datamanager.use_gaussian_splatting = False

    # TODO: remove fix online mode in ros_datamanager
    if config.pipeline.model.camera_optimizer.mode == "off":
        CONSOLE.log("Using Old DataLoader")
        config.pipeline.datamanager.use_camopt_dataloader = False
    elif config.pipeline.model.camera_optimizer.mode:
        CONSOLE.log("Using New DataLoader")
        config.pipeline.datamanager.use_camopt_dataloader = True


    # print and save config
    config.print_to_terminal()
    config.save_config()
    try:
        _set_random_seed(config.machine.seed)
        trainer = config.setup(local_rank=0, world_size=1)
        trainer.setup()
        trainer.train()

    except KeyboardInterrupt:
        # print the stack trace
        CONSOLE.print(traceback.format_exc())
    finally:
        profiler.flush_profiler(config.logging)
        if isinstance(trainer.pipeline.datamanager, ROSDataManager):
            datamanager = trainer.pipeline.datamanager
            if datamanager.config.track_avail_images_per_iter:
                avail_images_per_iter = datamanager.avail_images_per_iter
                avail_images_per_iter = np.array(avail_images_per_iter)
                np.savetxt(trainer.base_dir / "avail_frames_log.txt", avail_images_per_iter, fmt="%d")

        if trainer.config.dump_loss_track and trainer.pipeline.datamanager.config.training_sampler == "loss_sampling":
            pixel_loss = trainer.pipeline.datamanager.pixel_loss
            patch_loss = trainer.pipeline.datamanager.patch_loss
            frame_loss = trainer.pipeline.datamanager.frame_loss

            dict = {"pixel_loss": pixel_loss, "patch_loss": patch_loss, "frame_loss": frame_loss}
            torch.save(dict, trainer.base_dir / "loss_track.pth")

        dump_cnt_track = True
        if dump_cnt_track and hasattr(trainer.pipeline.datamanager, "frame_sample_cnt"):
            frame_sample_cnt = trainer.pipeline.datamanager.frame_sample_cnt
            torch.save(frame_sample_cnt, trainer.base_dir / "cnt_track.pth")

        if trainer.config.dump_received_images:
            trainer.pipeline.datamanager.train_image_dataloader.dump_data()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    entrypoint()
