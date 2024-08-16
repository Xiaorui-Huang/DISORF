# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
Method configurations
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.external_methods import get_external_methods
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

# from nerfstudio.models.nerfacto import NerfactoModelConfig
# from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.plugins.registry import discover_methods
from server_end.nsros.ros_datamanager import ROSDataManagerConfig
from server_end.nsros.ros_dataparser import ROSDataParserConfig
from server_end.nsros.ros_nerfacto import NerfactoModelConfig
from server_end.nsros.ros_pipeline import ROSDynamicBatchPipelineConfig, ROSPipelineConfig
from server_end.nsros.ros_splatfacto import SplatfactoModelConfig
from server_end.nsros.ros_trainer import ROSTrainerConfig

method_configs: Dict[str, ROSTrainerConfig] = {}
descriptions = {
    "ros_nerfacto": "Run the nerfstudio nerfacto method on data streamed from ROS.",
    "ros_instant_ngp": "Run the nerfstudio instant_ngp method on data streamed from ROS.",
    "ros_gaussian_splatting": "Run the nerfstudio gaussian_splatting method on data streamed from ROS.",
}

method_configs["ros_nerfacto"] = ROSTrainerConfig(
    method_name="ros_nerfacto",
    steps_per_eval_batch=500000,
    steps_per_eval_image=500000,
    steps_per_save=8000,
    max_num_iterations=8000,
    mixed_precision=True,
    pipeline=ROSPipelineConfig(
        datamanager=ROSDataManagerConfig(
            dataparser=ROSDataParserConfig(
                aabb_scale=0.8,  # pay attention [Ruofan]
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=20000),
    vis="viewer",
)

method_configs["ros_instant_ngp"] = ROSTrainerConfig(
    method_name="ros_instant_ngp",
    steps_per_eval_batch=10000,
    steps_per_eval_image=10000,
    steps_per_save=8000,
    max_num_iterations=8000,
    mixed_precision=True,
    pipeline=ROSDynamicBatchPipelineConfig(
        datamanager=ROSDataManagerConfig(
            dataparser=ROSDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            # disable_scene_contraction=True,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="viewer",
)

method_configs["ros_gaussian_splatting"] = ROSTrainerConfig(
    method_name="ros_gaussian_splatting",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=ROSPipelineConfig(
        datamanager=ROSDataManagerConfig(
            dataparser=ROSDataParserConfig(
                aabb_scale=0.8,  # pay attention [Ruofan]
            ),
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions, *discover_methods())
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
