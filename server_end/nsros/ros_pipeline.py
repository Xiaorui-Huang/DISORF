# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Abstracts for the Pipeline class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional, Type

import numpy as np
import torch
from server_end.nsros.ros_datamanager import ROSDataManager
from PIL import Image
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)

from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.pipelines.dynamic_batch import (DynamicBatchPipeline,
                                                DynamicBatchPipelineConfig)
from nerfstudio.utils import profiler


@dataclass
class ROSPipelineConfig(VanillaPipelineConfig):
    """ROS Pipeline Config."""

    _target: Type = field(default_factory=lambda: ROSPipeline)
    save_eval_images: bool = True
    """Whether to save eval rendered images to disk."""
    output_dir: Optional[Path] = None
    """Path to save eval rendered images to."""
    get_std: bool = True
    """Whether to return std with the mean metric in eval output."""
    eval_idx_list: str = ""


class ROSPipeline(VanillaPipeline):
    """Pipeline class for ROS."""

    config: ROSPipelineConfig
    datamanager: ROSDataManager

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, ROSDataManager)

        # if self.datamanager.config.eval_with_training_set and \
        #   hasattr(self.model.field, "use_training_cameras"):
        #     self.model.field.use_training_cameras = True

        if self.datamanager.config.eval_with_training_set and hasattr(self.model, "use_training_cameras"):
            self.model.use_training_cameras = True

        num_images = len(self.datamanager.fixed_indices_eval_dataloader)

        eval_idx_list = None
        if self.config.eval_idx_list != "":
            eval_idx_list = np.loadtxt(self.config.eval_idx_list, dtype=int)
            num_images = len(eval_idx_list)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i, (camera, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                if eval_idx_list is not None and i not in eval_idx_list:
                    continue
                # time this the following line
                inner_start = time()
                height, width = camera.height, camera.width
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera(camera=camera)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if self.config.save_eval_images:
                    camera_indices = self.datamanager.fixed_indices_eval_dataloader.image_indices[
                        self.datamanager.fixed_indices_eval_dataloader.count - 1
                    ]
                    output_path = self.config.output_dir / f"{step:06d}" if step is not None else ""
                    os.makedirs(output_path, exist_ok=True)
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        if key == "img":
                            Image.fromarray((val * 255).byte().cpu().numpy()).save(
                                output_path / "{0:06d}-{1}.jpg".format(camera_indices, key)
                            )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if self.config.get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        # metrics_dict["All PSNR"] = [{"psnr": metrics_dict["psnr"]} for metrics_dict in metrics_dict_list]
        # print(metrics_dict)
        for key, val in metrics_dict.items():
            print(f"{key}: {val}")
        import pandas as pd
        # dump avg metrics_dict to csv, each key a column
        df = pd.DataFrame(metrics_dict, index=[0])
        csv_path = str(self.config.output_dir / f"metrics.csv")
        df.to_csv(csv_path)
        self.train()
        return metrics_dict
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        if self.datamanager.config.use_gaussian_splatting:
            self._model.add_frame_camera = self.datamanager.train_dataset.cameras[self.datamanager.current_avail_images - 1: self.datamanager.current_avail_images]
            if self.datamanager.add_frame and not self.datamanager.add_random_pcd:
                if self.datamanager.receive_pcd:
                    # add pcd guided gaussian (When new frame is added)
                    self._model.new_points = (self.datamanager.train_dataset.pcd[0][self.datamanager.train_dataset.f2p[self.datamanager.previous_avail_images - 1]: self.datamanager.train_dataset.f2p[self.datamanager.current_avail_images - 1]],
                                            self.datamanager.train_dataset.pcd[1][self.datamanager.train_dataset.f2p[self.datamanager.previous_avail_images - 1]: self.datamanager.train_dataset.f2p[self.datamanager.current_avail_images - 1]])
                    self._model.add_frame = self.datamanager.add_frame
                else:
                    # do not add any pts (When new frame is added)
                    self._model.new_points = None
                    self._model.add_frame = not self.datamanager.add_frame
                    
            elif self.datamanager.add_frame and self.datamanager.add_random_pcd:
                # add random gaussian (When new frame is added)
                self._model.new_points = None
                self._model.add_frame = self.datamanager.add_frame
            elif not self.datamanager.add_frame:
                # do not add any pts (When new frame is not added)
                self._model.add_frame = self.datamanager.add_frame
            
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict


@dataclass
class ROSDynamicBatchPipelineConfig(DynamicBatchPipelineConfig):
    """ROS Dynamic Batch Pipeline Config."""

    _target: Type = field(default_factory=lambda: ROSDynamicBatchPipeline)
    save_eval_images: bool = True
    """Whether to save eval rendered images to disk."""
    output_dir: Optional[Path] = None
    """Path to save eval rendered images to."""
    get_std: bool = True
    """Whether to return std with the mean metric in eval output."""
    eval_idx_list: str = ""


class ROSDynamicBatchPipeline(DynamicBatchPipeline):
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, ROSDataManager)

        if self.datamanager.config.eval_with_training_set and hasattr(self.model.field, "use_training_cameras"):
            self.model.field.use_training_cameras = True
        
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)

        eval_idx_list = None
        if self.config.eval_idx_list != "":
            eval_idx_list = np.loadtxt(self.config.eval_idx_list, dtype=int)
            num_images = len(eval_idx_list)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i, (camera_ray_bundle, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                if eval_idx_list is not None and i not in eval_idx_list:
                    continue
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if self.config.save_eval_images:
                    camera_indices = camera_ray_bundle.camera_indices
                    output_path = self.config.output_dir / f"{step:06d}" if step is not None else ""
                    os.makedirs(output_path, exist_ok=True)
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        if key == "img":
                            Image.fromarray((val * 255).byte().cpu().numpy()).save(
                                output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                            )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if self.config.get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        # metrics_dict["All PSNR"] = [{"psnr": metrics_dict["psnr"]} for metrics_dict in metrics_dict_list]
        self.train()
        return metrics_dict
