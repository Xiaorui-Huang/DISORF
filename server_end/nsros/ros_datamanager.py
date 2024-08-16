"""
A datamanager for the NSROS Bridge.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Union

import random
import numpy as np
import torch
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.ray_generators import RayGenerator
from server_end.nsros.ros_dataloader import ROSDataloader, ROSFixedIndicesEvalDataloader, ROSRandIndicesEvalDataloader
from server_end.nsros.ros_dataparser import ROSDataParserConfig
from server_end.nsros.ros_dataset import ROSDataset

CONSOLE = Console(width=120)


@dataclass
class ROSDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A ROS datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ROSDataManager)
    dataparser: ROSDataParserConfig = ROSDataParserConfig()
    """ Must use only the ROSDataParser here """
    publish_training_posearray: bool = True
    """ Whether the dataloader should publish an pose array of the training image poses. """
    data_update_freq: float = 300.0
    """ Frequency, in Hz, that images are added to the training dataset tensor. """
    num_training_images: int = 300
    """ Number of images to train on (for dataset tensor pre-allocation). """
    num_msgs_to_start: int = 3
    """ Number of messages to wait for before starting training. """
    training_sampler: Literal[
        "default",
        "hybrid",
        "alternating",
        "inverse_delta",
        "loss_weighted_frame",
        "loss_weighted_patch", # NeRF only sampling method
        "imap",
        "hierarchical_loss",
    ] = "default"
    """ Sampler to use for training. """
    delta_sample_scale: float = 2.0
    delta_sample_ratio: float = 4.0
    multi_sampler_list: List[str] = field(default_factory=lambda: ['default', 'inverse_delta'])
    """ List of samplers to use for training, when hybrid or alternating samplers are used. """
    multi_sampler_ratio: List[float] = field(default_factory=lambda: [])
    """ hybrid ratio for the hybrid sampler. """
    max_num_iterations: int = 8000
    """ Maximum number of iterations to train for. """
    use_camopt_dataloader: bool = True
    """ Whether to use the new dataloader. """
    track_perpixel_loss: bool = False
    """ Whether to track the per-pixel loss. """
    track_frame_loss: bool = False
    """ Whether to track the frame loss. """
    eval_with_training_set: bool = True
    """ Whether to evaluate with the training set. """
    track_sample_cnt: bool = True
    """ Whether to track the number of samples per image. """
    track_avail_images_per_iter: bool = True
    """ Whether to track the number of available images per iteration. """
    replay_transmission_log: str = ""
    """ A txt file containing the transmission log. """
    track_frame_timestamp: bool = False
    """ Whether to track the frame timestamp, in the form of iteration id"""
    output_dir: str = ""
    """ Output directory for the eval images. """
    use_gaussian_splatting: bool = False
    """ Whether to use gaussian splatting """
    receive_pcd: bool = False
    """ Whether to receive pcd data """
    add_random_pcd: bool = False
    """ Whether to add random pcd data """


class ROSDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from ROS.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ROSDataManagerConfig
    train_dataset: ROSDataset
    eval_dataset: ROSDataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_avail_images = 0
        self.current_batch = None
        self.current_batch_metric = None
        self.dummy_batch = False

    def create_train_dataset(self) -> ROSDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return ROSDataset(dataparser_outputs=self.train_dataparser_outputs, device=self.device)

    def setup_train(self):
        CONSOLE.print("Setting up training dataset...")
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras)
        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            self.config.publish_training_posearray,
            self.config.receive_pcd,
            self.config.data_update_freq,
            self.config.use_camopt_dataloader,
            device=self.device,
            num_workers=0,  # maybe ...
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

        n, h, w = self.train_image_dataloader.dataset.image_tensor.shape[:3]
        self.height, self.width = h, w

        training_samplers = [self.config.training_sampler]
        if self.config.training_sampler in ["hybrid", "alternating", "imap", "hierarchical_loss"]:
            training_samplers += self.config.multi_sampler_list
        for sampler in training_samplers:
            if "loss" in sampler:
                if not self.config.use_gaussian_splatting:
                    self.config.track_perpixel_loss = True
                else:
                    self.config.track_frame_loss = True
            if sampler in ["imap", "inverse_delta"]:
                self.config.track_frame_timestamp = True
            if sampler == "hybrid":
                number_samplers = len(self.config.multi_sampler_list)
                if len(self.config.multi_sampler_ratio) == 0:
                    self.config.multi_sampler_ratio = [1.0] * number_samplers
                self.multi_sampler_ratio = np.array(self.config.multi_sampler_ratio)
                self.multi_sampler_ratio /= np.sum(self.multi_sampler_ratio)
                if not self.config.use_gaussian_splatting:
                    total_rays_per_batch = self.config.train_num_rays_per_batch
                    self.rays_per_sampler = np.floor(total_rays_per_batch * self.multi_sampler_ratio).astype(np.int32)
                    self.rays_per_sampler[-1] = total_rays_per_batch - np.sum(self.rays_per_sampler[:-1])

        if self.config.use_gaussian_splatting:
            assert self.config.training_sampler in ["default", "imap", "inverse_delta", "hybrid", "hierarchical_loss"]
            self.config.track_frame_timestamp = True
            self.add_frame = False
            self.viewpoint_stack = None


        if self.config.track_avail_images_per_iter:
            self.avail_images_per_iter = []

        self.replay_trans_log = False
        if self.config.replay_transmission_log != "":
            # a txt file containing the transmission log
            self.avail_images_per_iter = np.loadtxt(self.config.replay_transmission_log, dtype=np.int32)
            self.config.track_avail_images_per_iter = False
            self.replay_trans_log = True

        if self.config.track_sample_cnt:
            # self.frame_sample_cnt = np.array([0] * self.config.num_training_images)
            self.frame_sample_cnt = torch.zeros(self.config.num_training_images, device=self.device)
        
        if self.config.track_frame_loss:
            self.frame_loss = torch.zeros(self.config.num_training_images, device=self.device)


        if self.config.track_perpixel_loss:
            self.pixel_loss = torch.zeros(
                self.train_image_dataloader.dataset.image_tensor.shape[:3], device=self.device
            )
            self.pixel_sample_cnt = torch.zeros(
                self.train_image_dataloader.dataset.image_tensor.shape[:3], dtype=torch.int32, device=self.device
            )
            # patch level
            n, h, w = self.pixel_loss.shape[0:3]
            self.image_size = (h, w)
            self.patch_size = 16  # 16x16
            self.num_patches = self.patch_size * self.patch_size

            patch_h, patch_w = h // self.patch_size, w // self.patch_size
            self.patch = (patch_h, patch_w)
            pad_h, pad_w = h - patch_h * self.patch_size, w - patch_w * self.patch_size
            self.pad_patch = (pad_h + patch_h, pad_w + patch_w)  # some overlapping areas
            patch_loss = torch.nn.functional.avg_pool2d(
                self.pixel_loss, self.patch, ceil_mode=False, divisor_override=1
            )  # [Bs, Ps, Ps]
            patch_valid = torch.nn.functional.avg_pool2d(
                (self.pixel_sample_cnt > 0).float(), self.patch, ceil_mode=False, divisor_override=1
            )  # [Bs, Ps, Ps]
            self.patch_loss = patch_loss / (patch_valid + 1e-3)  # [Bs, Ps, Ps]
            self.frame_loss = self.patch_loss.view(self.patch_loss.shape[0], -1).mean(dim=1)  # [Bs]

        if self.config.track_frame_timestamp:
            self.frame_timestamp = torch.zeros(self.config.num_training_images, device=self.device)
            self.frame_timestamp[:] = 0
            self.previous_avail_images = 0
            self.last_timestamp = 0
            self.avg_frame_speed = 1.0
        
        # for gaussian splatting pcd
        self.receive_pcd = self.config.receive_pcd
        self.add_random_pcd = self.config.add_random_pcd

    #########################################
    ### modular sample functions

    def _default_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        if img_indices_only:
            image_indices = (
                torch.floor(
                    torch.rand((num_samples, 1), device=self.device) * (available_frames - eariliest_frame)
                ).long()
                + eariliest_frame
            )
            return image_indices

        indices = torch.floor(
            torch.rand((num_samples, 3), device=self.device)
            * torch.tensor([available_frames - eariliest_frame, self.height, self.width], device=self.device)
        ).long()
        if eariliest_frame > 0:
            indices[:, 0] += eariliest_frame
        return indices



    def _loss_weighted_patch_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        base_loss = 0.001
        patch_loss = self.patch_loss[eariliest_frame:available_frames]  # [Bs, Ps, Ps]
        mean_loss = torch.mean(patch_loss)
        patch_weights = patch_loss.clamp_max(5 * mean_loss) + mean_loss * 0.5 + base_loss

        patch_indices, pix_indices = multinomial_ray_sample(
            patch_weights.flatten(), num_samples, *self.pad_patch, self.device
        )
        image_indices = patch_indices // self.num_patches
        patch_loc_idx = patch_indices - image_indices * self.num_patches
        patch_h_offset = (patch_loc_idx // self.patch_size) * self.patch[0]
        patch_w_offset = (patch_loc_idx % self.patch_size) * self.patch[1]
        pix_indices[:, 0] += patch_h_offset
        pix_indices[:, 1] += patch_w_offset

        indices = torch.cat((image_indices.unsqueeze(-1) + eariliest_frame, pix_indices), dim=1)
        return indices

    def _loss_weighted_frame_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        base_loss = 0.001
        frame_loss = self.frame_loss[eariliest_frame:available_frames]
        mean_loss = torch.mean(frame_loss)  
        frame_loss = frame_loss.clamp_max(5 * mean_loss) + mean_loss * 0.5 + base_loss
        image_indices, pix_indices = multinomial_ray_sample(
            frame_loss, num_samples, self.height, self.width, self.device, img_indices_only
        )
        if img_indices_only:
            return image_indices
        indices = torch.cat((image_indices.unsqueeze(-1) + eariliest_frame, pix_indices), dim=1)
        return indices

    def _hybrid_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        multi_indices = []
        for i, sampler in enumerate(self.config.multi_sampler_list):
            indices_i = getattr(self, f"_{sampler}_sample")(available_frames, self.rays_per_sampler[i])
            multi_indices.append(indices_i)
        indices = torch.cat(multi_indices, dim=0)
        return indices

    def _alternating_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        current_sampler_idx = (self.train_count - 1) % len(self.config.multi_sampler_list)
        current_sampler = self.config.multi_sampler_list[current_sampler_idx]
        return getattr(self, f"_{current_sampler}_sample")(available_frames, num_samples)

    def _imap_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        # fixed 20 % rays from the latest frame
        # 80 % rays from the previous frames
        inner_sampler = self.config.multi_sampler_list[0]
        if (self.train_count - self.last_timestamp) <= 0.5 * 1 / self.avg_frame_speed:
            latest_ray_num = int(num_samples * 0.2)
            rest_ray_num = num_samples - latest_ray_num
            latest_indices = getattr(self, f"_{inner_sampler}_sample")(
                available_frames, latest_ray_num, available_frames - 1
            )
            rest_indices = getattr(self, f"_{inner_sampler}_sample")(available_frames - 1, rest_ray_num)
            indices = torch.cat((latest_indices, rest_indices), dim=0)
        else:
            indices = getattr(self, f"_{inner_sampler}_sample")(available_frames, num_samples)

        return indices

    def _inverse_delta_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        time_delta = self.train_count - 1 - self.frame_timestamp[eariliest_frame:available_frames] + 1e-3
        # sample_weight = 1 / time_delta
        scale, ratio = self.config.delta_sample_scale, self.config.delta_sample_ratio
        # 1/(ratio+1) from the latest frame; if pose refinement is not used, larger ratio=10 is better
        # scale, ratio = 1, 9
        sample_weight = (
            torch.exp(-(time_delta - 1) * max(self.avg_frame_speed * scale, 0.01)) + ratio / available_frames
        )
        image_indices, pix_indices = multinomial_ray_sample(
            sample_weight, num_samples, self.height, self.width, self.device, img_indices_only
        )
        if img_indices_only:
            return image_indices

        indices = torch.cat((image_indices.unsqueeze(-1) + eariliest_frame, pix_indices), dim=1)

        debug = False
        if debug and (self.train_count - 1) % 5 == 0:
            CONSOLE.print(f"[red]Avg frame speed: {self.avg_frame_speed}")
            plot_samples_debug(image_indices, self.train_count, self.config.output_dir)
        return indices

    def _hierarchical_loss_sample(
        self, available_frames: int, num_samples: int, eariliest_frame: int = 0, img_indices_only: bool = False
    ):
        frame_sampler = self.config.multi_sampler_list[0]
        # recent frames
        recent_iters = 10
        recent_frames = (
            ((self.frame_timestamp[:available_frames] + recent_iters - self.train_count + 1) >= 0).sum().item()
        )
        time_delta = self.train_count - 1 - self.frame_timestamp[eariliest_frame:available_frames] + 1e-3
        # sample_weight = torch.exp(-(time_delta - 1) * min(self.avg_frame_speed * 4, 1.2)) + 0.02
        scale, ratio = 2, 4
        sample_weight = (
            torch.exp(-(time_delta - 1) * max(self.avg_frame_speed * scale, 0.01)) + ratio / available_frames
        )
        # image_indices, pix_indices = multinomial_ray_sample(
        #     sample_weight, num_samples, self.height, self.width, self.device, img_indices_only
        # )
        if recent_frames == available_frames:
            recent_num_samples = num_samples
        elif recent_frames == 0:
            recent_num_samples = 0
        else:
            ratio = (sample_weight[-recent_frames:].sum() / sample_weight.sum()).item()
            recent_num_samples = int(num_samples * ratio)
        rest_num_samples = num_samples - recent_num_samples

        if recent_num_samples > 0:
            # image_indices = getattr(self, f"_{frame_sampler}_sample")(
            #         available_frames, recent_num_samples, available_frames - recent_frames, img_indices_only=True
            #     )
            # # loss patch sampling inside
            # patch_loss = self.patch_loss[image_indices].reshape(recent_num_samples, -1) # [Bs, Ps * Ps]
            # patch_weights = patch_loss + torch.mean(patch_loss, dim=1, keepdim=True) * 0.6 + 0.001 # [Bs, Ps * Ps]
            # patch_loc_idx = torch.multinomial(patch_weights, 1, replacement=True).flatten() # [Bs]
            # # import IPython; IPython.embed();
            # pix_indices = torch.floor(
            #     torch.rand((recent_num_samples, 2), device=self.device)
            #     * torch.tensor(self.pad_patch, device=self.device)
            # ).long()
            # patch_h_offset = (patch_loc_idx // self.patch_size) * self.patch[0]
            # patch_w_offset = (patch_loc_idx % self.patch_size) * self.patch[1]
            # pix_indices[:, 0] += patch_h_offset
            # pix_indices[:, 1] += patch_w_offset

            # indices = torch.cat((image_indices.unsqueeze(-1), pix_indices), dim=1)

            # indices = getattr(self, f"_{frame_sampler}_sample")(
            #     available_frames, recent_num_samples, available_frames - recent_frames, img_indices_only=False
            # )

            indices = self._loss_weighted_patch_sample(
                available_frames, recent_num_samples, available_frames - recent_frames
            )

            # image_indices, pix_indices = multinomial_ray_sample(
            #     sample_weight[-recent_frames:], recent_num_samples,
            #     self.height, self.width, self.device
            # )
            # indices = torch.cat((image_indices.unsqueeze(-1) + available_frames - recent_frames, pix_indices), dim=1)

        # rest frames
        if rest_num_samples > 0:
            rest_indices = self._loss_weighted_patch_sample(available_frames - recent_frames, rest_num_samples)
            if recent_num_samples > 0:
                indices = torch.cat((indices, rest_indices), dim=0)
            else:
                indices = rest_indices

        return indices

    def _imap_gs_sample(self, available_frames: int, eariliest_frame: int = 0):

        if (self.train_count - self.last_timestamp) <= 0.5 * 1 / self.avg_frame_speed:
            if random.random() < 0.1:
                indices = torch.tensor([available_frames - 1], device=self.device).unsqueeze(-1)
            else:
                indices = torch.tensor([random.randint(0, available_frames - 2)], device=self.device).unsqueeze(-1)
        else:
            indices = torch.tensor([random.randint(0, available_frames - 1)], device=self.device).unsqueeze(-1)

        return indices

    def _inverse_delta_gs_sample(self, available_frames: int, eariliest_frame: int = 0):
        if self.add_frame:
            indices = torch.tensor([available_frames - 1], device=self.device).unsqueeze(-1)
            return indices
        
        time_delta = self.train_count - 1 - self.frame_timestamp[eariliest_frame:available_frames] + 1e-3
        # sample_weight = 1 / time_delta
        scale, ratio = self.config.delta_sample_scale, self.config.delta_sample_ratio
        # 1/(ratio+1) from the latest frame; if pose refinement is not used, larger ratio=10 is better
        # scale, ratio = 1, 9
        sample_weight = (
            torch.exp(-(time_delta - 1) * max(self.avg_frame_speed * scale, 0.01)) + ratio / available_frames
        )
        indices, _ = multinomial_ray_sample(sample_weight, 1, self.height, self.width, self.device, True)

        return indices

    def _default_gs_sample(self, available_frames: int, eariliest_frame: int = 0):
        if self.add_frame:
            indices = torch.tensor([available_frames - 1], device=self.device).unsqueeze(-1)
        else:
            if not self.viewpoint_stack:
                self.viewpoint_stack = np.random.permutation(available_frames).tolist()
            indices = torch.tensor([self.viewpoint_stack.pop()], device=self.device).unsqueeze(-1)

        return indices
    
    def _hybrid_gs_sample(self, available_frames: int, eariliest_frame: int = 0):
        if self.add_frame:
            indices = torch.tensor([available_frames - 1], device=self.device).unsqueeze(-1)
        else:
            rand_sample = np.random.choice(self.config.multi_sampler_list, p=self.multi_sampler_ratio)
            indices = getattr(self, f"_{rand_sample}_gs_sample")(available_frames, eariliest_frame)

        return indices
    
    def _hierarchical_loss_gs_sample(self, available_frames: int, eariliest_frame: int = 0):
        # recent frames
        recent_iters = 50
        recent_frames = \
            ((self.frame_timestamp[:available_frames] + recent_iters - self.train_count + 1) >= 0).sum().item()
        time_delta = self.train_count - 1 - self.frame_timestamp[eariliest_frame:available_frames] + 1e-3
        scale, ratio = self.config.delta_sample_scale, self.config.delta_sample_ratio
        sample_weight = (
            torch.exp(-(time_delta - 1) * max(self.avg_frame_speed * scale, 0.01)) + ratio / available_frames
        )
        if recent_frames == available_frames:
            sample_recent = True
        elif recent_frames == 0:
            sample_recent = False
        else:
            ratio = (sample_weight[-recent_frames:].sum() / sample_weight.sum()).item()
            sample_recent = np.random.rand() < ratio

        if sample_recent:
            indices = self._loss_weighted_frame_sample(available_frames, 1, available_frames - recent_frames, True)
        else:
            indices = self._loss_weighted_frame_sample(available_frames - recent_frames, 1, eariliest_frame, True)

        return indices



    #########################################

    def next_train(self, step: int) -> Tuple[Union[Cameras, RayBundle], Dict]:
        """
        First, checks for updates to the ROSDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        if self.dummy_batch:
            batch = self.train_pixel_sampler.sample(self.train_image_dataloader._get_updated_batch(3))
            ray_indices = batch["indices"]
            ray_bundle = self.train_ray_generator(ray_indices)
            return ray_bundle, batch

        image_batch = next(self.iter_train_image_dataloader)
        # CONSOLE.print(f"[bold red]Updated Indices: {self.train_dataset.updated_indices}[/bold red]]")
        # simulate transmission
        if self.replay_trans_log:
            if self.train_count < len(self.avail_images_per_iter):
                received_image_number = self.avail_images_per_iter[self.train_count]  
            else:
                received_image_number = self.avail_images_per_iter[-1]
            # if received_image_number > self.current_avail_images:
            #     if self.config.use_camopt_dataloader and self.current_avail_images > 0:
            #         # To fix the accumulated pose adjustment. consistent with the true online mode.
            #         camera_optimizer = self.train_image_dataloader.camera_optimizer
            #         camera_optimizer.pose_adjustment.data[self.current_avail_images:received_image_number - 1, :] = \
            #             camera_optimizer.pose_adjustment.data[self.current_avail_images - 1, :]
            
            _batch = {}
            for k, v in image_batch.items():
                _batch[k] = v[:received_image_number]
            image_batch = _batch

        self.train_count += 1
        received_image_number, h, w, _ = image_batch["image"].shape
        self.current_avail_images = received_image_number

        if self.config.track_avail_images_per_iter:
            self.avail_images_per_iter.append(received_image_number)

        if self.config.use_gaussian_splatting:

            self.add_frame = self.current_avail_images > self.previous_avail_images

            indices = getattr(self, f"_{self.config.training_sampler}_gs_sample")(
                received_image_number, 0
            )
            # import IPython; IPython.embed()
            camera = self.train_dataset.cameras[indices[0]: indices[0] + 1].to(self.device)
            camera.camera_to_worlds = camera.camera_to_worlds.clone()
            if camera.metadata is None:
                camera.metadata = {}
            cam_idx = indices[0].item()
            camera.metadata["cam_idx"] = cam_idx
            if self.config.track_sample_cnt:
                camera.metadata["sample_cnt"] = self.frame_sample_cnt[cam_idx].item()
                
            data = {}

            data["image"] = image_batch["image"][indices[0].item()].to(self.device)
            data["image_idx"] = indices[0]

            batch = {"indices": indices.reshape(-1, 1)}

            self.current_batch = batch
            return camera, data
        else:
            num_rays_per_batch = self.train_pixel_sampler.num_rays_per_batch
            indices = getattr(self, f"_{self.config.training_sampler}_sample")(
                received_image_number, num_rays_per_batch
            )

            c, y, x = (i.flatten().cpu() for i in torch.split(indices, 1, dim=-1))
            sampled_rgb_value = image_batch["image"][c, y, x]
            batch = {"image": sampled_rgb_value, "indices": indices}

            ray_indices = batch["indices"]
            ray_bundle = self.train_ray_generator(ray_indices)

            # keep a copy of the current batch for loss tracking
            self.current_batch = batch

            return ray_bundle, batch

    def setup_eval(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        CONSOLE.print("Setting up evaluation dataset...")
        assert self.eval_dataset is not None
        self.eval_image_dataloader = ROSDataloader(
            self.eval_dataset,
            self.config.publish_training_posearray,
            self.config.receive_pcd,
            self.config.data_update_freq,
            self.config.use_camopt_dataloader,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras)

        # for loading full images
        self.fixed_indices_eval_dataloader = ROSFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=0,
        )
        self.fixed_indices_eval_dataloader.cameras = self.eval_dataset.cameras
        self.eval_dataloader = ROSRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=0,
        )
        self.eval_dataloader.cameras = self.eval_dataset.cameras

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        if self.config.eval_with_training_set:
            eval_dataset = self.train_dataset
        else:
            eval_dataset = ROSDataset(
                dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split), device=self.device
            )
        return eval_dataset

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print(f"Next Eval: {step}")
        self.eval_image_dataloader.updated = True
        self.eval_image_dataloader.current_idx = self.train_image_dataloader.current_idx

        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)

        if self.config.use_gaussian_splatting:
            image_idx = torch.randint(0, self.eval_image_dataloader.current_idx + 1, (1,)).item()
            camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx

            data = {}

            data["image"] = image_batch["image"][image_idx].to(self.device)
            data["image_idx"] = image_idx

            batch = {"indices": image_idx}

            self.current_batch = batch

            return camera, data
        else:
            batch = self.eval_pixel_sampler.sample(image_batch)
            ray_indices = batch["indices"]
            ray_bundle = self.eval_ray_generator(ray_indices)
            return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print(f"Next Eval Image: {step}")
        self.eval_image_dataloader.updated = True
        self.eval_image_dataloader.current_idx = self.train_image_dataloader.current_idx

        if self.config.use_gaussian_splatting:
            image_batch = next(self.iter_eval_image_dataloader)
            image_idx = torch.randint(0, self.eval_image_dataloader.current_idx, (1,)).item()
            camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx

            data = {}

            data["image"] = image_batch["image"][image_idx].to(self.device)
            data["image_idx"] = image_idx

            batch = {"indices": image_idx}

            self.current_batch = batch

            return camera, data
        else:
            for camera_ray_bundle, batch in self.eval_dataloader:
                assert camera_ray_bundle.camera_indices is not None
                image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
                return image_idx, camera_ray_bundle, batch
            raise ValueError("No more eval images")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""

        def track_frame_timestamp_status(self, step: int):
            if self.current_avail_images > self.previous_avail_images:
                self.frame_timestamp[self.previous_avail_images : self.current_avail_images] = step
                self.previous_avail_images = self.current_avail_images
                self.last_timestamp = step
                if step > 0 and self.current_avail_images > self.config.num_msgs_to_start:
                    self.avg_frame_speed = (self.current_avail_images - self.config.num_msgs_to_start) / step

        def track_sample_cnt_status(self, step: int):
            image_indices = self.current_batch["indices"][:, 0]
            # self.frame_sample_cnt[image_indices] += 1
            self.frame_sample_cnt.index_add_(
                0, image_indices, torch.ones_like(image_indices, dtype=self.frame_sample_cnt.dtype)
            )
            # if step >= 1500:
            #     import IPython; IPython.embed(); exit(1)

        def track_frame_loss_status(self, step: int):
            batch = self.current_batch
            batch_metric = self.current_batch_metric
            avail_images = self.current_avail_images
            frame_indices = batch["indices"][:, 0]
            avail_frame_loss = self.frame_loss[: avail_images + 1]
            avail_frame_loss *= 0.95
            curr_batch_loss = self.frame_loss[frame_indices]
            self.frame_loss[frame_indices] = torch.maximum(curr_batch_loss, batch_metric)

        def track_perpixel_loss_status(self, step: int):
            batch = self.current_batch
            batch_metric = self.current_batch_metric
            avail_images = self.current_avail_images
            pix_indices = batch["indices"]  # [image_idx, height_idx, width_idx]
            avail_pixel_loss = self.pixel_loss[: avail_images + 1]
            avail_patch_loss = self.patch_loss[: avail_images + 1]
            avail_frame_loss = self.frame_loss[: avail_images + 1]
            avail_patch_cnt = self.pixel_sample_cnt[: avail_images + 1]
            avail_pixel_loss *= 0.99
            curr_batch_loss = self.pixel_loss[pix_indices[:, 0], pix_indices[:, 1], pix_indices[:, 2]]
            self.pixel_loss[pix_indices[:, 0], pix_indices[:, 1], pix_indices[:, 2]] = torch.maximum(
                curr_batch_loss, batch_metric
            )
            self.pixel_sample_cnt[pix_indices[:, 0], pix_indices[:, 1], pix_indices[:, 2]] += 1
            avg_patch_loss = torch.nn.functional.avg_pool2d(
                avail_pixel_loss, self.patch, ceil_mode=False, divisor_override=1
            )  # [Bs, Ps, Ps]
            avg_patch_cnt = torch.nn.functional.avg_pool2d(
                (avail_patch_cnt > 0).float(), self.patch, ceil_mode=False, divisor_override=1
            )  # [Bs, Ps, Ps]
            avail_patch_loss[:] = avg_patch_loss / (avg_patch_cnt + 1e-3)  # [Bs, Ps, Ps]
            avail_frame_loss[:] = avail_patch_loss.sum(dim=(1, 2)) / (
                (avail_patch_loss != 0).sum(dim=(1, 2)) + 1e-3
            )  # [Bs]
            # [Bs]

        callbacks = []

        if self.config.track_frame_timestamp:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=track_frame_timestamp_status,
                    update_every_num_iters=1,
                    args=[self],
                )
            )

        if self.config.track_sample_cnt:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=track_sample_cnt_status,
                    update_every_num_iters=1,
                    args=[self],
                )
            )

        if self.config.track_perpixel_loss:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=track_perpixel_loss_status,
                    update_every_num_iters=1,
                    args=[self],
                )
            )

        if self.config.track_frame_loss:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=track_frame_loss_status,
                    update_every_num_iters=1,
                    args=[self],
                )
            )

        return callbacks


def multinomial_ray_sample(sample_weight, number_samples, h, w, device, image_indices_only=False):
    # Select image indices based on the specified probabilities
    # image_indices = np.random.choice(received_image_number, size=self.config.train_num_rays_per_batch, p=sampling_required)
    image_indices = torch.multinomial(sample_weight, number_samples, replacement=True)

    if image_indices_only:
        return image_indices, None

    # Select random coordinates within each chosen image
    # height_indices = np.random.choice(h, size=self.config.train_num_rays_per_batch)
    # width_indices = np.random.choice(w, size=self.config.train_num_rays_per_batch)
    pix_indices = torch.floor(
        torch.rand((number_samples, 2), device=device) * torch.tensor([h, w], device=device)
    ).long()
    # Combine the selected indices into a single array of samples
    # indices = np.stack([image_indices, height_indices, width_indices], axis=1)
    # sampled_rgb_value = image_batch["image"][indices[:, 0], indices[:, 1], indices[:, 2]]

    return image_indices, pix_indices


def plot_samples_debug(image_indices, train_count, output_dir):
    # debug = True
    # if debug and (train_count - 1) % 5 == 0:
    import matplotlib.pyplot as plt

    unique, counts = np.unique(image_indices.cpu().numpy(), return_counts=True)
    plt.figure(figsize=(12, 4))
    plt.bar(unique, counts)
    plt.xlim(0, 150)
    plt.ylim(0, 4096)
    plt.title(f"Sampling Distribution | Iteration {train_count - 1}")
    save_path = output_dir / "sample_distr"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path / f"sample_cnt_{train_count - 1:05d}.jpg")
    plt.close()
