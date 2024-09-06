from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
import functools

from rich import box, style
from rich.panel import Panel
from rich.console import Console
from rich.table import Table

from nerfstudio.utils import profiler, writer
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.decorators import check_viewer_enabled
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler
from server_end.nsros.ros_dataset import ROSDataset
from server_end.nsros.ros_datamanager import ROSDataManager
from server_end.nsros.ros_pipeline import ROSPipeline
from server_end.nsros.ros_utils import relocate_camera
from nerfstudio.engine.callbacks import TrainingCallbackLocation

import viser
import viser.transforms as vtf
# import torchvision

import numpy as np
from nerfstudio.configs import base_config as cfg


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
CONSOLE = Console(width=120)


def p_scheduler_step_all(optimizer, step: int, p_step: int) -> None:
    """Run step for all schedulers.

    Args:
        step: the current step
    """
    for param_group_name, scheduler in optimizer.schedulers.items():
        scheduler.step(p_step)
        lr = scheduler.get_last_lr()[0]
        writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)


@dataclass
class ROSTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: ROSTrainer)
    msg_timeout: float = 6000.0
    """ How long to wait (seconds) for sufficient images to be received before training. """
    num_msgs_to_start: int = 3
    """ Number of images that must be recieved before training can start. """
    draw_training_images: bool = False
    """ Whether or not to draw the training images in the viewer. """
    disable_streaming: bool = False
    """ Whether to disable streaming."""
    relocate_cameras: bool = False
    """ Whether to relocate cameras. """
    dump_loss_track: bool = False
    """ Whether to dump loss track. """
    dump_received_images: bool = False
    """ Whether to dump received images. """
    auto_transmission: bool = False
    """ Whether to auto start transmission. """
    dataset_name: str = "replica"


class ROSTrainer(Trainer):
    config: ROSTrainerConfig
    dataset: ROSDataset

    def __init__(self, config: ROSTrainerConfig, local_rank: int = 0, world_size: int = 0):
        # We'll see if this throws and error (it expects a different config type)
        super().__init__(config, local_rank=local_rank, world_size=world_size)
        self.msg_timeout = self.config.msg_timeout
        self.cameras_drawn = set()
        self.first_update = True
        self.camera_relocated = False
        self.num_msgs_to_start = config.num_msgs_to_start
        self.track_perpixel_loss = False
        self.track_frame_loss = False
        self.finished_iter = 0
        self.per_frame_schedule = False

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """
        Runs the Trainer setup, and then waits until at least one image-pose
        pair is successfully streamed from ROS before allowing training to proceed.
        """
        # This gets called in the script that launches the training.
        # In this case ns_ros/ros_train.py
        super().setup(test_mode=test_mode)

        if isinstance(self.pipeline.datamanager, ROSDataManager):
            self.pipeline.datamanager.train_image_dataloader.camera_optimizer = self.pipeline.model.camera_optimizer
            self.pipeline.datamanager.dummy_batch = True
            cpu_or_cuda_str: str = self.device.split(":")[0]
            try:
                if self.config.method_name == "ros_instant_ngp":
                    for callback in self.callbacks:
                        callback.run_callback_at_location(0, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    model_outputs, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(-1)
                    loss = functools.reduce(torch.add, loss_dict.values())
                self.grad_scaler.scale(loss).backward()  # type: ignore
            except Exception as e:
                CONSOLE.print(f"[bold red] (NSROS) Dummy batch failed, continue anyway.")

            self.pipeline.datamanager.dummy_batch = False

        self.pipeline.datamanager.config.num_msgs_to_start = self.num_msgs_to_start

        start = time.perf_counter()
        if self.config.disable_streaming:
            CONSOLE.print(f"[bold green] (NSROS) Streaming is disabled. No more images will be loaded ....")
            self.pipeline.datamanager.train_image_dataloader.disable_streaming = True
        else:
            # Start Status check loop
            status = False
            CONSOLE.print(f"[bold green] (NSROS) Waiting for image streaming to begin ....")
            # transmission auto start, some dirty codes here for quick logging.
            if self.config.auto_transmission:
                CONSOLE.print(f"data: {self.config.data}")
                scene_name = self.config.data.parts[-1]
                trans_workspace = os.environ.get("ROS_TRANS_WORKSPACE", "~/Project")
                if self.config.dataset_name == "replica":
                    # Replica
                    data_dir = os.path.join(trans_workspace, "nerfbridge_experiment/data/replica_slam/undistorted")
                    frame_rate = 30
                elif self.config.dataset_name == "tnt":
                    # TNT
                    data_dir = os.path.join(
                        trans_workspace, "nerfbridge_experiment/data/tank_and_temples_colmap/exp_720p"
                    )
                    frame_rate = 1
                elif self.config.dataset_name == "tnt_slam":
                    # TNT+SLAM
                    data_dir = os.path.join(trans_workspace, "nerfbridge_experiment/data/tank_and_temples_slam")
                    scene_name = "pcgpu_" + scene_name.lower()
                    frame_rate = 30
                elif self.config.dataset_name == "tnt_droid":
                    # TNT+Droid
                    data_dir = os.path.join(trans_workspace, "nerfbridge_experiment/data/tank_and_temples_droid")
                    frame_rate = 23
                script = os.path.join(trans_workspace, "nerfbridge_experiment/ros_posed_img_transfer.py")
                cmd = (
                    f"python {script} "
                    + f"--img_dir {data_dir}/{scene_name} "
                    + f"--pose_file {data_dir}/{scene_name}.txt "
                    + f"--frame_rate {frame_rate}"
                )
                CONSOLE.print(f"[bold red] (NSROS) Auto transmission start: {cmd}")
                os.system(cmd + " > /dev/null 2>&1 &")

            while time.perf_counter() - start < self.msg_timeout:
                if self.pipeline.datamanager.train_image_dataloader.msg_status(  # pyright: ignore
                    self.num_msgs_to_start
                ):
                    status = True
                    break
                time.sleep(0.03)

            if not status:
                raise NameError(
                    "ROSTrainer setup() timed out, check that topics are being published \
                    and that config.json correctly specifies their names."
                )
            else:
                CONSOLE.print("[bold green] (NSROS) Dataloader is successfully streaming images!")

        self.dataset = self.pipeline.datamanager.train_dataset  # pyright: ignore

        if self.config.relocate_cameras:
            # TODO: note that we currently only relocate the training cameras
            if self.dataset.camera_trans_meta is not None:
                scale, translation = self.dataset.camera_trans_meta
                if isinstance(translation, list):
                    translation = torch.tensor(translation, device=self.dataset.cameras.device)
            else:
                scale, translation = relocate_camera(self.dataset.cameras[: self.num_msgs_to_start])

            CONSOLE.print(f"[bold green] (NSROS) Relocating cameras by {translation} and scaling by {scale} ....")
            if self.config.disable_streaming:
                self.dataset.cameras.camera_to_worlds[:, :3, 3] = (
                    self.dataset.cameras.camera_to_worlds[:, :3, 3] * scale + translation
                )
                # import IPython; IPython.embed()
            self.dataset.camera_trans = (scale, translation)
        else:
            self.dataset.camera_trans = None

        if isinstance(self.pipeline.datamanager, ROSDataManager):
            self.track_perpixel_loss = self.pipeline.datamanager.config.track_perpixel_loss
            self.track_frame_loss = self.pipeline.datamanager.config.track_frame_loss

        # if isinstance(self.pipeline, ROSPipeline):
        self.pipeline.config.output_dir = self.base_dir

        if isinstance(self.pipeline.datamanager, ROSDataManager):
            self.pipeline.datamanager.config.output_dir = self.base_dir

        if hasattr(self.pipeline.model.config, "per_frame_schedule"):
            self.per_frame_schedule = self.pipeline.model.config.per_frame_schedule

        if self.config.method_name == "ros_gaussian_splatting":
            self.pipeline.model.mvs_point_init(self.pipeline)


    @check_viewer_enabled
    def _update_viewer_state(self, step: int, update_all=False):
        """
        Updates the viewer state by rendering out scene with current pipeline

        Args:
            step: current train step
        """
        super()._update_viewer_state(step)
        #
        # # Clear any old cameras!
        if self.config.draw_training_images:
            if self.first_update:
                self.first_update = False

            if self.pipeline.datamanager.train_image_dataloader.camera_relocated and not self.camera_relocated:
                self.camera_relocated = True
                update_all = True

            # Draw any new training images
            image_indices = self.dataset.updated_indices
            for idx in image_indices:
                if update_all or idx not in self.cameras_drawn:
                    # Do a copy here just to make sure we aren't
                    # changing the training data downstream.
                    # TODO: Verify if we need to do this
                    # image = self.dataset[idx]["image"]
                    # bgr = image[..., [2, 1, 0]]
                    # camera_json = self.dataset.cameras.to_json(
                    #     camera_idx=idx, image=bgr, max_size=10
                    # )

                    # self.viewer_state.viser_server.add_dataset_image(idx=f"{idx:06d}",
                    #                                                  json=camera_json)
                    # self.cameras_drawn.add(idx)
                    image = self.dataset[idx]["image"]
                    camera = self.dataset.cameras[idx]
                    image_uint8 = (image * 255).detach().type(torch.uint8)
                    image_uint8 = image_uint8.permute(2, 0, 1)

                    import torchvision

                    image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)
                    image_uint8 = image_uint8.permute(1, 2, 0)
                    image_uint8 = image_uint8.cpu().numpy()
                    c2w = camera.camera_to_worlds.cpu().numpy()
                    R = vtf.SO3.from_matrix(c2w[:3, :3])
                    R = R @ vtf.SO3.from_x_radians(np.pi)

                    camera_handle = self.viewer_state.viser_server.add_camera_frustum(
                        name=f"/cameras/camera_{idx:05d}",
                        fov=float(2 * np.arctan((camera.cx / camera.fx[0]).cpu())),
                        scale=cfg.ViewerConfig.camera_frustum_scale,
                        aspect=float(camera.cx[0] / camera.cy[0]),
                        image=image_uint8,
                        wxyz=R.wxyz,
                        position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                    )

                    @camera_handle.on_click
                    def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                        with event.client.atomic():
                            event.client.camera.position = event.target.position
                            event.client.camera.wxyz = event.target.wxyz

                    self.viewer_state.camera_handles[idx] = camera_handle
                    self.viewer_state.original_c2w[idx] = c2w

                    self.cameras_drawn.add(idx)

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            model_outputs, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        # TODO: per-frame lr scheduling
        if self.per_frame_schedule and "p_step" in model_outputs:
            p_step = model_outputs["p_step"]
            p_scheduler_step_all(self.optimizers, step, p_step)

        self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if not self.per_frame_schedule and scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        if self.track_perpixel_loss:
            assert isinstance(self.pipeline.datamanager, ROSDataManager)
            rbg_pred = model_outputs["rgb"].detach()
            rgb = self.pipeline.datamanager.current_batch["image"].to(self.device)
            with torch.no_grad():
                self.pipeline.datamanager.current_batch_metric = torch.abs(rgb - rbg_pred).sum(-1)
        
        if self.track_frame_loss:
            assert isinstance(self.pipeline.datamanager, ROSDataManager)
            self.pipeline.datamanager.current_batch_metric = loss_dict['main_loss'].detach()

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                if self.pipeline.datamanager.train_image_dataloader.finished:
                    self.finished_iter += 1
                    if self.finished_iter > 100:
                        break
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()
