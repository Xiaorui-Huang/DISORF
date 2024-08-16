# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""

import json
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from jaxtyping import Float
from message_filters import Subscriber, TimeSynchronizer
from PIL import Image as PILImage
from rich.console import Console
from scipy.spatial import transform
from sensor_msgs.msg import Image, PointCloud2
from torch import Tensor

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import DataLoader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.misc import get_dict_to_torch
from server_end.nsros.ros_dataset import ROSDataset

CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


def ros_pose_to_nerfstudio(pose: PoseStamped, static_transform=None):
    """
    Takes a ROS Pose message and converts it to the
    3x4 transform format used by nerfstudio.
    """
    pose_msg = pose.pose
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    position = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    R = torch.tensor(qvec2rotmat(quat))
    T = torch.cat([R, position.unsqueeze(-1)], dim=-1)
    T = T.to(dtype=torch.float32)
    if static_transform is not None:
        T = pose_utils.multiply(T, static_transform)
        T2 = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", -90, degrees=True).as_matrix()
        R = torch.from_numpy(R1)
        T2[:, :3] = R
        T = pose_utils.multiply(T2, T)

    return T.to(dtype=torch.float32)


def quaternion_to_transform_matrix(pose: Pose):
    """
    Takes a ROS Pose message and converts it to the
    4x4 transform matrix.
    """
    if isinstance(pose, PoseStamped):
        pose_msg = pose.pose
    else:
        pose_msg = pose
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    position = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    rotation_matrix = qvec2rotmat(quat)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position

    return transform_matrix


def relative_transformation_3x4(A, B):
    # Ensure A and B are both 3x4 matrices
    assert A.shape == (3, 4) and B.shape == (3, 4), "Both input matrices should be of shape 3x4"

    # Compute relative rotation using matrix slicing
    R_rel = torch.mm(B[:, :3], A[:, :3].transpose(0, 1))

    # Compute relative translation using matrix slicing
    t_rel = B[:, 3:].reshape(-1, 1) - torch.mm(R_rel, A[:, 3:].reshape(-1, 1))

    # Concatenate results
    T_rel = torch.cat([R_rel, t_rel], dim=1)

    return T_rel


def apply_relative_motion(R_combined, C):
    # Ensure the matrices are of correct size
    if R_combined.size() != (3, 4) or C.size() != (3, 4):
        raise ValueError("Both R_combined and C must have shape (3,4)")

    # Extract rotation and translation parts using matrix slicing
    R1 = R_combined[:, :3]
    t1 = R_combined[:, 3:].reshape(3, 1)

    R2 = C[:, :3]
    t2 = C[:, 3:].reshape(3, 1)

    # Compute resulting rotation and translation
    R3 = torch.mm(R1, R2)
    t3 = torch.mm(R1, t2) + t1

    # Combine R3 and t3 to get D
    D = torch.cat((R3, t3), dim=1)

    return D


class ROSDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from ROS, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        publish_posearray: publish a PoseArray to a ROS topic that tracks the poses of the
            images that have been added to the training set.
        data_update_freq: Frequency (wall clock) that images are added to the training
            data tensors. If this value is less than the frequency of the topics to which
            this dataloader subscribes (pose and images) then this subsamples the ROS data.
            Otherwise, if the value is larger than the ROS topic rates then every pair of
            messages is added to the training bag.
        device: Device to perform computation.
    """

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        publish_posearray: bool,
        receive_pcd: bool,
        data_update_freq: float,
        use_camopt_dataloader: bool,
        device: Union[torch.device, str] = "cpu",
        camera_optimizer: Optional[CameraOptimizer] = None,
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset
        self.process = self.dataset.process
        self.disable_streaming = False

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # Tracking ros updates
        self.current_idx = len(dataset.image_filenames)
        self.updated = True
        self.finished = False
        self.update_period = 1 / data_update_freq
        self.last_update_t = time.perf_counter()
        self.publish_posearray = publish_posearray
        self.poselist = []
        self.receive_pcd = receive_pcd

        self.coord_st = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", 180, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("z", 0, degrees=True).as_matrix()
        R = torch.from_numpy(R2 @ R1)
        self.coord_st[:, :3] = R
        self.camera_optimizer = camera_optimizer
        # self.coord_st = torch.eye(4, dtype=torch.float32)[:3]
        self.first_update = True
        self.camera_relocated = False

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        self.use_camopt_dataloader = use_camopt_dataloader

        super().__init__(dataset=dataset, **kwargs)

        # All of the ROS CODE
        rospy.init_node("nsros_dataloader", anonymous=True)
        self.image_sub = Subscriber(self.dataset.image_topic_name, Image)
        self.pose_sub = Subscriber(self.dataset.pose_topic_name, PoseStamped)
        if self.receive_pcd:
            self.pcd_sub = Subscriber(self.dataset.pcd_topic_name, PointCloud2)
            self.ts = TimeSynchronizer([self.image_sub, self.pose_sub, self.pcd_sub], 10)
        else:
            self.ts = TimeSynchronizer([self.image_sub, self.pose_sub], 10)
        self.ts.registerCallback(self.ts_image_pose_callback)
        self.posearray_pub = rospy.Publisher("training_poses", PoseArray, queue_size=1)

    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= num_to_start

    def ts_image_pose_callback(self, image: Image, pose: PoseStamped, pcd: Optional[PointCloud2] = None):
        """
        The callback triggered when time synchronized image and pose messages
        are published on the topics specifed in the config JSON passed to
        the ROSDataParser.
        """
        if self.process != "train" or self.disable_streaming:
            return

        now = time.perf_counter()
        # if now - self.last_update_t > self.update_period and # no more 
        if self.current_idx < self.num_images:
            if pose.pose.position.z < -99:
                self.finished = True
                return
            if self.current_idx not in self.dataset.updated_indices:
                # ----------------- Handling the IMAGE ----------------
                # Load the image message directly into the torch
                im_tensor = torch.frombuffer(image.data, dtype=torch.uint8)

                im_tensor = im_tensor.reshape(self.H, self.W, -1)
                # Convert BGR -> RGB
                channel_idx = torch.tensor([2, 1, 0])
                im_tensor = im_tensor[..., channel_idx]

                # Normalize Tensor
                im_tensor = im_tensor.to(dtype=torch.float32) / 255.0
                # COPY the image data into the data tensor
                self.dataset.image_tensor[self.current_idx] = im_tensor

                # ----------------- Handling the POSE ----------------
                scale, translation = 1, 0
                if not self.camera_relocated and self.dataset.camera_trans is not None:
                    # update the camera poses of all received images
                    scale, translation = self.dataset.camera_trans
                    c2ws = self.dataset.cameras.camera_to_worlds
                    c2ws[: self.current_idx, :3, 3] = c2ws[: self.current_idx, :3, 3] * scale + translation
                    self.camera_relocated = True

                c2w = ros_pose_to_nerfstudio(pose, static_transform=self.coord_st)
                device = self.dataset.cameras.device
                c2w = c2w.to(device)

                # TODO: Fix online
                if self.use_camopt_dataloader:
                    if self.current_idx > 0:
                        self.camera_optimizer.pose_adjustment.data[self.current_idx, :] = (
                            self.camera_optimizer.pose_adjustment.data[self.current_idx - 1, :]
                        )

                if self.dataset.camera_trans is not None:
                    scale, translation = self.dataset.camera_trans
                    c2w[:3, 3] = c2w[:3, 3] * scale + translation

                self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w
                self.poselist.append(pose.pose)

                if self.publish_posearray:
                    pa = PoseArray(poses=self.poselist)
                    pa.header.frame_id = "map"
                    self.posearray_pub.publish(pa)

                # ----------------- Handling the PCD ----------------
                if self.receive_pcd and pcd is not None:
                    row_step = pcd.row_step
                    height = pcd.height
                    pcd_tensor = torch.frombuffer(pcd.data, dtype=torch.uint8)

                    pcd_tensor = pcd_tensor.reshape(height, row_step)
                    # only accept two kinds of point cloud data
                    if row_step == 16:
                        # x y z frame_id
                        pcd_tensor = pcd_tensor[:, :12]
                        pcd_tensor = pcd_tensor.view(-1, 4)
                        pcd_tensor = pcd_tensor[:, :3].float()
                        # empty color np tensor
                        pcd_color = torch.zeros((0, 3))
                        self.dataset.pcd = (pcd_tensor, pcd_color)
                    elif row_step == 32:
                        # x y z r g b frame_id
                        pcd_tensor = pcd_tensor[:, :24]
                        pcd_tensor = pcd_tensor.view(-1, 6)
                        pcd_tensor = pcd_tensor[:, :3].float()
                        pcd_color = pcd_tensor[:, 3:].float()

                        # concatenate the point cloud data
                        self.dataset.pcd = (torch.cat((self.dataset.pcd[0], pcd_tensor), dim=0), torch.cat((self.dataset.pcd[1], pcd_color), dim=0))
                        # record the number of points till this frame
                        self.dataset.f2p[self.current_idx] = self.dataset.pcd[0].shape[0]
                    else:
                        raise ValueError("Unsupported point cloud data format")
                elif self.receive_pcd and pcd is None:
                    # no new pcd data with this image
                    self.dataset.f2p[self.current_idx] = self.dataset.f2p[self.current_idx - 1]
                else:
                    pass

                # First Update
                if self.first_update:
                    CONSOLE.log(f"Current Index: {self.current_idx}")
                    self.first_update = False
                    # no need to duplicate the first frame
                    # self.dataset.image_tensor[self.current_idx:] = im_tensor
                    self.dataset.cameras.camera_to_worlds[self.current_idx :] = c2w

                self.dataset.updated_indices.append(self.current_idx)
                # print(f"Updated Indices: {self.dataset.updated_indices}")

                self.current_idx += 1
                self.updated = True

            self.last_update_t = now

    def dump_data(self):
        number_frames = len(self.dataset.updated_indices)
        meta_file = self.dataset.data_path.parent / "transforms.json"
        meta = load_from_json(meta_file)
        frames = []
        for i in range(number_frames):
            output_path = str(self.dataset.data_path / f"frame_{i:04d}.jpg")
            PILImage.fromarray((self.dataset.image_tensor[i] * 255.0).byte().numpy()).save(output_path)
            frames.append(
                {
                    "file_path": str(Path("images") / f"frame_{i:04d}.jpg"),
                    "transform_matrix": quaternion_to_transform_matrix(self.poselist[i]).tolist(),
                }
            )
        meta["frames"] = frames
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=4)
            
        print(f"Dumped {number_frames} frames to {meta_file}")

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self, idx: int = None):
        batch = {}
        idx = self.current_idx if idx is None else idx
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch


def _get_data_from_image_idx(
    dataloader: DataLoader, image_idx: int, camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None
) -> Tuple[RayBundle, Dict]:
    """Returns the data for a specific image index.

    Args:
        image_idx: Camera image index
    """
    self = dataloader
    ray_bundle = self.cameras.generate_rays(
        camera_indices=image_idx, keep_shape=True, camera_opt_to_camera=camera_opt_to_camera
    )
    batch = self.input_dataset[image_idx]
    batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
    assert isinstance(batch, dict)
    return ray_bundle, batch


class ROSFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):
    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, image_indices, device, **kwargs)

    def __iter__(self):
        self.image_indices = list(range(len(self.dataset)))
        self.cameras = self.dataset.cameras.to(self.device)
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            camera, batch = self.get_camera(image_idx)
            if camera.metadata is not None:
                camera.metadata["cam_idx"] = image_idx  # see CameraOptimizer Class: apply_to_camera()
            else:
                camera.metadata = {"cam_idx": image_idx}
            self.count += 1
            return camera, batch
        raise StopIteration


class ROSRandIndicesEvalDataloader(RandIndicesEvalDataloader):
    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)

    def __next__(self):
        num_images = len(self.dataset)
        self.cameras = self.dataset.cameras.to(self.device)
        # choose a random image index
        image_idx = random.randint(0, num_images - 1)
        camera, batch = self.get_camera(image_idx)
        if camera.metadata is not None:
            camera.metadata["cam_idx"] = image_idx  # see CameraOptimizer Class: apply_to_camera()
        else:
            camera.metadata = {"cam_idx": image_idx}
        return camera, batch
