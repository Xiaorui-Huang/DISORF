"""Data parser for loading ROS parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from rich.console import Console
from scipy.spatial.transform import Rotation

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json, write_to_json

CONSOLE = Console(width=120)


@dataclass
class ROSDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: ROSDataParser)
    """target class to instantiate"""
    data: Path = Path("data/ros/default")
    """ Path to save SLAM frame. """
    ros_data: Path = Path("data/ros/nsros_config.json")
    """ Path to configuration JSON. """
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    aabb_scale: float = 2.0
    """ SceneBox aabb scale."""
    use_cached_training_data: bool = False
    """ Whether to use cached training data. """
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""


@dataclass
class ROSDataParser(DataParser):
    """ROS DataParser"""

    config: ROSDataParserConfig

    def __init__(self, config: ROSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.ros_data: Path = config.ros_data
        self.scale_factor: float = config.scale_factor
        self.aabb = config.aabb_scale

    def get_dataparser_outputs(self, split="train", num_images: int = 300):
        dataparser_outputs = self._generate_dataparser_outputs(split, num_images)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train", num_images: int = 300):
        """
        This function generates a DataParserOutputs object. Typically in Nerfstudio
        this is used to populate the training and evaluation datasets, but since with
        NSROS Bridge our aim is to stream the data then we only have to worry about
        loading the proper camera parameters and ROS topic names.

        Args:
            split: Determines the data split (not used, but left in place for consistency
                with Nerfstudio)

            num_images: The size limit of the training image dataset. This is used to
                pre-allocate tensors for the Cameras object that tracks camera pose.
        """
        meta = load_from_json(self.ros_data)

        image_height = meta["H"]
        image_width = meta["W"]
        fx = meta["fx"]
        fy = meta["fy"]
        cx = meta["cx"]
        cy = meta["cy"]

        k1 = meta["k1"] if "k1" in meta else 0.0
        k2 = meta["k2"] if "k2" in meta else 0.0
        k3 = meta["k3"] if "k3" in meta else 0.0
        k4 = meta["k4"] if "k4" in meta else 0.0
        p1 = meta["p1"] if "p1" in meta else 0.0
        p2 = meta["p2"] if "p2" in meta else 0.0
        distort = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)

        scale = meta["scale"] if "scale" in meta else None
        translation = meta["translation"] if "translation" in meta else None

        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[:, :-1, :]

        image_filenames = []

        if split in ["val", "test"]:
            CONSOLE.log(f"Process: {split}")
            transforms_meta = load_from_json(self.data / "transforms_eval.json")
            poses = []
            for frame in transforms_meta["frames"]:
                fname = self.data / Path(frame["file_path"])
                # search for png file
                if not fname.exists():
                    fname = self.data / Path(frame["file_path"] + ".png")
                if not fname.exists():
                    CONSOLE.log(f"couldn't find {fname} image")
                else:
                    image_filenames.append(fname)
                    poses.append(np.array(frame["transform_matrix"]))
            num_images = len(poses)
            camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[:, :-1, :]
            if len(poses) != 0 and len(image_filenames) == len(poses):
                poses = np.array(poses).astype(np.float32)
                poses[:, :3, 3] *= self.scale_factor

                static_transform = torch.zeros(3, 4)
                static_transform[:, :3] = torch.from_numpy(Rotation.from_euler("x", 180, degrees=True).as_matrix())

                pose_correction = torch.zeros(3, 4)
                pose_correction[:, :3] = torch.from_numpy(Rotation.from_euler("x", -90, degrees=True).as_matrix())

                camera_to_world[: len(poses)] = pose_utils.multiply(
                    pose_correction, pose_utils.multiply(torch.from_numpy(poses[:, :3]), static_transform)
                )
        elif split == "train":
            CONSOLE.log(f"Process: {split}")
            # Create Directory If Not Exists (0718 Update)
            if (self.data / "images").exists():
                if (self.data / "transforms.json").exists() and self.config.use_cached_training_data:
                    transforms_meta = load_from_json(self.data / "transforms.json")
                    poses = []
                    for frame in transforms_meta["frames"]:
                        fname = self.data / Path(frame["file_path"])
                        # search for png file
                        if not fname.exists():
                            fname = self.data / Path(frame["file_path"] + ".png")
                        if not fname.exists():
                            CONSOLE.log(f"couldn't find {fname} image")
                        else:
                            image_filenames.append(fname)
                            poses.append(np.array(frame["transform_matrix"]))
                    if len(poses) > num_images:
                        num_images = len(poses)
                        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[:, :-1, :]

                    if len(poses) != 0 and len(image_filenames) == len(poses):
                        poses = np.array(poses).astype(np.float32)
                        poses[:, :3, 3] *= self.scale_factor

                        static_transform = torch.zeros(3, 4)
                        static_transform[:, :3] = torch.from_numpy(
                            Rotation.from_euler("x", 180, degrees=True).as_matrix()
                        )

                        pose_correction = torch.zeros(3, 4)
                        pose_correction[:, :3] = torch.from_numpy(
                            Rotation.from_euler("x", -90, degrees=True).as_matrix()
                        )

                        camera_to_world[: len(poses)] = pose_utils.multiply(
                            pose_correction, pose_utils.multiply(torch.from_numpy(poses[:, :3]), static_transform)
                        )
                    elif len(image_filenames) != len(poses):
                        CONSOLE.log("FAULT: number of poses does not match number of images")
                    else:
                        # ROS Mode
                        pass
            else:
                (self.data / "images").mkdir(parents=True)
                meta["frames"] = []
                transform_json = self.data / "transforms.json"
                write_to_json(transform_json, meta)

        CONSOLE.log(f"Total: {num_images} of images")

        # in x,y,z order
        scene_size = self.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        # if scale is not None and translation is not None:
        #     camera_to_world[:, :3, 3] = camera_to_world[:, :3, 3] * scale + torch.tensor(translation)

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distort,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {
            "image_topic": meta["image_topic"],
            "pose_topic": meta["pose_topic"],
            "pcd_topic": meta["pcd_topic"],
            "num_images": num_images,
            "image_height": image_height,
            "image_width": image_width,
            "data_path": self.data / "images",
            "process": split,
        }

        metadata["points3D_xyz"] = torch.zeros((0, 3), dtype=torch.float32)
        metadata["points3D_rgb"] = torch.zeros((0, 3), dtype=torch.float32)
        metadata["frame2point"] = np.zeros(num_images, dtype=np.int8)

        if self.config.load_3D_points:
            CONSOLE.log("Loading 3D points from point cloud")
            if (self.data / "points.ply").exists() and self.config.use_cached_training_data:
                metadata.update(self._load_3D_points(self.data))
            elif (self.data / "points.ply").exists() and not self.config.use_cached_training_data:
                CONSOLE.log("Point cloud exists but not loaded since do not use cached training data")
            else:
                CONSOLE.log("Point cloud does not exist")
        else:
            CONSOLE.log("Skipping 3D point loading")

        if scale is not None and translation is not None:
            metadata["camera_trans"] = (scale, translation)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

    def _load_3D_points(self, pcd_path: Path):
        import open3d as o3d

        pcd = o3d.t.io.read_point_cloud(str(pcd_path / "points.ply"))
        if "frame_id" not in pcd.point:
            pcd.point.frame_id = np.zeros(pcd.point.colors.shape[0], dtype=np.int32)
        point2frame = pcd.point.frame_id.numpy().reshape(-1)  # (N, )
        fids, cnts = np.unique(point2frame, return_counts=True)
        frame2point = np.cumsum(cnts)

        pcd_colors = pcd.point.colors.numpy() / 255.0
        pcd_points = pcd.point.positions.numpy()

        out = {
            "points3D_xyz": pcd_points,
            "points3D_rgb": pcd_colors,
            "frame2point": frame2point,
        }

        return out
