from typing import Union

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from PIL import Image


class ROSDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in ROSDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "image_topic" in dataparser_outputs.metadata.keys()
            and "pose_topic" in dataparser_outputs.metadata.keys()
            and "pcd_topic" in dataparser_outputs.metadata.keys()
            and "num_images" in dataparser_outputs.metadata.keys()
        )
        self.image_topic_name = self.metadata["image_topic"]
        self.pose_topic_name = self.metadata["pose_topic"]
        self.pcd_topic_name = self.metadata["pcd_topic"]
        self.num_images = self.metadata["num_images"]
        assert self.num_images > 0
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.device = device
        self.num_avail_images = 0

        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        )

        self.f2p = self.metadata["frame2point"]
        self.pcd = (self.metadata["points3D_xyz"], self.metadata["points3D_rgb"])

        self.updated_indices = []
        self.camera_trans = None
        self.camera_trans_meta = dataparser_outputs.metadata["camera_trans"] \
            if "camera_trans" in dataparser_outputs.metadata.keys() else None

        if self.image_filenames != []:
            for image_idx in range(len(self.image_filenames)):
                self.image_tensor[image_idx] = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
            self.num_avail_images = len(self.image_filenames)
            self.updated_indices = list(range(self.num_avail_images))

        self.image_indices = torch.arange(self.num_images)

        self.data_path = self.metadata["data_path"]

        self.process = self.metadata["process"]

    def __len__(self):
        self.num_avail_images = len(self.updated_indices)
        if self.num_avail_images == 0:
            return self.num_images
        else:
            return min(self.num_avail_images, self.num_images)

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {"image_idx": idx, "image": self.image_tensor[idx]}
        return data

    def dump_dataset(self):
        """
        Dump the online received training data to disk.
        Call this function when the training is done, avoiding frequent disk IO.
        Another benifit is that we can directly dump optimized cameras, helpful for the later evaluation.
        """
        pass