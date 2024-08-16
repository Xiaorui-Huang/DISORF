import torch
import numpy as np

from torch.nn import functional as F
from nerfstudio.cameras.cameras import Cameras


def relocate_camera(cameras: Cameras, max_dist_thresh: float = 0.4, avg_dist_thresh: float = 0.15, init_box_scale: float = 0.3):
    """Relocate camera by a random amount.

    Args:
        cameras: Cameras object.
        max_dist_thresh: Maximum distance threshold.
        avg_dist_thresh: Average distance threshold.
        init_box_scale: Initial box scale, bbox: [-s, -s, -s, s, s, s].
    Output:
        scale: Scale factor.
        translation: Translation vector.
        Note: xyz_new = xyz * scale + translation
    """
    num_cameras = cameras.shape[0]
    if isinstance(cameras, torch.Tensor):
        camera_to_worlds = cameras
    else:
        camera_to_worlds = cameras.camera_to_worlds # (N, 3, 4)
    camera_orientation = F.normalize(camera_to_worlds[:, :3, 2], dim=-1) # (N, 3) image center to origin
    camera_position = camera_to_worlds[:, :3, 3] # (N, 3) 
    
    avg_orientation = F.normalize(torch.sum(camera_orientation, dim=0, keepdim=True), dim=-1) # (1, 3)

    # piecewise distance
    camera_distance = torch.norm(camera_position[1:] - camera_position[:-1], dim=-1) # (N-1)
    avg_distance = torch.mean(camera_distance).item() # (1)
    max_distance = torch.max(camera_distance).item() # (1)

    # determine the scale factor
    avg_scale, max_scale = 1.0, 1.0
    if avg_distance > avg_dist_thresh:
        avg_scale = avg_dist_thresh / avg_distance
    if max_distance > max_dist_thresh:
        max_scale = max_dist_thresh / max_distance

    scale = min(avg_scale, max_scale)

    scaled_camera_position = camera_position * scale # (N, 3) 
    avg_position = torch.mean(scaled_camera_position, dim=0, keepdim=True) # (1, 3)

    # determine the intersection of the camera orientation and bbox on xy plane
    # x = t * dir.x; y = t * dir.y
    t = init_box_scale / (avg_orientation + 1e-5) # (1, 3)
    t = torch.min(torch.abs(t))
    translation = t * avg_orientation * torch.tensor([1, 1, 0], device=avg_orientation.device) # only on xy plane
    translation = translation - avg_position # (1, 3)

    return scale, translation
