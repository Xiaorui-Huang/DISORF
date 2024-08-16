import copy

import numpy as np
import torch
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from PIL import Image
from contextlib import contextmanager
from time import time
filler = " "

@contextmanager
def timeit(msg=None):
    start = time()
    yield
    print(f"⏰ {msg:{filler}<30} Time elapsed: {time() - start:.2f}s")

def _preset_intrinsics(scene, known_intrinsics, msk=None):
    if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
        known_intrinsics = [known_intrinsics]
    for K in known_intrinsics:
        assert K.shape == (3, 3)
    scene.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
    scene.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)


def _process_poses(poses):
    if poses[0].shape == (3, 4):
        # Convert poses to 4x4 matrices
        poses = torch.cat([poses, torch.tensor([0, 0, 0, 1]).view(1, 1, 4).expand(poses.shape[0], -1, -1)], dim=1)
    return poses


def _process_intrinsics(intrinsics, n_samples):
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.expand(n_samples, -1, -1)
    return intrinsics



def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


class SceneReconstructor:
    def __init__(
        self,
        init="known_poses",
        niter=50,
        schedule="cosine",
        lr=0.01,
        weights_path="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.init = init
        self.niter = niter
        self.schedule = schedule
        self.lr = lr
        self.device = device

        # Load the model
        self.model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)

    def reconstruct_scene(
        self,
        imgs: list[dict],
        known_poses: torch.Tensor,
        known_intrinsics: torch.Tensor,
        scene_graph="complete",
        min_conf_thr=3,
        verbose=False,
    ):
        # Load images
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]["idx"] = 1

        # Make pairs of images
        # If we use sliding window pairs
        # winsize = 3
        # scene_graph = "swin" + "-" + str(winsize)
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        print(f"Number of pairs: {len(pairs)}")
        with timeit("Inference"):
            output = inference(pairs, self.model, self.device, batch_size=1, verbose=False)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            optimize_pp=True,  # will be set to False after preset_intrinsics (just a technicality)
            verbose=verbose,
            min_conf_thr=min_conf_thr,
        )

        known_poses = _process_poses(known_poses)
        known_intrinsics = _process_intrinsics(known_intrinsics, len(known_poses))
        assert known_poses[0].shape == (4, 4)

        scene.preset_pose(known_poses)
        _preset_intrinsics(scene, known_intrinsics)
        assert scene.im_poses.requires_grad is False

        # Run global alignment
        with timeit("Global alignment"):
            scene.compute_global_alignment(init=self.init, niter=self.niter, schedule=self.schedule, lr=self.lr)

        return scene


    def load_images_tensor(self, tensor, size: int = 512, square_ok=False, verbose=False, is_float=True):
        import torchvision.transforms as tvf
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        """Convert a tensor of float images (B, H, W, 3) to the proper input format for DUSt3R"""
        num_images, H, W, _ = tensor.shape
        if verbose:
            print(f">> Loading a tensor of {num_images} images")

        imgs = []
        for idx in range(num_images):
            img_array = tensor[idx] * 255 if is_float else tensor[idx]
            img_array = img_array.numpy().astype(np.uint8)
            img = Image.fromarray(img_array)
            W1, H1 = img.size

            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)

            W, H = img.size
            cx, cy = W // 2, H // 2

            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not square_ok and W == H:
                    halfh = 3 * halfw / 4
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

            W2, H2 = img.size
            if verbose:
                print(f" - processed image {idx} with resolution {W1}x{H1} --> {W2}x{H2}")

            # H, W, C :uint8
            imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx)))

        assert imgs, "No images found in tensor"
        if verbose:
            print(f" (Processed {len(imgs)} images)")
        return imgs
    
    @torch.no_grad()
    def get_points(self, scene, stride=3, return_rgb=True, max_points=None):
        scene = scene.clean_pointcloud()
        pts3d = torch.stack(scene.get_pts3d())[:, ::stride, ::stride, :] # [N, H, W, 3]
        mask = torch.stack(scene.get_masks())[:, ::stride, ::stride] # [N, H, W, 1]
        pts3d = pts3d[mask]
        n_pts = pts3d.shape[0]
        rgb = torch.empty(0)
        if return_rgb:
            rgb = np.stack(scene.imgs)[:, ::stride, ::stride, :] # [N, H, W, 3]
            rgb = torch.from_numpy(rgb).to(pts3d.device) # [N, H, W, 3]
            rgb = rgb[mask] * 255
        if max_points is not None and n_pts > max_points:
            idx = torch.randperm(n_pts)[:max_points]
            pts3d = pts3d[idx]
            if return_rgb:
                rgb = rgb[idx]
        return pts3d, rgb
    
    def get_mvs_data(self, dataset, id_list):
        long_side = max(dataset.image_height, dataset.image_width)
        mvs_imgs = self.load_images_tensor(dataset.image_tensor[id_list])
        img_scale = 512 / long_side
        poses = dataset.cameras.camera_to_worlds[id_list] # [N, 3, 4]
        poses = torch.cat([poses, poses.new_tensor([[[0, 0, 0, 1]]]).expand(poses.shape[0], -1, -1)], dim=1)
        blender2opencv = poses.new_tensor([1, -1, -1, 1]).reshape(1,1,4)
        poses = poses * blender2opencv # nrc,...c->nrc
        intrinsics = dataset.cameras[0].get_intrinsics_matrices() # [3,3]
        intrinsics[:2] = intrinsics[:2] * img_scale
        return mvs_imgs, poses, intrinsics