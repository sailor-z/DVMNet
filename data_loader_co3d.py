"""
CO3D (v2) dataset.
"""
import sys
import gzip
import json
import os.path as osp

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
from normalize_cameras import first_camera_transform, normalize_cameras
from utils import square_bbox, get_permutations

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Co3dDataset(Dataset):
    def __init__(
        self,
        cfg=None,
        category=("all",),
        split="train",
        transform=None,
        random_aug=True,
        jitter_scale=(1.1, 1.2),
        jitter_trans=(-0.07, 0.07),
        num_images=2,
        img_size=224,
        normalize_cameras=False,
        mask_images=False,
        first_camera_transform=False,
        first_camera_rotation_only=False,
        eval_time=False,
    ):
        """
        Args:
            category (list): List of categories to use.
            split (str): "train" or "test".
            transform (callable): Transformation to apply to the image.
            random_aug (bool): Whether to apply random augmentation.
            jitter_scale (tuple): Scale jitter range.
            jitter_trans (tuple): Translation jitter range.
            num_images: Number of images in each batch.
        """
        self.cfg = cfg
        self.normalize_cameras = normalize_cameras
        if "all" in category:
            category = TRAINING_CATEGORIES
        category = sorted(category)

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        for c in category:
            annotation_file = osp.join(self.cfg["CO3D"]["CO3D_ANNOTATION_DIR"], f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

            print(annotation_file)
            print(counter)

        self.sequence_list = list(self.rotations.keys())
        self.split = split

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size),
                    transforms.Normalize(
                        cfg['DATA']['PIXEL_MEAN'],
                        cfg['DATA']['PIXEL_STD']),
                ]
            )
        else:
            self.transform = transform

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size),
            ]
        )

        if random_aug:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1.15, 1.15]
            self.jitter_trans = [0, 0]

        self.num_images = num_images
        self.image_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.first_camera_rotation_only = first_camera_rotation_only
        self.mask_images = mask_images

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))

        s = (self.jitter_scale[1] - self.jitter_scale[0]) * torch.rand(1).item() + self.jitter_scale[0]
        tx = (self.jitter_trans[1] - self.jitter_trans[0]) * torch.rand(1).item() + self.jitter_trans[0]
        ty = (self.jitter_trans[1] - self.jitter_trans[0]) * torch.rand(1).item() + self.jitter_trans[0]

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self, index):
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        # ids = np.random.choice(len(metadata), self.num_images, replace=False)
        ids = torch.randperm(len(metadata))[:self.num_images]
        return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        if no_images:
            annos = [metadata[i] for i in ids]
            rotations = [torch.tensor(anno["R"]) for anno in annos]
            translations = [torch.tensor(anno["T"]) for anno in annos]
            batch = {}
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            return batch

        annos = [metadata[i] for i in ids]
        images = []
        masks = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        for anno in annos:
            filepath = anno["filepath"]

            image = Image.open(osp.join(self.cfg["CO3D"]["CO3D_DIR"], filepath)).convert("RGB")

            mask_name = osp.basename(filepath.replace(".jpg", ".png"))

            mask_path = osp.join(
                self.cfg["CO3D"]["CO3D_DIR"], category, sequence_name, "masks", mask_name
            )
            mask = Image.open(mask_path).convert("L")

            if mask.size != image.size:
                mask = mask.resize(image.size)
            mask = Image.fromarray(np.array(mask) > 125)

            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                image = Image.composite(image, white_image, mask)

            images.append(image)
            masks.append(mask)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))

        images_transformed, masks_transformed, crop_parameters, imgs_size, corner_parameters = [], [], [], [], []
        for i, (anno, image) in enumerate(zip(annos, images)):
            imgs_size.append(torch.tensor([image.height, image.width]).float())

            if self.cfg["DATA"]["OBJ_SIZE"] is None:
                images_transformed.append(self.transform(image))
            else:
                w, h = image.width, image.height
                bbox = np.array(anno["bbox"])
                bbox_jitter = self._jitter_bbox(bbox)

                image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)
                images_transformed.append(self.transform(image))

                mask = self._crop_image(masks[i], bbox_jitter, white_bg=False)
                masks_transformed.append(self.transform_mask(mask))

                crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
                cc = (2 * crop_center / min(h, w)) - 1
                crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

                crop_parameters.append(
                    torch.tensor([-cc[0], -cc[1], crop_width]).float()
                )
                ratio = float(bbox_jitter[2] - bbox_jitter[0]) / float(self.cfg["DATA"]["OBJ_SIZE"])
                corner_parameters.append(torch.tensor([bbox_jitter[0], bbox_jitter[1], ratio]).float())

        images = images_transformed
        masks = masks_transformed

        cameras = PerspectiveCameras(
            focal_length=[data["focal_length"] for data in annos],
            principal_point=[data["principal_point"] for data in annos],
            R=[data["R"] for data in annos],
            T=[data["T"] for data in annos],
        )
        R_cv, T_cv, K_cv = opencv_from_cameras_projection(cameras, torch.stack(imgs_size)*torch.ones([len(annos), 2]))

        batch = {
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": ids,
            "R_cv": R_cv,
            "T_cv": T_cv,
            "K_cv": K_cv,
        }

        if self.normalize_cameras:
            normalized_cameras, _, _, _, _ = normalize_cameras(cameras)

            if self.first_camera_transform or self.first_camera_rotation_only:
                normalized_cameras = first_camera_transform(
                    normalized_cameras,
                    rotation_only=self.first_camera_rotation_only,
                )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                assert False

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T

            batch["R_original"] = torch.stack(
                [torch.tensor(anno["R"]) for anno in annos]
            )
            batch["T_original"] = torch.stack(
                [torch.tensor(anno["T"]) for anno in annos]
            )

            if torch.any(torch.isnan(batch["T"])):
                print(ids)
                print(category)
                print(sequence_name)
                assert False

        else:
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)

        if len(crop_parameters) > 0:
            batch["crop_params"] = torch.stack(crop_parameters)
        if len(corner_parameters) > 0:
            batch["corner_params"] = torch.stack(corner_parameters)

        # Add relative rotations
        permutations = get_permutations(len(ids), eval_time=self.eval_time)
        n_p = len(permutations)
        relative_rotation = torch.zeros((n_p, 3, 3))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] = rotations[i].T @ rotations[j]
        batch["relative_rotation"] = relative_rotation

        # Add images
        if self.transform is None:
            batch["image"] = images
        else:
            batch["image"] = torch.stack(images)

        batch["mask"] = torch.stack(masks)

        return batch
