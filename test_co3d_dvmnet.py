"""
Script for pairwise evaluation of predictor (ie, given 2 images, compute accuracy of
highest scoring mode).

Note that here, num_frames refers to the number of images sampled from the sequence.
The input frames will be all NP2 permutations of using those image frames for pairwise
evaluation.
"""

import argparse
import sys
import os
import numpy as np
import cv2
import torch
from torchvision import transforms
import yaml
from tqdm.auto import tqdm
from pytorch3d.transforms import random_rotations
from modules.dvmnet_co3d import Estimator
from data_loader_co3d import Co3dDataset
from data_loader_co3d import TEST_CATEGORIES
from utils import patchify, unpatchify

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--use_pbar", action="store_true")
    return parser

def get_n_features(model, num_frames, images, crop_params=None):
    features = model.feature_extractor(images)
    return features.reshape((1, num_frames, model.full_feature_dim, 1, 1))

def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi

def get_permutations(num_frames):
    permutations = []
    for i in range(num_frames):
        for j in range(num_frames):
            if i != j:
                permutations.append((i, j))
    return torch.tensor(permutations)


def get_dataset(cfg=None, category="banana", split="train", dataset="co3d"):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["DATA"]["OBJ_SIZE"]),
            transforms.Normalize(
                cfg['DATA']['PIXEL_MEAN'],
                cfg['DATA']['PIXEL_STD']),
        ]
    )

    if dataset == "co3dv1":
        return Co3dv1Dataset(
            cfg=cfg,
            split=split,
            transform=trans,
            category=[category],
            random_aug=False,
            eval_time=True,
        )
    elif dataset in ["co3d", "co3dv2"]:
        return Co3dDataset(
            cfg=cfg,
            split=split,
            transform=trans,
            category=[category],
            random_aug=False,
            eval_time=True,
            img_size=cfg["DATA"]["OBJ_SIZE"]
        )
    else:
        raise Exception(f"Unknown dataset {dataset}")

def visualization(cfg, img, mask, pred):
    image_norm_mean = np.array(cfg["DATA"]["PIXEL_MEAN"])
    image_norm_std = np.array(cfg["DATA"]["PIXEL_STD"])

    masked_img = img * (mask > 0)

    patchified = patchify(masked_img, patch_size=16)

    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)

    pred = unpatchify(pred, patch_size=16, channels=4)

    pred_img = pred[:, :3]
    pred_mask = pred[:, 3:]

    pred_img = patchify(pred_img, patch_size=16)
    pred_img = unpatchify(pred_img * (var + 1.e-6)**.5 + mean, patch_size=16, channels=3)

    img = img[0].permute(1, 2, 0).cpu().detach().numpy() * image_norm_std + image_norm_mean
    img = np.ascontiguousarray(255*img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred_mask = pred_mask.expand(-1, 3, -1, -1)
    pred_mask = pred_mask[0].permute(1, 2, 0).cpu().detach().numpy() > 0
    pred_mask = np.ascontiguousarray(255*pred_mask, dtype=np.uint8)
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)

    pred_img = pred_img[0].permute(1, 2, 0).cpu().detach().numpy() * image_norm_std + image_norm_mean
    pred_img = pred_img * (pred_mask > 0)
    pred_img = np.ascontiguousarray(255*pred_img, dtype=np.uint8)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    return img, pred_img, pred_mask

def evaluate_category(
    cfg,
    model,
    category="banana",
    split="train",
    num_frames=2,
    use_pbar=False,
    dataset="co3d",
):
    dataset = get_dataset(cfg=cfg, category=category, split=split, dataset=dataset)
    device = next(model.parameters()).device

    permutations = get_permutations(num_frames)
    angular_errors = []
    iterable = tqdm(dataset) if use_pbar else dataset
    for idx, metadata in enumerate(iterable):
        n = metadata["n"]
        sequence_name = metadata["model_id"]
        key_frames = np.random.choice(n, num_frames, replace=False)
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"]
        masks = batch["mask"]

        images_permuted = images[permutations]
        images1 = images_permuted[:, 0].to(device)
        images2 = images_permuted[:, 1].to(device)
        masks_permuted = masks[permutations]
        masks1 = masks_permuted[:, 0].to(device)
        masks2 = masks_permuted[:, 1].to(device)

        rotations = batch["R_cv"]
        rotations_permuted = rotations[permutations].to(device)
        rotations_gt = torch.bmm(rotations_permuted[:, 1], torch.inverse(rotations_permuted[:, 0]))

        for i in range(len(permutations)):
            image1 = images1[i][None]
            image2 = images2[i][None]
            mask1 = masks1[i][None]
            mask2 = masks2[i][None]
            gt_delta_R = rotations_gt[i][None]

            with torch.no_grad():
                pred_src, pred_tgt, weight_src, weight_tgt, pred_delta_R, pred_delta_T = model(image1, image2)

            ### geo_dis
            sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
            err = (torch.arccos(sim) * 180. / np.pi).mean().item()

            angular_errors.append(err)

            iterable.set_description("Error: %.2f" % (err))

    return np.array(angular_errors)


def evaluate_pairwise(
    cfg=None,
    model=None,
    checkpoint_path=None,
    split="train",
    num_frames=2,
    print_results=True,
    use_pbar=False,
    categories=TEST_CATEGORIES,
    dataset="co3d",
):
    if model is None:
        print(checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
        model = model.model
        model.eval()

    errors = {}
    errors_15 = {}
    errors_30 = {}
    errors_hist = []
    for category in categories:
        angular_errors = evaluate_category(
            cfg=cfg,
            model=model,
            category=category,
            split=split,
            num_frames=num_frames,
            use_pbar=use_pbar,
            dataset=dataset,
        )
        errors[category] = np.mean(angular_errors)
        errors_15[category] = 100*np.mean(angular_errors < 15)
        errors_30[category] = 100*np.mean(angular_errors < 30)

        print(category + " err: %.2f || acc_30: %.2f || acc_15: %.2f " % (errors[category], errors_30[category], errors_15[category]))

        errors_hist += angular_errors.tolist()

    errors["mean"] = np.mean(list(errors.values()))
    errors_15["mean"] = np.mean(list(errors_15.values()))
    errors_30["mean"] = np.mean(list(errors_30.values()))
    if print_results:
        print(f"{'Category':>10s}{'<15':6s}{'<30':6s}")
        for category in errors_15.keys():
            print(
                f"{category:>10s}{errors_15[category]:6.02f}{errors_30[category]:6.02f}"
            )

    print("avg_err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (errors["mean"], errors_30["mean"], errors_15["mean"]))

    return errors, errors_30, errors_15, errors_hist


if __name__ == "__main__":
    args = get_parser().parse_args()

    args.num_frames = 2
    args.use_pbar = True

    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    args.checkpoint = "./models/checkpoint_co3d.ckpt"
    cfg["NETWORK"]["MASK"] = "both"
    cfg["NETWORK"]["LOSS"] = "both"

    cfg["DATA"]["OBJ_SIZE"] = 224
    cfg["TRAIN"]["FREEZE"] = False
    cfg["TRAIN"]["PRETRAIN"] = True
    iter_num = 5

    errors, errors_30, errors_15 = {}, {}, {}
    errors_hist = []
    for i in range(iter_num):
        error, error_30, error_15, error_hist = evaluate_pairwise(cfg=cfg,
            checkpoint_path=args.checkpoint,
            num_frames=args.num_frames,
            print_results=True,
            use_pbar=args.use_pbar,
            split="test",
        )
        if i == 0:
            for category in error.keys():
                errors[category] = []
                errors_30[category] = []
                errors_15[category] = []

        for category in error.keys():
            errors[category].append(error[category])
            errors_30[category].append(error_30[category])
            errors_15[category].append(error_15[category])

        errors_hist += error_hist

    for category in errors.keys():
        errors[category] = np.asarray(errors[category]).mean()
        errors_30[category] = np.asarray(errors_30[category]).mean()
        errors_15[category] = np.asarray(errors_15[category]).mean()

    print(f"{'Category':>10s}{'<15':6s}{'<30':6s}")
    with open('./co3d_result.txt', 'a') as f:
        for category in errors_15.keys():
            print(f"{category:>10s}{errors[category]:6.02f}{errors_15[category]:6.02f}{errors_30[category]:6.02f}")
            f.write(f"{category:>10s}{errors[category]:6.02f}{errors_15[category]:6.02f}{errors_30[category]:6.02f} \n")
    f.close()
