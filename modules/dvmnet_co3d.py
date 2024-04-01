import argparse, logging, copy, sys
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lightning.pytorch as pl
import cv2
from einops import rearrange
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations
from data_loader_co3d import Co3dDataset as Dataset_Loader
from modules.dvmnet_modules import DVMNet, MaskedMSE
from utils import *

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

class Estimator(pl.LightningModule):
    def __init__(self, cfg, img_size=224, num_rota=50000, freeze=True):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.num_rota = num_rota
        self.freeze = freeze

        self.model = DVMNet(transport="cosine", mask=cfg["NETWORK"]["MASK"])
        self.criterion = MaskedMSE(norm_pix_loss=True, masked=True, patch_size=16)

    def training_step(self, batch, batch_idx):
        img_src = batch["image"][:, 0]
        img_tgt = batch["image"][:, 1]
        mask_src = batch["mask"][:, 0]
        mask_tgt = batch["mask"][:, 1]

        R_src = batch["R_cv"][:, 0]
        R_tgt = batch["R_cv"][:, 1]

        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))
        bs = img_src.shape[0]

        pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, _ = self.model(img_src, img_tgt)

        gt_rota_6d = matrix_to_rotation_6d(gt_delta_R)
        pred_rota_6d = matrix_to_rotation_6d(pred_delta_R)

        loss_rota = torch.norm(pred_rota_6d - gt_rota_6d.detach(), p=1, dim=-1).mean()

        loss_img = 0.5 * (self.criterion(pred_src, img_src * (mask_src > 0), mask_src) + self.criterion(pred_tgt, img_tgt * (mask_tgt > 0), mask_tgt))
        loss_img = loss_img.mean()

        if self.cfg["NETWORK"]["LOSS"] == "rota":
            loss = loss_rota
        elif self.cfg["NETWORK"]["LOSS"] == "both":
            loss = loss_img + loss_rota
        else:
            raise RuntimeError("Unsupported loss function")

        current_lr = self.optimizers().param_groups[0]['lr']

        self.log_dict({'learning_rate': current_lr, "train_loss": loss.item()}, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg["TRAIN"]["STEP_SIZE"], gamma=0.1)

        return [optimizer], [scheduler]

def training(cfg, trainer):
    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"], num_rota=cfg["DATA"]["NUM_ROTA"], freeze=cfg["TRAIN"]["FREEZE"])

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["DATA"]["OBJ_SIZE"]),
            transforms.Normalize(
                cfg['DATA']['PIXEL_MEAN'],
                cfg['DATA']['PIXEL_STD']),
        ]
    )

    train_dataset = Dataset_Loader(
        cfg=cfg,
        category=["all"],
        split="train",
        random_aug=True,
        eval_time=False,
        num_images=2,
        normalize_cameras=False,
        transform=trans,
        img_size=cfg["DATA"]["OBJ_SIZE"]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    ckpt_path = os.path.join("models", cfg["RUN_NAME"], cfg["DATA"]["FILENAME"]+'.ckpt')

    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)
