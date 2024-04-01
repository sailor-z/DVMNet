import argparse, logging, copy, sys, math
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import lightning.pytorch as pl
import numpy as np
import cv2
from utils import *
from data_loader import Dataset_Loader_Objaverse_stereo as Dataset_Loader
from data_loader import Dataset_Loader_Objaverse_stereo_test as Dataset_Loader_Test
from data_loader import Dataset_Loader_LINEMOD_stereo_train as Dataset_Loader_LM
from modules.dvmnet_modules import DVMNet, MaskedMSE

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)


class Estimator(pl.LightningModule):
    def __init__(self, cfg=None, img_size=224):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size

        self.model = DVMNet(transport="cosine", mask=cfg["NETWORK"]["MASK"])
        self.criterion = MaskedMSE(norm_pix_loss=True, masked=True, patch_size=16)

        self.step_outputs = []
        self.gt_dis = []
        self.pred_Rs = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        ### mask the input image
        if self.cfg["DATA"]["BG"] is False:
            img_src = img_src * mask_src
            img_tgt = img_tgt * mask_tgt

        with torch.no_grad():
            gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))

        pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, pred_delta_T = self.model(img_src, img_tgt)

        gt_rota_6d = matrix_to_rotation_6d(gt_delta_R)
        pred_rota_6d = matrix_to_rotation_6d(pred_delta_R)

        loss_rota = torch.norm(pred_rota_6d - gt_rota_6d.detach(), p=1, dim=-1)
        loss_img = 0.5 * (self.criterion(pred_src, img_src * (mask_src > 0), mask_src) + self.criterion(pred_tgt, img_tgt * (mask_tgt > 0), mask_tgt))

        valid = (mask_src.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"]) * (mask_tgt.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"])
        if "dis_init" in batch.keys():
            dis_init = batch["dis_init"]
            valid = valid * (dis_init < self.cfg["DATA"]["VIEW_THR"]).float()

        loss_rota = loss_rota * valid
        loss_rota = loss_rota.sum() / valid.sum().clamp(min=1e-8)

        loss_img = loss_img * valid
        loss_img = loss_img.sum() / valid.sum().clamp(min=1e-8)

        loss = loss_rota + loss_img

        current_lr = self.optimizers().param_groups[0]['lr']

        self.log_dict({'learning_rate': current_lr, "train_loss": loss.item()}, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))

        pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, pred_delta_T = self.model(img_src, img_tgt)

        ### geo_dis
        sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        pred_acc_15 = (geo_dis <= 15).float().mean()
        pred_acc_30 = (geo_dis <= 30).float().mean()

        self.log("val_geo_dis", geo_dis.float().mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_15", pred_acc_15.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_30", pred_acc_30.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.step_outputs.append(geo_dis)

    def on_validation_epoch_end(self):
        geo_dis = torch.cat(self.step_outputs)

        pred_acc_15 = 100 * (geo_dis <= 15).float().mean()
        pred_acc_30 = 100 * (geo_dis <= 30).float().mean()
        geo_dis = geo_dis.float().mean()

        self.step_outputs.clear()

    def test_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        if torch.any(mask_src.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]) or torch.any(mask_tgt.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]):
            print("Skip bad case")
            return 0

        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))

        pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, pred_delta_T = self.model(img_src, img_tgt)

        ### geo_dis
        sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        self.step_outputs.append(geo_dis)
        self.pred_Rs.append(pred_delta_R.cpu().detach().numpy().reshape(-1))

        self.log("test_error", geo_dis.mean().item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg["TRAIN"]["STEP_SIZE"], gamma=0.1)
        return [optimizer], [scheduler]


def training(cfg, trainer):
    # val_dataset = Dataset_Loader_Test(cfg, None)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)
    #
    # train_dataset = Dataset_Loader(cfg, "train", None)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
    #     num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)
    #
    # model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    # ckpt_path = os.path.join("models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    # if os.path.exists(ckpt_path):
    #     print("Loading the pretrained model from the last checkpoint")
    #     trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    # else:
    #     print("Train from scratch")
    #     trainer.fit(model, train_dataloader, val_dataloader)

    train_dataset = Dataset_Loader(cfg, "train", None)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    ckpt_path = os.path.join("models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)


def training_lm(cfg, trainer):
    CATEGORY = ["APE", "CAN", "EGGBOX", "GLUE", "HOLEPUNCHER", "IRON", "LAMP", "PHONE"]
    clsIDs = [cfg["LINEMOD"][cat] for cat in CATEGORY]

    train_dataset = Dataset_Loader_LM(cfg, clsIDs)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    filename = "checkpoint_lm.ckpt"

    ckpt_path = os.path.join("models", cfg["RUN_NAME"], filename)
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)
