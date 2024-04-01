from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import os
from fastprogress import progress_bar
import lightning.pytorch as pl
from modules.dvmnet_objaverse_final import Estimator
from data_loader import Dataset_Loader_Objaverse_stereo_test as Dataset_Loader
from utils import to_cuda

def main(cfg):
    cfg["RUN_NAME"] = 'Objaverse_DVMNet'
    cfg["DATA"]["BG"] = True

    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')

    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
        model.eval()
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    test_dataset = Dataset_Loader(cfg, None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)

    trainer = pl.Trainer()
    model.step_outputs.clear()
    trainer.test(model, dataloaders=test_dataloader)

    pred_err = torch.cat(model.step_outputs)

    pred_acc_30 = 100 * (pred_err < 30).float().mean().item()
    pred_acc_15 = 100 * (pred_err < 15).float().mean().item()
    pred_err = pred_err.mean().item()

    pred_Rs = np.asarray(model.pred_Rs)
    np.savetxt("./objaverse_pred_Rs.txt", pred_Rs)

    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (pred_err, pred_acc_30, pred_acc_15))

    with open("./objaverse_result.txt", 'a') as f:
        f.write("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % (pred_err, pred_acc_30, pred_acc_15))
    f.close()

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
