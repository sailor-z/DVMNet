import sys
import yaml
import os
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

from modules.dvmnet_co3d import training

def main(cfg):
    cfg["RUN_NAME"] = 'Co3d_dvmnet'
    cfg["TRAIN"]["LR"] = 1e-5
    cfg["TRAIN"]["BS"] = 64
    cfg["DATA"]["OBJ_SIZE"] = 224
    cfg["TRAIN"]["MAX_EPOCH"] = 300
    cfg["TRAIN"]["STEP_SIZE"] = 200


    cfg["TRAIN"]["FREEZE"] = False
    cfg["TRAIN"]["PRETRAIN"] = True
    cfg["NETWORK"]["MASK"] = "both"
    cfg["NETWORK"]["LOSS"] = "both"
    cfg["DATA"]["FILENAME"] = "checkpoint_co3d"
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', dirpath=os.path.join("./models", cfg["RUN_NAME"]), \
        filename=cfg["DATA"]["FILENAME"])

    ### multiple GPUs
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="ddp_find_unused_parameters_true", accumulate_grad_batches=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"], sync_batchnorm=True, callbacks=[checkpoint_callback])

    training(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
