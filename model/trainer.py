from typing import Tuple
from CONFIG import CONFIG
import clip
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from argparse import ArgumentParser
from CONFIG import CONFIG
from dataset import prepare_data 
from aliproduct_model import ALIPRODUCT_CLIP
from pytorch_lightning.loggers import WandbLogger
import wandb 
import time
import warnings
# warnings.filterwarnings("once")
import shutup 
shutup.please()


def main():
    seed_everything(CONFIG.global_random_state, workers=True)
    clip_model,preprocess = clip.load(CONFIG.model_name,device=CONFIG.device,jit=False)
    print("loading clip model..")
    print("initalising finetuning clip model..")
    model = ALIPRODUCT_CLIP(clip_model)
    # clip.model.convert_weights(model)
    model.float()
    print("preparing data..")
    train_loader,val_loader = prepare_data(CONFIG.df_path,CONFIG.image_data_dir,CONFIG.image_data_folder
    ,CONFIG.image_col,CONFIG.label_col,CONFIG.batch_size,preprocess,CONFIG.global_random_state,use_all=True)
    checkpoint_callback = ModelCheckpoint(monitor="train_clip_loss",
    dirpath=".",
    filename="aliproduct_2022_cvpr"+CONFIG.model_name+"_{train_clip_loss:.5f}",
    save_top_k=CONFIG.epoch,
    mode="min",
    save_last=False)
 
    wandb_logger = WandbLogger(project='aliproduct_2022_cvpr', job_type='Train',
        reinit=dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)
    wandb_logger.watch(model,log="all", log_freq=100)          
    trainer = Trainer(accelerator="gpu",devices=4,strategy="deepspeed_stage_3",max_epochs=CONFIG.epoch,logger=wandb_logger,callbacks=[checkpoint_callback],precision=16,deterministic=True)

    trainer.fit(model,train_loader)

if __name__ == "__main__":
    wandb.login(key=CONFIG.wandb_api_key)
    main()
    wandb.finish()
