import os
import sys
sys.path.insert(0, '..')
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/model")
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer,LightningModule
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import gc
import os
import pickle
from aliproduct_blip_model import ALIPRODUCT_BLIP
from dataset import ALIPRODUCT_DATASET,prepare_data
from CONFIG import CONFIG
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning.callbacks import BasePredictionWriter



image_size = 384
preprocess = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str,file_prefix:str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module: "LightningModule", predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, f"{self.file_prefix}_{trainer.global_rank}.pt"))

def test_feature(train_csv,caption_col,clip_model,head,num_of_gpu,file_prefix_name,file_output):
    pl.seed_everything(CONFIG.global_random_state)
    test_loader,df = prepare_data(train_csv,
    CONFIG.test_image_data_dir,CONFIG.test_image_data_folder
    ,CONFIG.image_col,caption_col,256,preprocess,CONFIG.global_random_state,test=False,use_all=True,tokenize=False)
    print("shape of data to extract:",df.shape)
    if num_of_gpu > 1:
        trainer = Trainer(gpus=4,strategy="deepspeed",callbacks=[PredictionWriter(file_output,file_prefix_name)])
    else:
        trainer = Trainer(gpus=1,callbacks=[PredictionWriter(file_output,file_prefix_name)])
    pred = trainer.predict(clip_model,test_loader)
    if head == "itc":
        full_pred = tuple(map(torch.concat, zip(*pred)))
        image_embed,text_embed = full_pred    
        return image_embed,text_embed
    else:
        return pred
    





if __name__ == "__main__":

    #filepath and file name to train csv file 
    train_csv = "/home/user/Desktop/AliProduct2022/train_data_v5.csv"

    #caption column to run itm score on found in train csv  
    col_to_test =  "caption"

    #checkpoint of blip model you want to use 
    checkpoint = "/home/user/Desktop/large_v5/save_checkpoint_4.pth"

    #number of gpu to use for generating itm score recommended 4 gpus 
    num_of_gpu = 4

    #prefix name of each file that stores itm scores, example of output prefix_gpu_rank.pt 
    file_prefix_name = "blip_itm_train_v5"

    #folder where itm scores will be stored 
    file_output = "./itm_score_predictions"

    clip_model =ALIPRODUCT_BLIP(checkpoint,image_size,vit="large",head="itm")
    pred = test_feature(train_csv,col_to_test,clip_model,"itm",num_of_gpu,file_prefix_name,file_output)




