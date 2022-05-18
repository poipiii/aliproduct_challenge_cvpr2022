import os
import sys
sys.path.insert(0, '..')
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/model")
import clip
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
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
import faiss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

clean_df = pd.read_csv("/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/eda/train_data_v1.csv")





image_size = 384
preprocess = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

def save_pred(pred):
    filename = "blip_train_features.pkl"
    file = open(filename,'wb')
    pickle.dump(pred,file)
    file.close()

def test_feature(caption_col,clip_model):
    test_loader,df = prepare_data("/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/eda/train_data_v1.csv",
    CONFIG.test_image_data_dir,CONFIG.test_image_data_folder
    ,CONFIG.image_col,caption_col,CONFIG.test_batch_size,preprocess,CONFIG.global_random_state,test=False,use_all=True,tokenize=False)
    trainer = Trainer(gpus=4,strategy="deepspeed")
    pred = trainer.predict(clip_model,test_loader)
    save_pred(pred)
    full_pred = tuple(map(torch.concat, zip(*pred)))
    image_embed,text_embed = full_pred    
    return image_embed,text_embed


col_to_test =  "caption"
checkpoint = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/output/save_checkpoint_6_wed.pth"
clip_model =ALIPRODUCT_BLIP(checkpoint,image_size)
image_embed,text_embed = test_feature(col_to_test,clip_model)



single_pair_cosine = np.array(list((txt @ img.T).item() for txt ,img in tqdm(zip(text_embed,image_embed))))

clean_df["single_pair_cosine"] = single_pair_cosine

clean_df.to_csv("../data/train_data_v2.csv")


