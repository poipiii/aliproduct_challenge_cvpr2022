import clip
from cv2 import transform
from sqlalchemy import true
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
import cv2
from open_clip import tokenizer

class ALIPRODUCT_DATASET():
    def __init__(self,images,texts,dir,folder,preprocess,tokenize=True,test=False):
        self.images = images
        self.texts = texts
        self.dir = dir
        self.folder = folder
        self.preprocess = preprocess
        self.test = test
        self.tokenize = tokenize
    def __len__(self):
        return(len(self.images))
    def __getitem__(self,idx):
        image_name = self.images[idx]
        text = self.texts[idx]
        if self.test:
            image = Image.open(f"{self.dir}{self.folder}/{image_name}").convert("RGB")
        else:
            image = Image.open(image_name).convert("RGB")
        image = self.preprocess(image)
        if self.tokenize:
             text_caption = tokenizer.tokenize(text)
        else:
            text_caption = text
        return image,text_caption
   


def prepare_data(df_path,image_data_dir,image_data_folder,image_col,label_col,batch_size,preprocess,random_state,split_size=0.2,test=False,use_all=False,tokenize=True):
    df = pd.read_csv(df_path)
    images = df[image_col].values.tolist()
    texts = df[label_col].values.tolist()
    if test:
        test_data = ALIPRODUCT_DATASET(images,texts,image_data_dir,image_data_folder,preprocess,test=test,tokenize=tokenize)
        test_dataloader = DataLoader(test_data,batch_size,num_workers=12,pin_memory=True,shuffle=False)
        return test_dataloader,df

    else:
        if use_all:
            train_data =  ALIPRODUCT_DATASET(images,texts,image_data_dir,image_data_folder,preprocess,test=test,tokenize=tokenize)
            train_dataloader = DataLoader(train_data,batch_size,shuffle=False,num_workers=12,pin_memory=True)
            return train_dataloader,df

        else:
            train_image,val_image ,train_text,val_text = train_test_split(images,texts,random_state= random_state,test_size=split_size)
            train_data =  ALIPRODUCT_DATASET(train_image,train_text,image_data_dir,image_data_folder,preprocess,test=test,tokenize=tokenize)
            val_data = ALIPRODUCT_DATASET(val_image,val_text,image_data_dir,image_data_folder,preprocess,test=test,tokenize=tokenize)
            train_dataloader = DataLoader(train_data,batch_size,shuffle=False,num_workers=12,pin_memory=True)
            val_dataloader = DataLoader(val_data,batch_size,shuffle=False,num_workers=12)


            return torch.tensor(train_dataloader,val_dataloader)

        



