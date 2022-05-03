import clip
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

class ALIPRODUCT_DATASET():
    def __init__(self,images,texts,dir,folder,preprocess,test=False):
        self.images = images
        self.texts = texts
        self.dir = dir
        self.folder = folder
        self.preprocess = preprocess
        self.test = test
        # self.z_shot_labels,self.z_shot_label_map = self.create_z_shot_labels(self.df["label_name"].unique(),self.df["label"].unique())
    def __len__(self):
        return(len(self.images))
    def __getitem__(self,idx):
        image_name = self.images[idx]
        text = self.texts[idx]
        if self.test:
            image = cv2.imread(image_name)
        else:
            image = cv2.imread(image_name)
        image = Image.fromarray(image).convert("RGB")
        image = self.preprocess(image)
        text_caption = clip.tokenize(f"this is a {text}",truncate=True)
        return image,text_caption
    # def create_z_shot_labels(self,label_names,label):
    #     z_shot_labels = []
    #     z_shot_label_map  = {}
    #     for label_name, label, idx in zip(label_names, label, range(len(label))):
    #         z_shot_label = f"this is a {label_name}"
    #         z_shot_labels.append(z_shot_label)
    #         z_shot_label_map[idx] = {"label_name": label_name, "label": label}
    #     z_shot_labels = clip.tokenize(z_shot_labels)
    #     return z_shot_labels,z_shot_label_map



def prepare_data(df_path,image_data_dir,image_data_folder,image_col,label_col,batch_size,preprocess,random_state,split_size=0.2,test=False,use_all=False):
    df = pd.read_csv(df_path)
    images = df[image_col].values.tolist()
    texts = df[label_col].values.tolist()
    if test:
        test_data = ALIPRODUCT_DATASET(images,texts,image_data_dir,image_data_folder,preprocess,test=test)
        test_dataloader = DataLoader(test_data,batch_size,num_workers=12,pin_memory=True)
        return test_dataloader,df

    else:
        if use_all:
            train_data =  ALIPRODUCT_DATASET(images,texts,image_data_dir,image_data_folder,preprocess,test=test)
            train_dataloader = DataLoader(train_data,batch_size,shuffle=True,num_workers=12,pin_memory=True)
            return train_dataloader,None

        else:
            train_image,val_image ,train_text,val_text = train_test_split(images,texts,random_state= random_state,test_size=split_size)
            train_data =  ALIPRODUCT_DATASET(train_image,train_text,image_data_dir,image_data_folder,preprocess,test=test)
            val_data = ALIPRODUCT_DATASET(val_image,val_text,image_data_dir,image_data_folder,preprocess,test=test)
            train_dataloader = DataLoader(train_data,batch_size,shuffle=True,num_workers=12,pin_memory=True)
            val_dataloader = DataLoader(val_data,batch_size,shuffle=False,num_workers=12)


            return torch.tensor(train_dataloader,val_dataloader)

        



