import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from CONFIG import CONFIG 
from BLIP.models.blip_itm import blip_itm



class ALIPRODUCT_BLIP(pl.LightningModule):
    def __init__(self,pretrained, image_size, vit='base'):
        super().__init__()
        self.med_config_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/configs/med_config.json"
        self.model = blip_itm(med_config= self.med_config_path,pretrained=pretrained, image_size=image_size, vit=vit)
    def forward(self,img,label):
        image_features,text_features = self.model(img,label,match_head='itc')
        return image_features,text_features 

    def predict_step(self,batch,batch_idx):
        with torch.no_grad():
            self.model.eval()
            image,text = batch
            image_features,text_features = self(image,text)
            image_features = image_features.detach().cpu()
            text_features = text_features.detach().cpu()
            return image_features,text_features
