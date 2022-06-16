import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from CONFIG import CONFIG 
from BLIP.models.blip_itm import blip_itm



class ALIPRODUCT_BLIP(pl.LightningModule):
    def __init__(self,pretrained, image_size, vit='base',head='itc'):
        super().__init__()
        self.med_config_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/configs/med_config.json"
        self.head = head
        print(self.head)
        self.model = blip_itm(med_config= self.med_config_path,pretrained=pretrained, image_size=image_size, vit=vit)
    def forward(self,img,label):
        if self.head =="itm":
            itm_score = self.model(img,label,match_head=self.head)
            return itm_score

        else:
            image_features,text_features = self.model(img,label,match_head=self.head)
            return image_features,text_features 

    def predict_step(self,batch,batch_idx):
        if self.head == "itm":
            with torch.no_grad():
                self.model.eval()
                image,text = batch
                itm_output = self(image,text)
                itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
                itm_score = itm_score.detach().cpu()
                return itm_score
        else:
            with torch.no_grad():
                self.model.eval()
                image,text = batch
                image_features,text_features = self(image,text)
                image_features = image_features.detach().cpu()
                text_features = text_features.detach().cpu()
                return image_features,text_features
    # def on_predict_epoch_end(self, results):
    #     if self.trainer.is_global_zero:
    #         print("gather all")
    #         all_preds = self.all_gather(results)
    #         print("gather all finish")
    #         print(len(all_preds))
    #         # print(all_preds[0][0].size())
    #         return all_preds
