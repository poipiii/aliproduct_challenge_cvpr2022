from pyexpat import model
from turtle import forward
import clip
from sqlalchemy import true
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from CONFIG import CONFIG 
from Multilingual_CLIP.src import multilingual_clip
from deepspeed.ops.adam import FusedAdam





def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()
        if p.grad !=None:
            p.grad.data = p.grad.data.float() 



class text_encoder(pl.LightningModule):
    def __init__(self,text_model):
        super().__init__()
        self.model = text_model
    def forward(self,text):
        return self.model(text)

class image_encoder(pl.LightningModule):
    def __init__(self,image_model):
        super().__init__()
        self.model = image_model
        self.logit_scale = self.model.logit_scale.exp()
    def forward(self,image):
        return self.model.encode_image(image),self.logit_scale

class ALIPRODUCT_CLIP_MULTILINGUAL(pl.LightningModule):
    def __init__(self,image_model,text_model):
        super().__init__()
        self.image_encoder = image_encoder(image_model)
        self.text_enocoder  =text_encoder(text_model)
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self,img,label):
        # print(img)
        # print(label)
        # label = torch.squeeze(label,1)
        # img = img.to(CONFIG.device)
        # label = label.to(CONFIG.device)
        
        image_features,logit_scale = self.image_encoder(img)
        text_features = self.text_enocoder(label)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.model.logit_scale.exp()
    


        #singe gpu code
        # image_features,text_features = self.model(img,label)

        return logit_scale* image_features,logit_scale *text_features,logit_scale



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=CONFIG.learning_rate,
            betas=CONFIG.betas,
            eps=CONFIG.epsilion,
            weight_decay=CONFIG.weight_decay
        )
        return optimizer
    
    def training_step(self,batch,batch_idx):
        image,text = batch
        # print(text.size())
        # text = torch.squeeze(text,1)
        # print(text.size())
        
        image_features,text_features,logit_scale = self(image,text)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        ground_truth = torch.arange(len(logits_per_image)).to(CONFIG.device)
        total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
        self.log('train_clip_loss',total_loss,on_step=False, on_epoch=True,prog_bar=True)
        return total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        convert_models_to_fp32(self.model)           
        #self.model.float()
        optimizer.step(closure=optimizer_closure)
        clip.model.convert_weights(self.model)


    def validation_step(self,batch,batch_idx):
        image,text = batch
        image_features,text_features,logit_scale = self(image,text)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        ground_truth = torch.arange(len(logits_per_image)).to(CONFIG.device)
        total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
        self.log('val_clip_loss',total_loss,on_step=False, on_epoch=True,prog_bar=True)
        return total_loss
    def predict_step(self,batch,batch_idx):
        with torch.no_grad():
            image,text = batch
            image_features,text_features,logit_scale  = self(image,text)
            image_features = image_features.detach().cpu()
            text_features = text_features.detach().cpu()
            return image_features,text_features



def load_aliproduct_clip_multilingual():
    txt_model = multilingual_clip.load_model("M-BERT-Base-ViT-B")
    img_model,preprocess  = clip.load("ViT-B/32",CONFIG.device)
    model = ALIPRODUCT_CLIP_MULTILINGUAL(img_model,txt_model)
    return model,preprocess


