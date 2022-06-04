import PIL
import torch
from torch.utils.data import Dataset,DataLoader
from BLIP.models.blip import blip_decoder
import pandas as pd
from PIL import Image
import json
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pickle as pkl
class captioning_dataset():
    def __init__ (self,image_path:str,mapping_path:str,mapping_name:str,image_key_name:str,image_transform):
        self.image_path = image_path
        self.mapping_path = mapping_path
        self.mapping_name = mapping_name
        self.image_key_name = image_key_name
        self.data = self.load_mappings(self.mapping_path,self.mapping_name,self.image_key_name)
        self.image_transform = image_transform
    def load_mappings(self,mapping_path:str,mapping_name:str,image_key_name:str):
        file_extension = mapping_name.split(".")[-1]
        if file_extension == "csv":
            df =pd.read_csv(mapping_path+mapping_name)
            data = df[image_key_name].values
            return data
        elif file_extension == "json":
            with open(mapping_path+mapping_name,"r") as file:
                data_json = json.load(file)
            data = [i[image_key_name] for  i in data_json]
            return data 
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        image = self.data[idx]
        image = Image.open(f"{self.image_path}/{image}").convert("RGB")
        image = self.image_transform(image)
        return image 


class caption_blip(pl.LightningModule):
    def __init__(self,pretrained,image_size=384, vit='base'):
        super().__init__()
        self.model = blip_decoder(pretrained,image_size=image_size,vit=vit)
    def forward(self,x):
        pred = self.model.generate(x,sample=True, top_p=0.9, max_length=20, min_length=5)
        return pred
    def predict_step(self,batch,batch_idx):
        with torch.no_grad():
            self.model.eval()
            image = batch
            captions = self(image)
            # captions = captions.detach().cpu()

            return captions
def save_pred(pred,process_rank):
    print("saving features")
    filename = f"/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/output/caption_finetune/results/blip_gen_caption_{str(process_rank)}.pkl"
    file = open(filename,'wb')
    pkl.dump(pred,file)
    file.close()

def generate_captions():
    pl.seed_everything(101)
    preprocess = transforms.Compose([
        transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    caption_data = captioning_dataset("/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/","/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/"
    ,"aliproduct2_val_ann.json","image",preprocess)
    caption_data_loader = DataLoader(caption_data,batch_size=100,shuffle=False,pin_memory=True,num_workers=12)
    model = caption_blip("https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth")
    trainer = pl.Trainer(gpus=4,strategy="deepspeed")
    preds = trainer.predict(model,caption_data_loader)
    save_pred(preds,trainer.global_rank)
    return preds
preds = generate_captions()



