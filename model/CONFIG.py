import torch
import time
import os
class CONFIG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size  = 160
    test_batch_size = 160
    epoch = 10
    learning_rate = 5e-6
    betas = (0.9,0.98)
    epsilion = 1e-6
    weight_decay = 0.05
    image_data_dir = "/home/ubuntu/Downloads/Auto-retail-syndata-release"
    image_data_folder = "syn_image_train"
    df_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/train_data_v3.csv"
    test_image_data_dir = "../../"
    test_image_data_folder = "val_imgs"
    test_df_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_map.csv"
    image_col = "full_path"
    label_col = "step_4_no_num"
    test_image_col = "product"
    test_label_col = "caption"  
    global_random_state = 101
    model_name = "ViT-B-32-quickgelu"
    group = model_name+'-'+str(int(time.time()))
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    
