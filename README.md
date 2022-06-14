<h1 align="center">cvpr aliproduct 2022</h1>

  <p align="center">
    Data processing , training and evalaution code used for CVPR 2022 AliProducts Challenge 
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
this project contains the codes used for the CVPR  aliproduct 2 Challenge competition on cross-modal image retrival 
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow the Steps below to setup your enviroment to run the scripts in thie repository 

### Prerequisites

Ensure that you have [anaconda](https://www.anaconda.com/products/distribution) installed on your machine and optionaly it is recommmded that you run these scripts on the machine that has at least 1 GPU with 12gb of vram 

### Installation
```pip install -r requirements.txt ```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
To use this scripts you have to first download the aliproduct 2 dataset as well as the `.json` files  and unzip each downloaded image file into a folder after this is done follow the steps below to prepare the data,
### Data preperation 
1. navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/eda/aliproduct_cvpr_eda_2022.ipynb` and edit the code below with your own file paths and desired file names 
```py
# file path to where your train.json is stored in your machine
train_data_map = "../../train.json"

# file path to where your val.json is stored in your machine
val_data_map = "../../val.json"

# file path to the folder where your train images are stored in your machine
image_folder_dir = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge"

# file path to the folder where your validaion images are stored in your machine
val_data_dir = "../../val_imgs"
```

2. run the notebook to generate `.csv` files containing captions,images and filepaths for each image for train and validation data 
3. **OPTIONAL STEP** if you want to augment the validaion data for testing run `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/eda/aliproduct_val_data_eng.ipynb` and edit the 2nd code cell as shown in the sample code below to specify the input and output csv files and run this notebook to create another csv file containing the original and modified/augmented captions
```py
#file path to where you store your validation csv file in your machine if you do not have this file run aliproduct_cvpr_eda_2022.ipynb to generate the base csv file 
val_data_csv = "../data/val_data_map.csv"
#output file path and name for modified validation data 
output_modified_val_data_csv = "../data/val_data_prompt_clean.csv"
```
#### Cleaning data using Cosine/ITM score 
1. to generate the Cosine/ITM score for your train data first navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/test_blip_filter_multi.py` next configure the variables shown in the sample code below to specify the type of scoring to use as well as other options 
```py

    #which score type to use , if Cosine use "itc" head if itm use "itm" head
    blip_score_head = "itm"
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

```
2. once you have configured the variables to your own specifications , open your terminal and type in or copy the following command `cd /home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model` to change your working directory to the `code` folder 
3. lastly run the script by typing or copying the following command `python test_blip_filter_multi.py` the generated result will be split into number of files = `num_of_gpu` specified in the variables configuration step above 
4. to combine all files and to explore distribution of scores for **Cosine** navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/check_pair_cosine.ipynb` and edit the variables containing the file paths to each result generated in the previous step , please ensure you combine the files in order of the rank number contain within generated the file name. to do the same for **itm** score navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/itm_distribution.ipynb` and repeat the same steps and lastly run the notebook to generate a pickle file that contains all of the scores combined into 1 file 
5. lastly to filter the data at a certain score threshold navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/test_threshold.ipynb` and edit the variables to specify the file path of the pickle file that contains all of the generated scores , the train `.csv` file used to generate the scores and the threshold which you want to filter by ,once the variables have been edited run the notebook to generate a new train `.csv` file  
### Validating models
to test the models 2 almost identical scripts  have been prepared to test all checkpoints of blip base and blip large to test the models ensure you have access to a gpu woth a minimum of 12 gb of vram 
1. To test all **blip base** checkpoints navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/test_blip_checkpoints.ipynb` and edit the 2nd code cell as shown in the sample code below to specify the caption you want to test and the checkpoints you want to test and the path to the validation data csv you have generated in the data preparation step 
```py Edit this to specify checkpoints of models you want to test 
defined as key value pair of {"checpoint name/identifier:"full path to checkpoint"}'''

checkpoints = {"base":"path to base model"}

#Edit this to specify the augmented/original captions you want to test the model checkpoints on 
caption_to_test =  ["caption"]

#path to validation data after preporcessing into csv file format 
dataframe_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_prompt_clean.csv"
```

2. To test all **blip large** checkpoints navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/test_blip_checkpoints large.ipynb` and edit the 2nd code cell as shown in the sample code below to specify the caption you want to test and the checkpoints you want to test and the path to the validation data csv you have generated in the data preparation step 
```py Edit this to specify checkpoints of models you want to test 
defined as key value pair of {"checkpoint name/identifier:"full path to checkpoint"}'''

checkpoints = {"base":"path to base model"}

#Edit this to specify the augmented/original captions you want to test the model checkpoints on 
caption_to_test =  ["caption"]

#path to validation data after preporcessing into csv file format 
dataframe_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_prompt_clean.csv"
```

### Running inference on test data
before you can run any of the test notebooks for submission you will first need to prepare the test data for testing to do this please follow the steps below 
1. to format the data for testing navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/prepare_test_data.ipynb` edit the following file paths in the 2nd code cell if needed as shown in the sample code snippet below and finally run the notebook and you will generate a `.csv` file of the test data 
```py
caption_file = "path to test_captions.json"
test_images  = os.listdir("path to test_imgs folder")
``` 
2. to generate the embeddings for the test data first navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/blip_gen_embed.ipynb` next edit the 2nd code block to specify the models you want to use to generate embeds as well as the path of the `.csv` file of the test data and lastly run the notebook
```py
models = [
       {"path":"path to model checkpoint","model_size":"base or large","model_name":"identifer/name of the model"}

    ]
test_df_path = "path to test.csv"
```
3. lastly to run inference on test data to generate and prepare submission navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/single_model_pred.ipynb` and run the notebook if you want to generate submissions/predictions for only 1 model,if you would like to generate submissions/predictions for ensembles of models navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/ensemble.ipynb` and run the notebook 
#### testing feature fusion and score fusion and other ensambling methods 
if you would like to test feature fusion,score fusion and other ensembling methods navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/test ensemble strategy.ipynb` and edit and run the notebook to experiment with methods

### Training BLIP
Before you can train blip you will first need to process the aliproduct 2 data by following the data preparation steps above 
#### Preparing/formatting data for blip training
1. to prepare your data for blip training navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/eda/blip_dataset_format.ipynb` and edit the 2nd code cell if needed to specify the,input and output files and file paths for each image, refer the the sample code below on how to edit the code , after editing run the notebook to generate the `.json` file for blip training
```py
#file path to where you store your train csv filein your machine if you do not have this file run aliproduct_cvpr_eda_2022.ipynb to generate the base csv file 
train_csv_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/train_data_v6_base.csv"

#file path to where you store your validation csv file in your machine if you do not have this file run aliproduct_cvpr_eda_2022.ipynb to generate the base csv file 
val_csv_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_map.csv"

'''column name in your train csv file that contains the file path for each image 
e.g train_text_img_pairs_{i}/train_text_img_pairs_{i}_compressed/  or train_text_img_pairs_{i}_compressed/ 
where i denotes the file number from 1-9 in the aliproduct dataset''' 

train_images_folder_col = "image_reletive_folder"


#name of folder that your validation images are stored  
val_images_folder = "val_imgs"



'''ONLY FOR PRETRAINING: column name in your train csv file that contains the file path for each image 
e.g /home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/train_text_img_pairs_{i}/train_text_img_pairs_{i}_compressed/image_1.jpg  
or /home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/train_text_img_pairs_{i}_compressed/image_1.jpg 
where i denotes the file number from 1-9 in the aliproduct dataset''' 
train_images_full_path_col = "full_path"


#name of column of train csv that contains the captions
train_caption_col_name = "caption"

#name of column of train csv that contains the the image names with file extension e.g sample_image_1.jpg,sample_image_2.png,etc
train_image_col_name = "product"

#name of column of validation csv that contains the captions
val_caption_col_name = "caption"

#name of column of validation csv that contains the the image names with file extension e.g sample_image_1.jpg,sample_image_2.png,etc
val_image_col_name = "product"


#name of output json file for train data 
train_output_filename = "../data/aliproduct2_train_ann_v6_large_08.json"

#name of output json file for validation data 
val_output_filename = "../data/aliproduct2_val_ann.json"

#name of output json file for training data used only for pretraining  
pretrain_output_filename = "../data/aliproduct2_pretrain_ann_v2.json"
```
##### Training configuration for Retrieval task  
  2. once you have generated the `.json` file containing your training and validation data for training blip you need to edit the blip config file and dataset file to do this first navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/configs/retrieval_aliproduct2.yaml` and edit the `.yaml` file accordingly next navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/data/aliproduct2_dataset.py` and look for the `filename/filenames` variables in each dataset `class` and edit the `.json` filenames to match the ones you have created in the Preparing/formatting data for blip training step
##### Training configuration for Captioning task  
3. similar to training blip for Retrieval task , once you have generated the `.json` file containing your training and validation data for training blip you need to edit the blip config file and dataset file to do this first navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/configs/caption_aliproduct2.yaml` and edit the `.yaml` file accordingly next navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP/data/aliproduct2_dataset.py` and look for the `filename/filenames` variables in each dataset `class` and edit the `.json` filenames to match the ones you have created in the Preparing/formatting data for blip training step

#### Running BLIP training scripts
to run blip training scripts first open your terminal and type in `cd /home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/model/BLIP` to change your working directory to the BLIP folder 
##### Running Retrieval training script
1. to run the blip training script for retrieval task type or copy the following command
```bash
python -m torch.distributed.run --nproc_per_node=4 train_retrieval.py \
--config ./configs/"name of config file".yaml \
--output_dir output/"name of out put folder where model checkpoints are stored"
``` 
##### Running Captioning training script
2. to run the blip training script for retrieval task type or copy the following command
```bash
python -m torch.distributed.run --nproc_per_node=4 train_caption.py \
--config ./configs/"name of config file".yaml \
--output_dir output/"name of out put folder where model checkpoints are stored"
``` 
<p align="right">(<a href="#top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

### [CLIP github repo](https://github.com/openai/CLIP)
```
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```
### [BLIP Github repo](https://github.com/salesforce/BLIP)
```@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language
  Understanding and Generation}, 
  author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
  year={2022},
  booktitle={ICML},
}
```
### [Faiss Github repo](https://github.com/facebookresearch/faiss)
```{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```
<p align="right">(<a href="#top">back to top</a>)</p>
