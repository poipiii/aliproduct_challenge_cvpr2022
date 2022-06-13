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

### Validating models
to test the models 3 almost identical scirpts have been prepared to test all checkpoints of clip,blip base and blip large to test the models ensure you have access to a gpu woth a minimum of 12 gb of vram 
1. To test all **blip base** checkpoints navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/test_blip_checkpoints.ipynb` and edit the 2nd code cell as shown in the sample code below to specify the caption you want to test and the checkpoints you want to test and the path to the validation data csv you have generated in the data preperation step 
```py Edit this to specify checkpoints of models you want to test 
defined as key value pair of {"checpoint name/identifier:"full path to checkpoint"}'''

checkpoints = {"base":"path to base model"}

#Edit this to specify the augmented/original captions you want to test the model checkpoints on 
caption_to_test =  ["caption"]

#path to validation data after preporcessing into csv file format 
dataframe_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_prompt_clean.csv"
```

2. To test all **blip large** checkpoints navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/testing/test_blip_checkpoints large.ipynb` and edit the 2nd code cell as shown in the sample code below to specify the caption you want to test and the checkpoints you want to test and the path to the validation data csv you have generated in the data preperation step 
```py Edit this to specify checkpoints of models you want to test 
defined as key value pair of {"checpoint name/identifier:"full path to checkpoint"}'''

checkpoints = {"base":"path to base model"}

#Edit this to specify the augmented/original captions you want to test the model checkpoints on 
caption_to_test =  ["caption"]

#path to validation data after preporcessing into csv file format 
dataframe_path = "/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/data/val_data_prompt_clean.csv"
```

### Running infrence on test data
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
3. lastly to run infrence on test data to geneerate and prepare submission navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/single_model_pred.ipynb` and run the noetbook if you want to generate submissions/predictions for only 1 model,if you would like to generate submissions/predictions for ensamble of models navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/ensemble.ipynb` and run the notebook 
#### testing feature fusion and score fusion and other ensambling methods 
if you would like to test feature fusion,score fusion and other ensambling methods navigate to `/home/ubuntu/Desktop/CVPR 2022 AliProducts Challenge/code/test_set_inference/test ensemble strategy.ipynb` and edit and run the notebook to expriment with methods


<p align="right">(<a href="#top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

### [CLIP github repo](https://github.com/openai/CLIP)
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
