

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

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`


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
1. navigate to `aliproduct_cvpr_eda_2022.ipynb` and edit the code below with your own file paths and desired file names 
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


