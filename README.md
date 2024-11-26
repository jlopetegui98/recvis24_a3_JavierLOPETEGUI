## Object recognition and computer vision 2024/2025

Javier Alejandro LOPETEGUI GONZALEZ
*This repository is mainly based on the one provided with the orientation of the assigment*

### Assignment 3: Sketch image classification
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PxshEMwNm4tLu8f_Bz_Z0emUlC1TPob4?usp=sharing)
#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 500 different classes of sketches adapted from the [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch).
Download the training/validation/test images from [here](https://www.kaggle.com/competitions/mva-recvis-2024/data). The test image labels are not provided.

#### Training and validating your model

To run the training process you must download the dataset using the next commands (kaggle credentials required):
```bash
!kaggle competitions download -c mva-recvis-2024
!unzip mva-recvis-2024.zip
```
Then, run the following line to run the solution which obtained the best public test accuracy in the challenge:
```bash
!python recvis24_a3/main.py --model_name dinov2
```
The main customization parameters added in my solution for the training are the following:
- weight_path: Dinov2 version to use for feature extraction in the model. The default value will be "facebook/dinov2-giant". You can use also "facebook/dinov2-base" or "facebook/dinov2-large"
- embedding_strategy: You can use on of the following approaches:
  - "cls": Use the cls token embedding
  - "seq_emb": Use the pooled average embedding (average over all the tokens)
  - "cls+seq_emb": The concatenation of the two embeddings mentioned before. This will be the default value
- frozen_strategy: For the freezing strategie:
  - "none": do not freeze any parameter
  - "all": freeze all feature extractor model parameters
  - "n-1_attention": freeze all feature extractor model parameters but the last attention head
- aug_flag: a boolean indicating wheter to use data augmentation or not (False as default value)
- dropout: A float value between 0 and 1 indicating the level of dropout to apply after the feature extraction with the base model
 

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file] --model_name [model_name]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.


#### Logger

The training details for the approaches already tried are in the following wand report: [HW3_Report_JavierLOPETEGUI](https://api.wandb.ai/links/nlp-tasks/qr77to53)

### Report

The report of the implementation is available in the file: [HW3_Report_JavierLOPETEGUI.pdf]()


#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Origial adaptation done by Gul Varol: https://github.com/gulvarol<br/>
New Sketch dataset and code adaptation done by Ricardo Garcia and Charles Raude: https://github.com/rjgpinel, http://imagine.enpc.fr/~raudec/
