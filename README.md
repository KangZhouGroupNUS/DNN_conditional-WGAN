## Contents
- [Overview](#overview)
- [Files](#files)
- [Usage](#usage)
- [Publication](#publication)


## Overview
Protein solubility plays a critical role in improving production yield of recombinant proteins in biocatalysis applications. To some extent, protein solubility can represent the function and activity of biocatalysts which are mainly composed of recombinant proteins. In literature, many machine learning models have been investigated to predict protein solubility from protein sequence, whereas parameters of those models were underdetermined with insucient data of protein solubility. Here we propose a deep neural network (DNN) as a more accurate regression predictive model. Moreover, to tackle the insucient data problem, a novel data augmentation algorithm, Protein Solubility Generative Adversarial Nets (ProGAN), was proposed for improving the prediction of protein solubility. After adding mimic data produced from ProGAN, the prediction performance measured by R2 was improved compared with that without data augmentation. A R2 value of 0.4504 was achieved, which was enhanced about 10% compared with the previous study using the same dataset.


## Files
* "cleandata_1.csv": The whole dataset including 3148 proteins after data preprocessing
* "cleandata_1train.csv": The training data of the balanced dataset
* "cleandata_1test.csv": The test data of balanced dataset
* "fake_sample_epoch_009.npy", "netG_epoch_9.pth": The generated data from conditional WGAN
* "gan.py": The code implementing conditional WGAN
* "main.py": The code implementing DNN
* "bio_dataset.py": The code reshaping the dataset for tensor
* "network.py": The code defining the architecture of DNN

## Usage
* Run main.py with "BioDataset" to get the prediction results for the original dataset
* Run gan.py to generate and save the mimic data files
* Run main.py with "BioMixDataset" to predict protein solubility with the original and generated data

## Publication
Han, X., Zhang, L., Zhou, K., & Wang, X. (2019). ProGAN: Protein solubility generative adversarial nets for data augmentation in DNN framework. Computers & Chemical Engineering, 131, 106533.



