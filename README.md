# MiE-X

## Introduction

This is an implementation of MiE-X by Pytorch. MiE-X is a large-scale synthetic dataset that improves data-driven micro-expression methods. MiE-X introduces three types of effective Actions Units (AUs) that constitute trainable micro-expressions. This repository provides the implementation of acquiring these AUs and using these AUs to obtain MiE-X.

<!-- ## Overview
Overview of computing three types of Action Units.
![system overview](system.png "System overview of XX.") -->

## Dependencies
MiE-X uses the same libraries as [GANimation](https://github.com/albertpumarola/GANimation)
- python 3.7+
- pytorch 1.6+ & torchvision
- numpy
- matplotlib
- tqdm
- dlib
- face_recognition
- opencv-contrib-python

## Datasets

Baidunetdisk is available. Google drive is coming soon.

|    Variant      | MiE-X (AU_mie)        | MiE-X (AU_mae)  |  MiE-X (AU_exp)        |
|--------------|------------------|------------------|-----------|
| Access     | [Baidu](https://pan.baidu.com/s/1vcOrXyPks-T8PY_UJimGKw?pwd=42i8),[Google](https://drive.google.com/file/d/) | [Baidu](https://pan.baidu.com/s/1kfAz5W2MP1jiVIzfAU9x9g?pwd=81vm),[Google](https://drive.google.com/file/d/) | [Baidu](https://www.) |

## Usage

go to ganimation_replicate

```shell script
cd ganimation_replicate
```


### Extract AUs by the OpenFace toolkit
This work use OpenFace to extract Action Units from real-world expressions. If you would like to extract
AUs by yourself, please follow the official [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
repo. We have extracted AUs from MEGC, MMEW, and Oulu and placed them in the `MER/Data` folder.

### Prepare AUs pools for simulation

E.g., prepare the AU pool for the AUs extracted from the MMEW dataset.

```shell script
python prepare_AUMMEW_pool.py
```


### Simulate MiEs

Simulate image based micro-expressions: 

<!--- use AU<sub>MiE</sub> to simulate --->
```shell script
python simulate.py --mode test --data_root datasets/celebA --gpu_ids 2,3 --ckpt_dir ckpts/emotionNet/ganimation/190327_160828/ --load_epoch 30
```

Preliminary: You need to download your face source dataset and place its path after `--data_root`. The pretrained
GANimation model should be placed in `--ckpt_ddir`. You can train your own GANimation model by following the 
[official](https://github.com/albertpumarola/GANimation) GANimation repo or directly downloading the pretrained model from the 
[third-party](https://github.com/donydchen/ganimation_replicate) implementation. 

Simulate video based micro-expressions: 

<!--- use AU<sub>MaE</sub> to simulate --->

```shell script
python simulate_video_AUexp.py --mode test --data_root datasets/celebA --gpu_ids 2,3 --ckpt_dir ckpts/emotionNet/ganimation/190327_160828/ --load_epoch 30
```

## Train MiE classifers

go to MER

```shell script
cd MER
```
### Train on MiE-X and fine-tune on real-world data
```shell script
bash run_train_fold_fineT.sh
```
