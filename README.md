# MiE-X

## Introduction

This is an implementation of MiE-X by Pytorch. MiE-X is a large-scale synthetic dataset that improves data-driven micro-expression methods. MiE-X introduces three types of effective Actions Units (AUs) that constitute trainable micro-expressions. This repository provides the implementation of acquiring these AUs and using these AUs to obtain MiE-X.

<!-- ## Overview
Overview of computing three types of Action Units.
![system overview](system.png "System overview of XX.") -->

## Datasets
MiE-X uses the same libraries as [GANimation](https://github.com/albertpumarola/GANimation)
- python 3.7+
- pytorch 1.6+ & torchvision
- numpy
- matplotlib
- tqdm
- dlib
- face_recognition
- opencv-contrib-python

## Dependencies
We make generated images from VehicleX directly. We have performed domain adaptation (both content level and style level) from VehicleX to VeRi-776, VehicleID and CityFlow respectively. They can be used to augment real-world data. The adaptated images can be downloaded the tabel below. 

|    Variant      | MiE-X (MEGC)        | MiE-X (MMEW)        | MiE-X (Oulu)  |
|--------------|------------------|------------------|-----------|
| Access     | [Baidu](https://pan.baidu.com/s/)(pwd:nz36),[Google](https://drive.google.com/file/d/) | [Baidu](https://pan.baidu.com/s/)(pwd:akjh),[Google](https://drive.google.com/file/d/) | [Website](https://www.) |

## Usage

### Extract AUs by the OpenFace toolkit
```shell script
python3 get_aus.py --persons_path PATH_TO_YOUR_VIDEOS
```
### Simulate MiEs 

use AU<sub>MiE</sub> to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_realAU.py 
```
use AU<sub>MaE</sub> to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_ck.py 
```
use AU<sub>exp</sub> to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_data.py 
```
