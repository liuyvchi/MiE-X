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

## Usage

### Extract AUs by OpenFace toolkit
```shell script
python3 get_aus.py --persons_path PATH_TO_YOUR_VIDEOS
```
### Simulate MiEs 

use $AU_{MiE}$ to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_realAU.py 
```
use AU<sub>MiE</sub> to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_ck.py 
```
use $AU_{exp}$ to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_data.py 
```
