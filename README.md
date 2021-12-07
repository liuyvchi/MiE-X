# MiE-X

## Introduction

This is an implementation of MiE-X by Pytorch. MiE-X is a large-scale synthetic dataset that improves data-driven micro-expression methods. MiE-X introduces three types of effective Actions Units (AUs) that constitute trainable micro-expressions. This repository provides the implementation of acquiring these AUs and using these AUs to obtain MiE-X.


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
use $AU_{MaE}$ to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_ck.py 
```
use $AU_{exp}$ to simulate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 simulate_data.py 
```
