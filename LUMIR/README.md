# Large Scale Unsupervised Brain MRI Image Registration (LUMIR)

![](figure/logo.jpeg)

## Introduction

<div align=center>
<img src=".\figure\net.png" width="80%" />
</div>

We introduce our solution for the  LUMIR challenge at [**Learn2Reg 2024**](https://learn2reg.grand-challenge.org/). This official repository houses our method (EOIR), training scripts, and pretrained models. According to the leaderboard, our method achieved a Dice coefficient of **77.37%**, which is **1.43%** higher than the TransMorph. For further details, please check out our paper and [**MICCAI WBIR 2024 workshop**](https://www.wbir.info/).

 [<img src="https://img.shields.io/badge/License-MIT-yellow.svg">](https://opensource.org/license/MIT)    [![arXiv](https://img.shields.io/badge/arXiv-2409.00917-b31b1b.svg)](https://arxiv.org/abs/2409.00917)

|            Method             |     Dice↑      |  TRE(mm)↓  | NDV(%)↓ | HD95(mm)↓  |
| :---------------------------: | :------------: | :--------: | :-----: | :--------: |
| Zero Displacement(Before Reg) |   56.57±2.63​   |   4.3543​   | 0.0000​  |   4.7876​   |
|          TransMorph           |   75.94±3.19​   |   2.4225​   | 0.3509​  |   3.5074​   |
|          uniGradICON          |   73.69±4.12​   |   2.5727​   | 0.0000​  |   3.6107​   |
|          SynthMorph           |   72.43±2.94​   |   2.6099​   | 0.0000​  |   3.5730​   |
|          VoxelMorph           |   71.86±3.40​   |   3.1545​   | 1.1836​  |   3.9821​   |
|           deedsBCV            |   69.77±2.74​   | **2.2230** | 0.0001​  |   3.9540​   |
|        **Ours(EOIR)**         | **77.37±3.11** |   2.3498​   | 0.0002​  | **3.3296** |

## Competition Ranking

- ### Final Ranking (Test Phase):

The final ranking will only be announced during the [**MICCAI WBIR 2024 workshop**](https://www.wbir.info/) in Marrakesh by Sunday, October 6th.

- ### Early Acceptance and Initial Results [Snapshot Rank](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/6fd1cc17fe6ad1f460617163a6e824d5cdda105b/README.md) (updated: 07/31/2024):

In the early acceptance, our team (**next-gen-nn**) have won the rank 2 on the LUMIR leaderboard! 

According to the information released by the official organization of the competition, *the ranking process involved normalizing the scores using the Min-Max normalization technique.* After normalization, the scores were aggregated into a weighted average, calculated as follows:
$$\text{Average Score}=\frac{1}{6}\times Norm. Dice + \frac{1}{6}\times Norm. HdDist95 + \frac{1}{3}\times Norm. TRE + \frac{1}{3}\times Norm. NDV$$

|           Author(Team)           | Normalized Dice | Normalized TRE | Normalized NDV | Normalized HdDist95 | Average Score |  Rank   |
| :------------------------------: | :-------------: | :------------: | :------------: | :-----------------: | :-----------: | :-----: |
|             honkamj              |     1.0000      |     0.9815     |     0.9991     |       0.9934        |    0.9925     |   1.0   |
|   ***hnuzyx  (next-gen-nn)***    |   **0.9975**    |   **0.9764**   |   **1.0000**   |     **0.9867**      |  **0.9895**   | **2.0** |
|   tsubasaz025  (DutchMasters)    |     0.9798      |     0.9809     |     0.9999     |       0.9876        |    0.9882     |   3.0   |
|            793407238             |     0.9874      |     0.9816     |     0.9987     |       0.9708        |    0.9865     |   4.0   |
|     Wjiazheng  (next-gen-nn)     |     0.9957      |     0.9718     |     0.9947     |       0.9824        |    0.9852     |   5.0   |
|            lie_weaver            |     0.9947      |     0.9760     |     0.9901     |       0.9753        |    0.9837     |   6.0   |
|  windforever118  (next-gen-nn)   |     0.9932      |     0.9626     |     0.9944     |       0.9915        |    0.9831     |   7.0   |
|             Bailiang             |     0.9652      |     0.9754     |     0.9992     |       0.9638        |    0.9797     |   8.0   |
|           zhuoyuanw210           |     0.9836      |     0.9572     |     0.9982     |       0.9828        |    0.9795     |   9.0   |
|        LYU-zhouhu  (LYU1)        |     0.9861      |     0.9388     |     0.9984     |       0.9816        |    0.9737     |  10.0   |
|        cwmokab  (Orange)         |     0.9768      |     0.9550     |     0.9878     |       0.9727        |    0.9725     |  11.0   |
| jchen245  (Challenge Organizers) |     0.9626      |     0.9583     |     0.9863     |       0.9508        |    0.9671     |  12.0   |
|         Sparkling_Poetry         |     0.9798      |     0.9459     |     0.9791     |       0.9608        |    0.9651     |  13.0   |
|            zahid_aziz            |     0.9631      |     0.9484     |     0.9877     |       0.9546        |    0.9650     |  14.0   |
|              lukasf              |     0.9634      |     0.9506     |     0.9861     |       0.9474        |    0.9640     |  15.0   |
|               ...                |       ...       |      ...       |      ...       |         ...         |      ...      |   ...   |

## Setup

### 1、Create Environment Using Conda:

```bash
conda create -n LUMIR python=3.8
conda activate LUMIR
```

### 2、Install the Pytorch Environment:

- **Pytorch(CUDA):**

```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

- **Pytorch(CPU):**

```bash
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3、Install the Other packages

```bash
pip install -r requirements.txt
```

### 4、Download the Pretrained Weights

Download the checkpoint of our model (EOIR) via Google Drive [<img src="https://img.shields.io/badge/Google-Checkpoint-darkgreen?style=for-the-badge&labelColor=black&logo=google" width="15%" />](https://drive.google.com/file/d/1DcwfuxM5NfjucYJ1Kr_iDAsG4q-Xf6Hy/view)(~194MB).

## Dataset

- All images are converted to NIfTI, resampled, and cropped to the region of interest, resulting in an image size of $160\times 224\times 192$ with a voxel spacing of $1\times 1\times 1$ mm.
- Training images of 3384 subjects (Including 10 Segmentation label for sanity-check), 38 image pairs for validation, Test images of 590 subjects.
- Download the *Training/Validation* data and *Dataset JSON* file by [Learn2Reg 2024](https://learn2reg.grand-challenge.org/).
- Dataset structure:

```bash
LUMIR_L2R24_TrainVal/imagesTr/------
        LUMIRMRI_0000_0000.nii.gz
        LUMIRMRI_0001_0000.nii.gz
        LUMIRMRI_0002_0000.nii.gz
        .......
LUMIR_L2R24_TrainVal/imagesVal/------
        LUMIRMRI_3454_0000.nii.gz
        LUMIRMRI_3455_0000.nii.gz
        .......
LUMIR_L2R24_TrainVal/SanityCheckLabelsTr/------
        LUMIRMRI_3364_0000.nii.gz   
        LUMIRMRI_3365_0000.nii.gz
        .......
```

<div align=center>
<img src=".\figure\dataset.png" width="50%" />
</div>

## Getting Started

- ### Train

**Run the bash script:**

```bash
bash train.sh
```

**or use the command:**

```bash
python train_registration_lumir.py --epochs 201 -bs 1 --start_channel 32 --gpu_id 0
```

- ### Test

```bash
python test.py
```

## Docker

Docker allows for running an algorithm in an isolated environment called a container. To avoid compatibility issues with CUDA, we provided our CPU-based Pytorch docker during test phase submission of the L2R LUMIR Challenge. 

- ### Build the Docker Container

```bash
docker build -f Dockerfile -t eoir .
```

- ### Run the Docker

```bash
bash test.sh
```

- ### Save the Docker Container

```bash
docker save eoir | gzip -c > eoir.tar.gz
```

## Citations

If you find this code is useful in your research, please consider to cite:

```
@misc{zhang2024largescaleunsupervisedbrain,
      title={Large Scale Unsupervised Brain MRI Image Registration Solution for Learn2Reg 2024}, 
      author={Yuxi Zhang and Xiang Chen and Jiazheng Wang and Min Liu and Yaonan Wang and Dongdong Liu and Renjiu Hu and Hang Zhang},
      year={2024},
      eprint={2409.00917},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00917}, 
}
```

## Reference

- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- [SynthMorph](https://martinos.org/malte/synthmorph/)
- [UniGradICON](https://github.com/uncbiag/uniGradICON)
- [BrainMorph](https://github.com/alanqrwang/brainmorph)



