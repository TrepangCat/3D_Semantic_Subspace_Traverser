# 3D_Semantic_Subspace_Traverser (ICCV 2023)

### 3D Semantic Subspace Traverser: Empowering 3D Generative Model with Shape Editing Capability
[**Ruowei Wang**](https://scholar.google.com/citations?user=_-R8Wn8AAAAJ&hl=en),
[**Yu Liu**](https://scholar.google.com/citations?user=-rtPdQ4AAAAJ&hl=en),
[**Pei Su**](https://scholar.google.com/citations?user=ayVfs1kAAAAJ&hl=zh-CN),
Jianwei Zhang,
[**Qijun Zhao**](https://scholar.google.com/citations?user=c2fckoYAAAAJ&hl=en)

<!--**\*** Equal contribution.-->

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.14051)
<!--[![GitHub Stars](https://img.shields.io/github/stars/TrepangCat/3D_Semantic_Subspace_Traverser?style=social)](https://github.com/TrepangCat/3D_Semantic_Subspace_Traverser)-->
<!--[![GitHub Stars](https://img.shields.io/github/stars/aimagelab/multimodal-garment-designer?style=social)](https://github.com/aimagelab/multimodal-garment-designer)-->

This is the **official repository** for the [**paper**](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_3D_Semantic_Subspace_Traverser_Empowering_3D_Generative_Model_with_Shape_ICCV_2023_paper.html) "*3D Semantic Subspace Traverser: Empowering 3D Generative Model with Shape Editing Capability*".

## Overview

<!--
<p align="center">
    <img src="images/1.gif" style="max-width:500px">
</p>
An image should be presented here:
-->

>**Abstract**: <br>
> Shape generation is the practice of producing 3D shapes as various representations for 3D content creation. 
> Previous studies on 3D shape generation have focused on shape quality and structure, without or less considering the importance of semantic information. 
> Consequently, such generative models often fail to preserve the semantic consistency of shape structure or enable manipulation of the semantic attributes of shapes during generation. 
> In this paper, we proposed a novel semantic generative model named 3D Semantic Subspace Traverser that utilizes semantic attributes for category-specific 3D shape generation and editing. 
> Our method utilizes implicit functions as the 3D shape representation and combines a novel latent-space GAN with a linear subspace model to discover semantic dimensions in the local latent space of 3D shapes. 
> Each dimension of the subspace corresponds to a particular semantic attribute, and we can edit the attributes of generated shapes by traversing the coefficients of those dimensions. 
> Experimental results demonstrate that our method can produce plausible shapes with complex structures and enable the editing of semantic attributes.

## Citation
If you make use of our work, please cite our paper:

```bibtex
@inproceedings{ruowei2023traverser,
  title={3D Semantic Subspace Traverser: Empowering 3D Generative Model with Shape Editing Capability},
  author={Wang, Ruowei and Liu, Yu and Su, Pei and Zhang, Jianwei and Zhao, Qijun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Requirements
The code is developed with
- Python 3.7
- Pytorch 1.8.0
- Cuda 11.4

You can also create an environment using the setting file by ```conda env create -f environment.yml```.

## Data Preprocessing

Our data preprocessing code is based on <a href="https://github.com/jchibane/if-net" target="_blank">IF-NET</a> and
<a href="https://github.com/Xharlie/DISN" target="_blank">DISN</a>. Please refer them if you this code. And if you 
have any questions, you can read the original code of 
<a href="https://github.com/jchibane/if-net" target="_blank">IF-NET</a> and
<a href="https://github.com/Xharlie/DISN" target="_blank">DISN</a>.

The following data preprocessing codes are customized to the dataset `ShapeNetCore.v1`, but you can also apply them
to any `*.obj` files after editing.

First, install the needed libraries with:

```
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../..
```

Second, please set the following files executable:
```
data_processing/bin/PreprocessMesh
data_processing/bin/SampleVisibleMeshSurface
data_processing/isosurface/computeDistanceField
data_processing/isosurface/computeMarchingCubes
data_processing/isosurface/displayDistanceField
```

Third, we offer some data samples in `sample_data` folder. You can process the data with:
```
python data_processing/1_create_isosurf.py --input_path=../sample_data/ShapeNetCore.v1 --thread_num=8 --category=chair
python data_processing/2_convert_to_scaled_off.py  --processed_data_dir=../sample_data/ShapeNetCore.v1_processed_tmp
python data_processing/3_voxelize.py --processed_data_dir=../sample_data/ShapeNetCore.v1_processed_tmp -res=64
python data_processing/4_boundary_sampling.py --processed_data_dir=../sample_data/ShapeNetCore.v1_processed_tmp
```

we save the processed data in `../sample_data/ShapeNetCore.v1_processed_tmp`, which is defined in a dictionary named 
`data` in the `data_processing/1_create_isosurf.py`.

## Training Code

1. we train the VAE with:
```
python train_VAE.py --path=<str>
```
- ```path``` is the path of the processed dataset. `./sample_data/ShapeNetCore.v1_processed_tmp` by default.

All training results of VAE are saved in `./run_CubeCodeCraft/****-AE-batch*-steps*`.

2. we can use the trained VAE to generate the latent code for training the GAN:
```
python VAE_produce_data.py path_of_VAE_ckpt
```
- ```path_of_VAE_ckpt``` is the path of a trained VAE ckpt produced by step 1.

It produces shape codes in `./sample_data/ShapeNetCore.v1_processed_tmp_encoded` by default.

3. we use these shape codes to train the GAN:
```
python train_GAN.py \
--path=<str> \
--ae_ckpt_path=<str>
```
- ```path``` is the path of encoded data produced by step 2. `./sample_data/ShapeNetCore.v1_processed_tmp_encoded` by default.
- ```ae_ckpt_path``` is the path of a trained VAE ckpt produced by step 1.

All training results of GAN are saved in `./run_CubeCodeCraft/****-GAN-batch*-steps*`.

## Pre-training Models

Baidu Netdisk: https://pan.baidu.com/s/1BAytVXCq5D3zOdtdkF_inA?pwd=68jr,
(password=68jr)

Dropbox: https://www.dropbox.com/scl/fo/b1yvcsk8yx1rl48gyhb3c/h?rlkey=4ruwe4yzx43q4vpaynhvw4at1&dl=0
## Inference

### generation
```
python GAN_generate_data.py path_of_GAN_ckpt --ae_ckpt_path=<str> --num_generated=<int>
```
- ```path_of_GAN_ckpt``` is the path of a trained GAN ckpt.
- ```ae_ckpt_path``` is the path of a trained VAE ckpt.
- ```num_generated``` is the number of generated results.

Generated results are saved in a folder named as ```generate_*_r=*_threshold=*_***_num*``` nearby ```path_of_GAN_ckpt```.

### shape manipulation (or shape exploration) by the local linear subspace models.
```
python GAN_generate_data_Eigen_newRender.py path_of_GAN_ckpt --ae_ckpt_path=<str> \
--traverse_range=<float> \
--intermediate_points=<int>
--tvs_seed <int> <int> ……
```
- ```path_of_GAN_ckpt``` is the path of a trained GAN ckpt.
- ```ae_ckpt_path``` is the path of a trained VAE ckpt.
- ```traverse_range``` is the number of generated results.
- ```intermediate_points``` is the number of intermediate points during traversing.
- ```tvs_seed``` are seeds to generate shapes for manipulation.

Generated results are saved in a folder named as ```generateEigen_*_r=*_threshold=*_***``` nearby ```path_of_GAN_ckpt```.


## Acknowledgements

our code is heavily built upon 
[**IF-NET**](https://github.com/jchibane/if-net) 
and 
[**EigenGAN**](https://openaccess.thecvf.com/content/ICCV2021/html/He_EigenGAN_Layer-Wise_Eigen-Learning_for_GANs_ICCV_2021_paper.html). 
If you find my code useful, please also consider to cite those papers.

## Contact
Besides the email address in the paper, feel free to send an email to **772438854@qq.com**.

## LICENSE
[**LICENSE**](https://github.com/TrepangCat/3D_Semantic_Subspace_Traverser/blob/master/LICENSE). 


## TODO
- [x] overview
- [x] requirements
- [x] data preprocessing
- [x] training code
- [x] pre-training models
- [x] inference code
- [x] acknowledgements
- [x] contact
- [x] license
