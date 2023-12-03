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
-->

An image should be presented here:

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


## TODO
- [ ] introduction
- [ ] requirements
- [x] data preprocessing
- [x] training code
- [ ] pre-training models
- [x] inference code
- [ ] acknowledgements
- [ ] license
- [ ] contact

<!--


## Pre-trained models
The model and checkpoints are available via torch.hub.

Load the MGD denoising UNet model using the following code:

```
unet = torch.hub.load(
    dataset=<dataset>, 
    repo_or_dir='aimagelab/multimodal-garment-designer', 
    source='github', 
    model='mgd', 
    pretrained=True
    )
```

- ```dataset``` dataset name (dresscode | vitonhd)

Use the denoising network with our custom diffusers pipeline as follow:

```
from pipes.sketch_posemap_inpaint_pipe import StableDiffusionSketchPosemapInpaintPipeline
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "runwayml/stable-diffusion-inpainting"

text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="text_encoder"
    )

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="vae"
    )

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    )

val_scheduler = DDIMScheduler.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="scheduler"
    )
val_scheduler.set_timesteps(50)

val_pipe = ValPipe(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=val_scheduler,
    )
```

For an extensive usage case see the file ```eval.py``` in the main repo.

## Datasets
We do not hold rights on the original Dress Code and Viton-HD datasets. Please refer to the original papers for more information.

Start by downloading the original datasets from the following links:
- Viton-HD **[[link](https://github.com/shadow2496/VITON-HD)]**
- Dress Code **[[link](https://github.com/aimagelab/dress-code)]**


Download the Dress Code Multimodal and Viton-HD Multimodal additional data annotations from here.

- Dress Code Multimodal **[[link](https://drive.google.com/file/d/1y0lHA-4ogjjo9g7VuvcQJrD_CtgjAKhv/view?usp=drive_link)]**
- Viton-HD Multimodal **[[link](https://drive.google.com/file/d/1Z2b9YkyBPA_9ZDC54Y5muW9Q8yfAqWSH/view?usp=share_link)]**

### Dress Code Multimodal Data Preparation
Once data is downloaded prepare the dataset folder as follow:

<pre>
Dress Code
| <b>fine_captions.json</b>
| <b>coarse_captions.json</b>
| test_pairs_paired.txt
| test_pairs_unpaired.txt
| train_pairs.txt
| <b>test_stitch_map</b>
|---- [category]
|-------- images
|-------- keypoints
|-------- skeletons
|-------- dense
|-------- <b>im_sketch</b>
|-------- <b>im_sketch_unpaired</b>
...
</pre>

### Viton-HD Multimodal Data Preparation
Once data is downloaded prepare the dataset folder as follow:

<pre>
Viton-HD
| <b>captions.json</b>
|---- train
|-------- image
|-------- cloth
|-------- image-parse-v3
|-------- openpose_json
|-------- <b>im_sketch</b>
|-------- <b>im_sketch_unpaired</b>
...
|---- test
...
|-------- <b>im_sketch</b>
|-------- <b>im_sketch_unpaired</b>
...
</pre>




## Acknowledgements
This work has partially been supported by the PNRR project “Future Artificial Intelligence Research (FAIR)”, by the PRIN project “CREATIVE: CRoss-modal understanding and gEnerATIon of Visual and tExtual content” (CUP B87G22000460001), both co-funded by the Italian Ministry of University and Research, and by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** you've made.

-->
