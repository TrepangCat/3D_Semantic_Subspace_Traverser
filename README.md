# 3D_Semantic_Subspace_Traverser (ICCV 2023)

### 3D Semantic Subspace Traverser: Empowering 3D Generative Model with Shape Editing Capability
[**Ruowei Wang**](https://scholar.google.com/citations?user=_-R8Wn8AAAAJ&hl=en),
Yu Liu,
[**Pei Su**](https://scholar.google.com/citations?user=ayVfs1kAAAAJ&hl=zh-CN),
Jianwei Zhang,
[**Qijun Zhao**](https://scholar.google.com/citations?user=c2fckoYAAAAJ&hl=en)

<!--**\*** Equal contribution.-->

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](TODO:待定)
<!--[![GitHub Stars](https://img.shields.io/github/stars/TrepangCat/3D_Semantic_Subspace_Traverser?style=social)](https://github.com/TrepangCat/3D_Semantic_Subspace_Traverser)-->
<!--[![GitHub Stars](https://img.shields.io/github/stars/aimagelab/multimodal-garment-designer?style=social)](https://github.com/aimagelab/multimodal-garment-designer)-->

This is the **official repository** for the [**paper**](TODO:arxiv链接) "*3D Semantic Subspace Traverser: Empowering 3D Generative Model with Shape Editing Capability*".

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

## TODO
- [ ] introduction
- [ ] requirements
- [ ] data preporocessing
- [ ] training code
- [ ] pre-training models
- [ ] inference code
- [ ] acknowledgements
- [ ] license
- [ ] contact

<!--
## Inference

To run the inference please use the following:

```
python eval.py --dataset_path <path> --batch_size <int> --mixed_precision fp16 --output_dir <path> --save_name <string> --num_workers_test <int> --sketch_cond_rate 0.2 --dataset <dresscode|vitonhd> --start_cond_rate 0.0
```

- ```dataset_path``` is the path to the dataset (change accordingly to the dataset parameter)
- ```dataset``` dataset name to be used
- ```output_dir``` path to the output directory
- ```save_name``` name of the output dir subfolder where the generated images are saved
- ```start_cond_rate``` rate {0.0,1.0} of denoising steps that will be used as offset to start sketch conditioning
- ```sketch_cond_rate``` rate {0.0,1.0} of denoising steps in which sketch cond is applied
- ```test_order``` test setting (paired | unpaired)

Note that we provide few sample images to test MGD simply cloning this repo (*i.e.*, assets/data). To execute the code set 
- Dress Code Multimodal dataset
    - ```dataset_path``` to ```assets/data/dresscode```
    - ```dataset``` to ```dresscode```
- Viton-HD Multimodal dataset
    - ```dataset_path``` to ```assets/data/vitonhd```
    - ```dataset``` to ```vitonhd```

It is possible to run the inference on the whole Dress Code Multimodal or Viton-HD Multimodal dataset simply changing the ```dataset_path``` and ```dataset``` according with the downloaded and prepared datasets (see sections below).


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
