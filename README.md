## [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)<br> <sub>A PyTorch implementation</sub>


<img src="uvit.png" alt="drawing" width="400"/>

This codebase implements the transformer-based backbone 💡*U-ViT*💡 for diffusion models, as introduced in the [paper](https://arxiv.org/abs/2209.12152).
U-ViT treats all inputs as tokens and employs long skip connections. *The long skip connections grealy promote the performance and the convergence speed*.



<img src="skip_im.png" alt="drawing" width="400"/>


💡 This codebase contains:
* An implementation of [U-ViT](libs/uvit.py) with optimized attention computation
* Pretrained U-ViT models on common image generation benchmarks (CIFAR10, CelebA 64x64, ImageNet 64x64, ImageNet 256x256, ImageNet 512x512)
* Efficient training scripts for [pixel-space diffusion models](train.py), [latent space diffusion models](train_ldm_discrete.py) and [text-to-image diffusion models](train_t2i_discrete.py)
* Efficient evaluation scripts for [pixel-space diffusion models](eval.py) and [latent space diffusion models](eval_ldm_discrete.py) and [text-to-image diffusion models](eval_t2i_discrete.py)
* TODO: add a Colab notebook for ImageNet demos

💡 This codebase supports useful techniques for efficient training and sampling of diffusion models:
* Mixed precision training with the [huggingface accelerate](https://github.com/huggingface/accelerate) library
* Efficient attention computation with the [facebook xformers](https://github.com/facebookresearch/xformers) library
* Gradient checkpointing trick, which reduces ~80% memory



## Dependency

```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.3
pip install accelerate==0.12.0 timm==0.3.2 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1
conda install xformers  # This is optional, but it would greatly speed up the attention computation.
```

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.
* We highly suggest install [xformers](https://github.com/facebookresearch/xformers), which would greatly speed up the attention computation for *both training and inference*.



## Pretrained Models


|                                                         Model                                                          |  FID  | training iterations | batch size |
|:----------------------------------------------------------------------------------------------------------------------:|:-----:|:-------------------:|:----------:|
|      [CIFAR10 (U-ViT-S/2)](https://drive.google.com/file/d/1yoYyuzR_hQYWU0mkTj659tMTnoCWCMv-/view?usp=share_link)      | 3.11  |        500K         |    128     |
|   [CelebA 64x64 (U-ViT-S/4)](https://drive.google.com/file/d/13YpbRtlqF1HDBNLNRlKxLTbKbKeLE06C/view?usp=share_link)    | 2.87  |        500K         |    128     |
|  [ImageNet 64x64 (U-ViT-M/4)](https://drive.google.com/file/d/1igVgRY7-A0ZV3XqdNcMGOnIGOxKr9azv/view?usp=share_link)   | 5.85  |        300K         |    1024    |
|  [ImageNet 64x64 (U-ViT-L/4)](https://drive.google.com/file/d/19rmun-T7RwkNC1feEPWinIo-1JynpW7J/view?usp=share_link)   | 4.26  |        300K         |    1024    |
| [ImageNet 256x256 (U-ViT-L/2)](https://drive.google.com/file/d/1w7T1hiwKODgkYyMH9Nc9JNUThbxFZgs3/view?usp=share_link)  | 3.40  |        300K         |    1024    |
| [ImageNet 256x256 (U-ViT-H/2)](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=share_link)  | 2.29  |        500K         |    1024    |
| [ImageNet 512x512 (U-ViT-L/4)](https://drive.google.com/file/d/1mkj4aN2utHMBTWQX9l1nYue9vleL7ZSB/view?usp=share_link)  | 4.67  |        500K         |    1024    |
| [ImageNet 512x512 (U-ViT-H/4)](https://drive.google.com/file/d/1uegr2o7cuKXtf2akWGAN2Vnlrtw5YKQq/view?usp=share_link)  | 4.05  |        500K         |    1024    |
|      [MS-COCO (U-ViT-S/2)](https://drive.google.com/file/d/15JsZWRz2byYNU6K093et5e5Xqd4uwA8S/view?usp=share_link)      | 5.95  |         1M          |    256     |
|   [MS-COCO (U-ViT-S/2, Deep)](https://drive.google.com/file/d/1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71/view?usp=share_link)   | 5.48  |         1M          |    256     |



## Data Preparation
* ImageNet 64x64: Put the standard ImageNet dataset (which contains the `train` and `val` directory) to `assets/datasets/ImageNet`.
* ImageNet 256x256 and ImageNet 512x512: Extract ImageNet features according to `scripts/extract_imagenet_feature.py`.
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.



## Training

Firstly prepare the data as instructed above. Then download reference statistics for FID (which is used to monitor FID during training) and the autoencoder (converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), which is necessary for training in the latent space) from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing).
Put the downloaded directories as `assets/fid_stats` and `assets/stable-diffusion` respectively.


We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:
```sh
# the training setting
num_processes=2  # the number of gpus you have, e.g., 2
train_script=train.py  # the train script, one of <train.py|train_ldm.py|train_ldm_discrete.py|train_t2i_discrete.py>
                       # train.py: training on pixel space
                       # train_ldm.py: training on latent space with continuous timesteps
                       # train_ldm_discrete.py: training on latent space with discrete timesteps
                       # train_t2i_discrete.py: text-to-image training on latent space
config=configs/cifar10_uvit_small.py  # the training configuration
                                      # you can change other hyperparameter by modifying the configuration file

# launch training
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 $train_script --config=$config
```


We provide all commands to reproduce U-ViT training in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/cifar10_uvit_small.py

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/celeba64_uvit_small.py 

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_mid.py

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_large.py

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet256_uvit_large.py

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet512_uvit_large.py

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16
```



## Evaluation (Compute FID)

Firstly download reference statistics for FID and the autoencoder (converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), which is necessary for latent diffusion models) from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing).
Put the downloaded directories as `assets/fid_stats` and `assets/stable-diffusion` repectively.


We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:
```sh
# the training setting
num_processes=2  # the number of gpus you have, e.g., 2
eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/cifar10_uvit_small.py  # the training configuration

# launch training
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 $train_script --config=$config
```
The generated images are stored in a temperary directory, and will be deleted after evaluation. 💡If you want to keep these images, set `--config.sample.path=/save/dir`.


We provide all commands to reproduce FID results in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small.py --nnet_path=cifar10_uvit_small.pth

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/celeba64_uvit_small.py --nnet_path=celeba64_uvit_small.pth

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_mid.py --nnet_path=imagenet64_uvit_mid.pth

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_large.py --nnet_path=imagenet64_uvit_large.pth

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet256_uvit_large.py --nnet_path=imagenet256_uvit_large.pth

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet512_uvit_large.py --nnet_path=imagenet512_uvit_large.pth

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16 --nnet_path=mscoco_uvit_small_deep.pth
```




## References
If you find the code useful for your research, please consider citing
```bib
@article{bao2022all,
  title={All are Worth Words: a ViT Backbone for Score-based Diffusion Models},
  author={Bao, Fan and Li, Chongxuan and Cao, Yue and Zhu, Jun},
  journal={arXiv preprint arXiv:2209.12152},
  year={2022}
}
```

This implementation is based on
* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [https://github.com/LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
