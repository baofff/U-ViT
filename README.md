## All are Worth Words: A ViT Backbone for Diffusion Models

<img src="uvit.png" alt="drawing" width="400"/>


This is a PyTorch implementation of the paper [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152).


## Dependency

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3
pip install accelerate==0.12.0 timm==0.3.2 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1
```

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.



## Pretrained Models


|                                                      Model                                                      | FID  |
|:---------------------------------------------------------------------------------------------------------------:|:----:|
|        [CIFAR10](https://drive.google.com/file/d/1yoYyuzR_hQYWU0mkTj659tMTnoCWCMv-/view?usp=share_link)         | 3.11 |
|      [CelebA 64x64](https://drive.google.com/file/d/13YpbRtlqF1HDBNLNRlKxLTbKbKeLE06C/view?usp=share_link)      | 2.87 |
|  [ImageNet 64x64 (Mid)](https://drive.google.com/file/d/1igVgRY7-A0ZV3XqdNcMGOnIGOxKr9azv/view?usp=share_link)  | 5.85 |
| [ImageNet 64x64 (Large)](https://drive.google.com/file/d/19rmun-T7RwkNC1feEPWinIo-1JynpW7J/view?usp=share_link) | 4.26 |
|    [ImageNet 256x256](https://drive.google.com/file/d/1w7T1hiwKODgkYyMH9Nc9JNUThbxFZgs3/view?usp=share_link)    | 3.40 |
|    [ImageNet 512x512](https://drive.google.com/file/d/1mkj4aN2utHMBTWQX9l1nYue9vleL7ZSB/view?usp=share_link)    | 4.67 |
|    [MS-COCO (Small)](https://drive.google.com/file/d/15JsZWRz2byYNU6K093et5e5Xqd4uwA8S/view?usp=share_link)     | 5.95 |
| [MS-COCO (Small-Deep)](https://drive.google.com/file/d/1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71/view?usp=share_link)   | 5.48 |



## Data Preparation
* ImageNet 64x64: Put the standard ImageNet dataset (which contains the `train` and `val` directory) to `assets/datasets/ImageNet`.
* ImageNet 256x256 and ImageNet 512x512: Extract ImageNet features according to `scripts/extract_imagenet_feature.py`.
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.


## Evaluation

Firstly download reference statistics for FID and the autoencoder (converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)) from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing).
Put the downloaded directory as `assets/fid_stats` and `assets/stable-diffusion`.

Then compute the FID score:

```
# CIFAR10
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval.py --config=configs/cifar10_uvit.py --nnet_path=cifar10_uvit.pth

# CelebA 64x64
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval.py --config=configs/celeba64_uvit.py --nnet_path=celeba_uvit.pth

# ImageNet 64x64 (Mid)
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval.py --config=configs/imagenet64_uvit_mid.py --nnet_path=imagenet64_uvit_mid.pth

# ImageNet 64x64 (Large)
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval.py --config=configs/imagenet64_uvit_large.py --nnet_path=imagenet64_uvit_large.pth

# ImageNet 256x256
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval_ldm.py --config=configs/imagenet256_uvit.py --nnet_path=imagenet256_uvit.pth

# ImageNet 512x512
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval_ldm.py --config=configs/imagenet512_uvit.py --nnet_path=imagenet512_uvit.pth

# MS-COCO (Small)
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval_t2i_discrete.py --config=configs/mscoco_uvit.py --nnet_path=mscoco_uvit_small.pth

# MS-COCO (Small-Deep)
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args eval_t2i_discrete.py --config=configs/mscoco_uvit.py --config.nnet.depth=16 --nnet_path=mscoco_uvit_small_deep.pth
```

* The generated images are stored in a temperary directory, and will be deleted after evaluation. If you want to keep these images, set `--config.sample.path=/save/dir`.


## Training


```
# CIFAR10
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train.py --config=configs/cifar10_uvit.py

# CelebA 64x64
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train.py --config=configs/celeba64_uvit.py 

# ImageNet 64x64 (Mid)
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train.py --config=configs/imagenet64_uvit_mid.py

# ImageNet 64x64 (Mid)
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train.py --config=configs/imagenet64_uvit_large.py

# ImageNet 256x256
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train_ldm.py --config=configs/imagenet256_uvit.py

# ImageNet 512x512
accelerate_args="--multi_gpu --num_processes 8 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train_ldm.py --config=configs/imagenet512_uvit.py

# MS-COCO (Small)
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train_t2i_discrete.py --config=configs/mscoco_uvit.py

# MS-COCO (Small-Deep)
accelerate_args="--multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port $(expr $RANDOM % 10000 + 10000)"
accelerate launch $accelerate_args train_t2i_discrete.py --config=configs/mscoco_uvit.py --config.nnet.depth=16
```


## This implementation is based on

* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [https://github.com/LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
