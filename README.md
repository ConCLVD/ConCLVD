# ConCLVD

This repository is the official implementation of [ConCLVD].

**[ConCLVD: Controllable Chinese Landscape Video Generation via Diffusion Model]**


## Introduction

<br><img src="__assets__/figs/1.jpg" style="width: 50em; margin-top: 1em">

Chinese landscape painting is a gem of Chinese cultural and artistic heritage that showcases the splendor of nature through the deep observations and imaginations of its painters. Limited by traditional techniques, these artworks were confined to static imagery in ancient times, leaving the dynamism of landscapes and the subtleties of artistic sentiment to the viewer's imagination. Recently, emerging text-to-video (T2V) diffusion methods have shown significant promise in video generation, providing hope for the creation of dynamic Chinese landscape paintings. However, challenges such as the lack of specific datasets, the intricacy of artistic styles, and the creation of extensive, high-quality videos pose difficulties for these models in generating Chinese landscape painting videos. This paper introduces CLV-HD (Chinese Landscape Video-High Definition), a novel T2V dataset for Chinese landscape painting videos, and ConCLVD (Controllable Chinese Landscape Video Diffusion), a T2V model that utilizes Stable Diffusion. Specifically, we present a motion module featuring a dual attention mechanism to capture the dynamic transformations of landscape imageries, alongside a noise adapter to leverage unsupervised contrastive learning in the latent space. Following the generation of keyframes, we employ sparse optical flow for frame interpolation to enhance video smoothness. Our method not only retains the essence of the landscape painting imageries but also achieves dynamic transitions, significantly advancing the field of artistic video generation.

## CLV-HD(Our Dataset)

We have created CLV-HD, a pioneering text-to-Chinese landscape painting video dataset, which includes 1,300 high-definition text-video pairs. You can download it from [CLV-HD](https://drive.google.com/drive/folders/1uw_rPz5zbCpGO1yBQ4X-IubVB_jXEHwP?usp=drive_link).

## Setups for Inference

### Prepare Environment

***ConCLVD takes only ~24GB VRAM to train and inference, and run on a single RTX3090 !!***

```
git clone https://github.com/ConCLVD/ConCLVD.git
cd ConCLVD

conda env create -f environment.yaml
conda activate conclvd
```

### Download Base T2I
```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/

```
### Download Motion Module Checkpoints
You can directly download the motion module checkpoints from [ConCLVD-motion.ckpt]https://drive.google.com/file/d/1aDMdAe0o8OTCmy9_CF7UGEqYyKCx-yYu/view?usp=sharing, then put them in `models/Motion_Module/` folder.

### Prepare Personalize T2I
Here we provide inference configs for a demo T2I on CivitAI. 
You may run the following bash script to download this checkpoint.

```
bash download_bashscripts/ToonYou.sh
```

### Inference
We have created a Gradio demo to make ConCLVD easy to use. To launch the demo, please run the following commands:
```
conda activate conclvd
python app.py
```
(We provide two preset text descriptions, or you can also input your own custom description.)

<br><img src="__assets__/figs/gradio.png" style="width: 50em; margin-top: 1em">
## Steps for Training

### Dataset
Before initiating the training process, ensure to download the video files and csv annotations from [CLV-HD](https://drive.google.com/drive/folders/1uw_rPz5zbCpGO1yBQ4X-IubVB_jXEHwP?usp=drive_link) to your local machine. It's important to note that the training script we provide as an example requires all video files to be stored in a single designated folder.

### Configuration
After dataset preparations, update the below data paths in the config `.yaml` files in `configs/training/` folder:
```
train_data:
  csv_path:     [Replace with csv Annotation File Path]
  video_folder: [Replace with Video Folder Path]
  sample_size:  256
```
Other training parameters (lr, epochs, validation settings, etc.) are also included in the config files.

### Training
To train motion modules,do
```
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/training.yaml
```


## Gallery
Here we demonstrate several excellent results we found in our experiments.

<table class="center">
    <tr>
    <td><img src="__assets__/animations/1.gif"></td>
    <td><img src="__assets__/animations/2.gif"></td>
    <td><img src="__assets__/animations/3.gif"></td>
    <td><img src="__assets__/animations/4.gif"></td>
    <td><img src="__assets__/animations/5.gif"></td>
    </tr>
</table>
<table class="center">
    <td><img src="__assets__/animations/6.gif"></td>
    <td><img src="__assets__/animations/7.gif"></td>
    <td><img src="__assets__/animations/8.gif"></td>
    <td><img src="__assets__/animations/9.gif"></td>
    <td><img src="__assets__/animations/10.gif"></td>
    </tr>
</table>


