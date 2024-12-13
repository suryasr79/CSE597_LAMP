<p align="center">
  <img src="assets/LOGO.png" height=170>
</p>

# <p align="center"> LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation </p>
This repo is a clone of the [code repository](https://github.com/RQ-Wu/LAMP) for the [LAMP paper](https://arxiv.org/pdf/2310.10769) and implementing the same as a part of the course project report for CSE597 Vision and Language course.
The task addressed in this project is few-shot-based text-to-video (T2V) generation using diffusion models. 
The goal is to generate high quality and temporally consistent videos from textual prompts while using minimal video data (8–16 videos) for model training. 
The challenge lies in finding a balance between generation freedom and resource efficiency, while ensuring coherence in the generated frames.
We only need 8~16 videos and 1 GPU with atleast 15 GB VRAM for training.

## Prerequisites
### Dependencies and Installation
- Ubuntu > 18.04
- CUDA=11.3
- Others:

```bash
# Since I was working with W135 Machine I've dcumented the additional steps I follwed to get everthing working
ssh <username>@<w135_ip_addr>
cd /scratch/<username>

# set up git and clone the repo
git clone https://github.com/RQ-Wu/LAMP.git
cd LAMP

# setup conda in the scratch space
mkdir -p /scratch/<username>/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/<username>/miniconda3/miniconda.sh
bash /scratch/<username>/miniconda3/miniconda.sh -b -u -p /scratch/<username>/miniconda3
rm /scratch/<username>/miniconda3/miniconda.sh
source /scratch/<username>/miniconda3/bin/activate

# create virtual environment
conda create -n LAMP python=3.8
conda activate LAMP

# install packages
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install xformers==0.0.13

#setup jupyter notebook with remote forwarding to run .ipynb file in local browser
pip install notebook 
jupyter notebook --no-browser --port=7070

#run below command on local machine to enable remote forwarding
ssh -NfL localhost:7979:localhost:7070 <username>@<w135_ip_addr>
```

### Weights and Data
1. You can download pre-trained T2I diffusion models on Hugging Face.
In our work, we use [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) as our backbone network. Clone the pretrained weights by `git-lfs` and put them in `./checkpoints`

2. Our checkpoint and training data are listed as follows. You can also collect video data by your own and put .mp4 files in `./training_videos/[motion_name]/`

3. [Update] You can find the training video for video editing demo in `assets/run.mp4`
<table class="center">
<tr>
    <td align="center"> Motion Name </td>
    <td align="center"> Checkpoint Link </td>
    <td align="center"> Training data </td>
</tr>
<tr>
    <td align="center">Birds fly</td>
    <td align="center"><a href="https://pan.baidu.com/s/1nuZVRj-xRqkHySQQ3jCFkw">Baidu Disk (pwd: jj0o)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/10fi8KoBrGJMpLQKhUIaFSQ">Baidu Disk (pwd: w96b)</a></td>
</tr>
<tr>
    <td align="center">Firework</td>
    <td align="center"><a href="https://pan.baidu.com/s/1zJnn5bZpGzChRHJdO9x6WA">Baidu Disk (pwd: wj1p)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1uIyw0Q70svWNM5z7DFYkiQ">Baidu Disk (pwd: oamp)</a></td>
</tr>
<tr>
    <td align="center">Helicopter</td>
    <td align="center"><a href="https://pan.baidu.com/s/1oj6t_VFo9cX0vTZWDq8q3w">Baidu Disk (pwd: egpe)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1MYMjIFyFTiLGEX1w0ees2Q">Baidu Disk (pwd: t4ba)</a></td>
</tr>
<tr>
    <td align="center">Horse run</td>
    <td align="center"><a href="https://pan.baidu.com/s/1lkAFZuEnot4JGruLe6pR3g">Baidu Disk (pwd: 19ld)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1z7FHN-aotdOF2MPUk4lDJg">Baidu Disk (pwd: mte7)</a></td>
</tr>
<tr>
    <td align="center">Play the guitar</td>
    <td align="center"><a href="https://pan.baidu.com/s/1uY47E08_cUofmlmKWfi46A">Baidu Disk (pwd: l4dw)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1cemrtzJtS_Lm8y8nZM9kSw">Baidu Disk (pwd: js26)</a></td>
</tr>
<tr>
    <td align="center">Rain</td>
    <td align="center"><a href="https://pan.baidu.com/s/1Cvsyg7Ld2O0DEK_U__2aXg">Baidu Disk (pwd: jomu)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1hMGrHCLNRDLJQ-4XKk6hZg">Baidu Disk (pwd: 31ug)</a></td>
</tr>
<tr>
    <td align="center">Turn to smile</td>
    <td align="center"><a href="https://pan.baidu.com/s/1UYjWncrxYiAhwpNAafH5WA">Baidu Disk (pwd: 2bkl)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1ErFSm6t-CtYBzsuzxi08dg">Baidu Disk (pwd: l984)</a></td>
</tr>
<tr>
    <td align="center">Waterfall</td>
    <td align="center"><a href="https://pan.baidu.com/s/1tWArxOw6CMceaW_49rIoSA">Baidu Disk (pwd: vpkk)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1hjlqRwa35nZ2pc2D-gIX9A">Baidu Disk (pwd: 2edp)</a></td>
</tr>
<tr>
    <td align="center">All</td>
    <td align="center"><a href="https://pan.baidu.com/s/1vRG7kMCTC7b9YUd4qsSP_A">Baidu Disk (pwd: ifsm)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1h5HrIGWP5OlMqp9gkD9cyQ">Baidu Disk (pwd: 2i2k)</a></td>
</tr>
</table>

## Get Started
### 1. Training
```bash
# Training code to learn a motion pattern
CUDA_VISIBLE_DEVICES=X accelerate launch train_lamp.py config="configs/horse-run.yaml"

# Training code for video editing (The training video can be found in assets/run.mp4)
CUDA_VISIBLE_DEVICES=X accelerate launch train_lamp.py config="configs/run.yaml"
```

### 2. Inference
Here is an example command for inference
```bash
# Motion Pattern
python inference_script.py --weight ./my_weight/turn_to_smile/unet --pretrain_weight ./checkpoints/stable-diffusion-v1-4 --first_frame_path ./benchmark/turn_to_smile/head_photo_of_a_cute_girl,_comic_style.png --prompt "head photo of a cute girl, comic style, turns to smile"

# Video Editing
python inference_script.py --weight ./outputs/run/unet --pretrain_weight ./checkpoints/stable-diffusion-v1-4 --first_frame_path ./bemchmark/editing/a_girl_runs_beside_a_river,_Van_Gogh_style.png --length 24 --editing

#########################################################################################################
# --weight:           the path of our model
# --pretrain_weight:  the path of the pre-trained model (e.g. SDv1.4)
# --first_frame_path: the path of the first frame generated by T2I model (e.g. SD-XL)
# --prompt:           the input prompt, the default value is aligned with the filename of the first frame
# --output:           output path, default: ./results 
# --height:           video height, default: 320
# --width:            video width, default: 512
# --length            video length, default: 16
# --cfg:              classifier-free guidance, default: 12.5
#########################################################################################################
```
## Visual Examples
### Few-Shot-Based Text-to-Video Generation
<table class="center">
    <tr>
        <td align="center" style="width: 7%"> Birds fly</td>
        <td align="center">
            <img src="assets/inference-birds_fly_in_the_pink_sky.gif">
        </td>
        <td align="center">
            <img src="assets/inference-birds_fly_in_the_sky,_over_the_sea.gif">
        </td>
        <td align="center">
            <img src="assets/inference-many_birds_fly_over_a_plaza.gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"></td>
        <td align="center">Birds fly in the pink sky.</td>
        <td align="center">Birds fly in the sky, over the sea.</td>
        <td align="center">Many Birds fly over a plaza.</td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Horse run</td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_in_the_universe (1).gif">
        </td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_on_the_Mars (3).gif">
        </td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_on_the_road (1).gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"> </td>
        <td align="center">A horse runs in the universe.</td>
        <td align="center">A horse runs on the Mars.</td>
        <td align="center">A horse runs on the road.</td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Firework</td>
        <td align="center">
            <img src="assets/inference-fireworks_in_desert_night.gif">
        </td>
        <td align="center">
            <img src="assets/inference-fireworks_over_the_mountains (1).gif">
        </td>
        <td align="center">
            <img src="assets/inference-fireworks_in_the_night_city.gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"> </td>
        <td align="center">Fireworks in desert night.</td>
        <td align="center">Fireworks over the mountains.</td>
        <td align="center">Fireworks in the night city.</td>
    </tr>
</table>
