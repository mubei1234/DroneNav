<<<<<<< HEAD
# DroneNav: Unified Text-Visual Representation and Structured Spatial Reasoning for Robust UAV Vision-and-Language Navigation

## First Stage
The project is running on Ubuntu 22.04
All code in this work was implemented with ​​PyTorch 2.2.2​​, ​​CUDA 11.8​​, and ​​Python 3.10.16​​.First, you need to create and activate a virtual environment:

```bash
conda create -n dronenav python=3.10.16  
conda activate dronenav
```

Next, install the specified versions of PyTorch and CUDA:
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install dependencies related to Mobile SAM:
```bash
conda install mpi4py
pip install git+https://github.com/water-cookie/Segment-Everything-Everywhere-All-At-Once.git@package
pip install git+https://github.com/water-cookie/Semantic-SAM.git@package
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Now install LLaVA:
```bash
pip install git+https://github.com/water-cookie/LLaVA.git
```

Install Grounding DINO:
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Install Mobile SAM:
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

Download the model weights:
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip

put them into weights floder.


Download the datasets:
https://www.dropbox.com/scl/fi/ekbogjn2ptxdde2gik6nx/data.tar.gz?rlkey=oq5smcqlbgc6do5mcowetj3mp&st=gx563bhw&dl=0
https://github.com/QingyongHu/SensatUrban?tab=readme-ov-file#4-training-and-evaluation

Run the following commands to process the dataset:
```bash
sh navfw/tools/scripts/rasterize.sh path_to_ply_dir/train
sh navfw/tools/scripts/rasterize.sh path_to_ply_dir/test
```
put them into data floder.


You can the use following script to evaluate model.
```bash
sh navfw/tools/scripts/eval.sh
```

## Second Stage

Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
```

Install additional packages for training cases
```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Using "target_rgb" to obtain training images data. Then using the following script to lora fine-tuning model. 

```bash
sh LLaVA-1.6-ft-main/scripts/v1_6/finetune_lora_llava_mistral.sh
```
=======
# DroneNav
>>>>>>> 09d75a2811dfb1985223cc62c48292d0b8f8ebf8
