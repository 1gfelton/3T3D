# 3T3D - A Vision Transformer Based 2D to 3D Model
## Overview
Our final project for 11-685, Introduction to Deep Learning was a long, arduous one, but a huge learning opportunity. I was extremely fortunate to be on a team with 3 very passionate and patient folks (Chia, David, Karthick) for whom I'm forever indebted for their long hours and thoughtful feedback.

We had a relatively simple vision: If given 3 orthographic napkin sketches as input, could a 2D to 3D model produce a detailed 3D model of a potential piece of architecture based on this input? How could we build such a model?

### Our Model
It turns out, this is possible! SOTA Vision Transformer architectures allow for a rich and positionally grounded 2D image to provide information for something spatial, like a 3D model. As a base model, we turned to [DINOv2](https://arxiv.org/abs/2304.07193) for its multi-view embeddings. Essentially we're able to break the input images into 'patches' and then use the patches as tokens to feed to our encoder (DINOv2), combine the resulting embeddings, then decode them using a custom transformer architecture.

## Architecture Overview
![Diagram of the model architecture](/img/arch_diagram.jpg)

Our inputs are the three orthographics sketches representing top, side, and front views of the building (just like floorplan and elevation). We wanted the designer to have full control of the final 3D form, which is why we chose to have this many inputs. There are essentially four stages of this architecture:
1. Input Processing
2. Encoder (DINOv2) - Extract Patches
2. Multi-view Fusion (Combine Patches)
3. Decoder (Custom Transformer)
4. Progressive Upsampling to Output

### Input 
We start off by feeding the 3 input views into the DINOv2 Encoder to extract the patch embeddings. DINOv2 includes a classification `cls` token in its embeddings however we remove this as we don't have any classes in our training data. DINO produces vectors of size `[B, H, W]` however we will concatenate our three views into tensors of shape `[B, C, H, W]`and rescale them from `256x256` to `512x512`. Once this is done we are ready to move to the next step.

### Fusion
![How fusion works](/img/fusion_diagram.jpg)

We call this step in the process 'Fusion' since we're sort of fusing everything together. The idea is that each patch embedding produced by the vision transformer will correspond to the same space in 3D, therefore we can sum them all together into a single embedding vector. This single vector will then get passed to the decoder to be turned into the output triplanes.

### Decoder
This is where we had to do the most engineering ~ we opted to use a custom transformer structure that consists of 6 decoder layers, each with 8 attention heads performing self-attention. This provided us with enough of a balance between total runtime, model size, and output quality.
<div align="center">
    <img src="https://github.com/1gfelton/3T3D/blob/main/img/decoder_diagram.jpg" width="200">
</div>

### Upsampling
We continuously upscale the decoded output from ![equation](https://latex.codecogs.com/svg.image?\mathbb{R}^{16}\rightarrow\mathbb{R}^{128})

## Training
### Custom Dataset
![How we created the dataset for our trainign](/img/dataset_creation1.jpg)

During our literature review, we were unable to find a dataset that was 'good enough' for our intent of producing designs of a quality suitable for architectural design. We needed something new, ultimately opting for a custom pipeline that allowed us to produce thousands of images of architectural renderings that we would then use to generate thousands of 3D models using [TripoSR](https://github.com/VAST-AI-Research/TripoSR). Once the 3D models were generated, for each model we generate views of the model from the top, front, and side, and convert these to sketchy images using [Informative Drawings](https://github.com/carolineec/informative-drawings).

### Training Objective
Our objective is to minimize the L1 loss between the predicted triplane output features for the given object and the ground truth triplane features. We also adopted a bit of an interesting training strategy in order to get the most out of DINOv2. 

This consisted of a two stage training process whereby we begin by first freezing the encoder, and training the decoder only. Once we achieve good predictions from the decoder, we then set about unfreezing the entire model and fine-tuning with differentiated learning rates.  

### Results
Here we show the validation/training loss from our final training run. As shown, the model is able to slowly learn the triplane representations, albeit quite slowly. We trained for 37 epochs but believe that with greater optimization when it comes to lr scheduler and hyperparameters, we could beat our previous score and produce even better outputs. 
![Wandb training results](/img/val_train_loss.png)

We trained on an A100 for 5 hours, giving us 37 total epochs. It was `480s` per epoch on average.

Here's an example output from our model:

<div align="center">
    <img src="https://github.com/1gfelton/3T3D/blob/main/img/comparison.jpg" width="500">
</div>

# Running our Code
## Setup
We recommend running our notebook in google colab, as we had done much of our experimentation this way.
You will need to update the following directories to point to the location of your dataset:

```
TRAIN_IMAGE_DIR_V1 = "/your/path/to/sketch/front"
TRAIN_IMAGE_DIR_V2 = "/your/path/to/sketch/right" 
TRAIN_IMAGE_DIR_V3 = "/your/path/to/sketch/top"
TRAIN_MESH_DIR = "/your/path/to/3dmodel"

TEST_IMAGE_PATH_V1 = "/your/path/to/sketch/front/test_image.png"
```

## Data
Download our dataset from [here](https://drive.google.com/drive/folders/1jQuu2hA1_R0IRaaHouJ5B9rVDO61THqD?usp=drive_link).

```
Dataset/
├── sketch/
│   ├── front/     # Front view sketches (.png, .jpg)
│   ├── right/     # Right view sketches  
│   └── top/       # Top view sketches
└── 3dmodel/       # 3D meshes (.obj files)
```
