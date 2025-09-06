# 3T3D - A Vision Transformer Based 2D to 3D Model
## Overview
Our final project for 11-685, Introduction to Deep Learning was a long, arduous one, but a huge learning opportunity. I was extremely fortunate to be on a team with 3 very passionate and patient folks (Chia, David, Karthick) for whom I'm forever indebted for their long hours and thoughtful feedback.

We had a relatively simple vision: If given 3 orthographic napkin sketches as input, could a 2D to 3D model produce a detailed 3D model of a potential piece of architecture based on this input? How could we build such a model?

### Our Model
It turns out, this is possible! SOTA Vision Transformer architectures allow for a rich and positionally grounded 2D image to provide information for something spatial, like a 3D model. As a base model, we turned to [DINOv2](https://arxiv.org/abs/2304.07193) for its multi-view embeddings. Essentially we're able to break the input images into 'patches' and then use the patches as tokens to feed to our encoder (DINOv2), combine the resulting embeddings, then decode them using a custom transformer architecture.

## Architecture Overview
<!--add diagrams of the architecture here-->
Our inputs are the three orthographics sketches representing top, side, and front views of the building (just like floorplan and elevation). We wanted the designer to have full control of the final 3D form, which is why we chose to have this many inputs. There are essentially four stages of this architecture:
    1. Input Processing
    2. Encoder (DINOv2) - Extract Patches
    2. Multi-view Fusion (Combine Patches)
    3. Decoder (Custom Transformer)
    4. Progressive Upsampling to Output

### Input 
We start off by feeding the 3 input views into the DINOv2 Encoder to extract the patch embeddings. DINOv2 includes a classification `cls` token in its embeddings however we remove this as we don't have any classes in our training data. DINO produces vectors of size `[B, H, W]` however we will resize these to `[B, C, H, W]`and rescale them from `256x256` to `512x512`. Once this is done we are ready to move to the next step.

### Fusion
We call this step in the process 'Fusion' since we're sort of fusing everything together. The idea is that each patch embedding produced by the vision transformer will correspond to the same space in 3D, therefore we can sum them all together into a single embedding vector. This single vector will then get passed to the decoder to be turned into the output triplanes.

### Decoder
This is where we had to do the most engineering ~ we opted to use a custom transformer structure that consists of 6 decoder layers, each with 8 attention heads performing self-attention. This provided us with enough of a balance between total runtime, model size, and output quality.

### Upsampling
We continuously upscale the decoded output from :math:{\mathbb{R}^16 \rightarrow \mathbb{R}^128}.

## Training
Our objective is to minimize the L1 loss between the predicted triplane output features for the given object and the ground truth triplane features. We also adopted a bit of an interesting training strategy in order to get the most out of DINOv2. 

This consisted of a two stage training process whereby we begin by first freezing the encoder, and training the decoder only. Once we achieve good predictions from the decoder, we then set about unfreezing the entire model and fine-tuning with differentiated learning rates.  

