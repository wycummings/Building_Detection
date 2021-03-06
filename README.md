# Building Detection
The following scripts were used for semantic segmentation of buildings using aerial images . The data set was 100 aerial images over several Japanese cities. 

## Required Libraries 
torchvision, torch, Numpy, OpenCV and PIL.

## Learning
- train.py
  - The model used was NestedUnet.
  - NestedUnet was used from the following paper https://arxiv.org/abs/1807.10165

## Implementation
- Use the trained model in apply.py.
- Stride was used when inputting images to implement an overlap.
  - When stride is 256 while the patch size is 256, there would be no overlap
  - When stride is 228 while the patch size is 256, half of the input image would be used in the next input.
- To use tta when inputting images, set --augument to True


## Results
The following data is for models trained in 20epoch. 

Patch Size | Stride | tta | Mean IoU Score
--- | --- | --- | ---
448 | 448 | no | 48.24
448 | 224 | no | 49.73
448 | 112 | no | 50.37
448 | 112 | yes | 45.48
448 | 90 | no | 51.02
448 | 90 | yes | 48.77
**448** | **75** | **no** | **51.27**
448 | 64 | no | 51.02

The use of tta was proven to lower the score

After adjusting the stride, 1/6 of the patch size yielded the highest score.  

The GPU capacity was 8GB.
The largest patch size runnable on the GPU was 704. 704 patch size returned the following results. 

Patch Size | Stride | Mean IoU Score
--- | --- | ---
704 | 704 | 48.7
704 | 140 | 50.87
**704** | **117** | **51.02**

For a patch size of 704, a stride of 1/6th of the patch size produced the best results.  

Models trained under 100epochs generated higher scores.

Patch Size | Stride | Mean IoU Score
--- | --- | --- 
704 | 117 | 56.27
448 | 75 | 56.70
**224** | **37** | **56.81**
