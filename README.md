# Image_Segmentation (VGG+FCN8 and UNet)
1. **VGG16 + FCN8** in the folder `VFF_FCN8`. <br>
This work uses `VGG16` from scratch as a `encoder` and `FCN8` as `decoder` to build a image segmentation model. The diagram of the network shown as,
<p float="left">
  <img src='https://drive.google.com/uc?export=view&id=1lrqB4YegV8jXWNfyYAaeuFlwXIc54aRP' alt='fcn-8' width="1200" height="400">
</p>

The dataset uses [dataset1](https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing) prepared by [divamgupta](https://github.com/divamgupta/image-segmentation-keras).<br>

The dataset contains folders for images and annotations. The *images* contain the video frames while the *annotations* contain the pixel-wise label maps. Each label map has the shape `(height, width , 1)` with each point in this space denoting the corresponding pixel's class. Classes are in the range `[0, 11]` (i.e. 12 classes) and the pixel labels correspond to these classes<br>
`data_processing.py` contains functions of pre-processing dataset<br>
`image_processing.py` includes functions of image visulization.


The `VGG16` weights can be downloaded from [VGG16_weight](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

Here is the result after fitting the model with 170 epochs receiving 0.85 accuracy(IOU and Dice were calculated to assess the performance):
<p float="left">
  <img src='./VGG_FCN8/result.jpg' width="800" height="400"/>
</p>

2. **UNet** in the folder `UNet`.
The model will use the dataset [Oxford Pets - IIT dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which is included in `TensorFlow Datasets` already. The UNet diagram is shown as, 
<img src='https://drive.google.com/uc?export=view&id=1BeQSKL2Eq6Fw9iRXsN1hgunY-CS2nH7V' alt='unet' width="1200" height="500">
The validation accuracy after 20 epochs reaches 0.89, and one of the resulting examples is shown as,<br>
<p float="left">
  <img src='./UNet/result.jpg' width="800" height="400"/>
</p>

