## Detecting Diabetic Retinopathy With Deep Learning

## Objective

Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. The condition is estimated to affect over 93 million people.

The need for a comprehensive and automated method of diabetic retinopathy screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With photos of eyes as input, the goal of this capstone is to create a new model, ideally resulting in realistic clinical potential.

The motivations for this project are twofold:

* Image classification has been a personal interest for years, in addition to classification
on a large scale data set.

* Time is lost between patients getting their eyes scanned (shown below), having their images analyzed by doctors, and scheduling a follow-up appointment. By processing images in real-time, EyeNet would allow people to seek & schedule treatment the same day.




## Table of Contents
1. [Data](#data)
2. [Extraction and Preprocessing](#preprocessing)
    * [Download Images to Google Colab](#download-all-images-to-colab)
    * [Resize Images](#crop-and-resize-all-images)
    * [Checking Blurness of Images](#Checking-Blur)
    * [Data Augmentation](#Data-Augmentation)
3. [CNN Architecture](#neural-network-architecture)
5. [Results](#results)
7. [References](#references)
8. [Authors](#Authors)

## Data

The data originates from a [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). However, is an atypical Kaggle dataset. In most Kaggle competitions, the data has already been cleaned, giving the data scientist very little to preprocess. With this dataset, this isn't the case.

All images are taken of different people, using different cameras, and of different sizes. Pertaining to the [preprocessing](#preprocessing) section, this data is extremely noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

The training data is comprised of 35,126 images, which are augmented during preprocessing.

### Prerequisites

You'll need to install:

* [Anaconda](https://www.continuum.io/downloads)
* [Python (Minimum 3)](https://www.continuum.io/blog/developer-blog/python-3-support-anaconda)
* [pandas](http://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Fast.ai](https://seaborn.pydata.org/)


## Extraction and Preprocessing

The preprocessing pipeline is the following:

### Download Images to Google Colab
Google Colab was used as a platform for this dataset.


### Resize All Images
The given images were mostly very large. ~2000 by ~2000. So to keep things unifrom all images were scaled down to 512 by 512.

### Checking Blurness of Images

The method is simple. Straightforward. Has sound reasoning. And can be implemented in only a single line of code:

`cv2.Laplacian(image, cv2.CV_64F).var()`

We simply take a single channel of an image and convolve it with the following 3 x 3 kernel:

| 0 	|  1 	| 0 	|
|:-:	|:--:	|:-:	|
| 1 	| -4 	| 1 	|
| 0 	|  1 	| 0 	|

And then take the variance (i.e. standard deviation squared) of the response.
If the variance falls below a pre-defined threshold, then the image is considered blurry; otherwise, the image is not blurry. Here is the paper with talks about it's implementation, Ariation of the Laplacian by Pech-Pacheco et al. in their 2000 ICPR paper, [Diatom autofocusing in brightfield microscopy: a comparative study](http://optica.csic.es/papers/icpr2k.pdf).

### Data Augmentation
All images were rotated and mirrored.Images without retinopathy were mirrored;
images that had retinopathy were mirrored, and rotated 90, 120, 180, and 270
degrees.

The first images show two pairs of eyes, along with the black borders. Notice in
the cropping and rotations how the majority of noise is removed.



## Neural Network Architecture

| Layer (type)             	| Output Shape        	| Param # 	|
|--------------------------	|---------------------	|---------	|
| Conv2d-1                 	| [-1, 64, 256, 256]  	| 9,408   	|
| BatchNorm2d-2            	| [-1, 64, 256, 256]  	| 128     	|
| ReLU-3                   	| [-1, 64, 256, 256]  	| 0       	|
| MaxPool2d-4              	| [-1, 64, 128, 128]  	| 0       	|
| BatchNorm2d-5            	| [-1, 64, 128, 128]  	| 128     	|
| ReLU-6                   	| [-1, 64, 128, 128]  	| 0       	|
| Conv2d-7                 	| [-1, 128, 128, 128] 	| 8192    	|
| BatchNorm2d-8            	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-9                   	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-10                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-11           	| [-1, 96, 128, 128]  	| 192     	|
| ReLU-12                  	| [-1, 96, 128, 128]  	| 0       	|
| Conv2d-13                	| [-1, 128, 128, 128] 	| 12288   	|
| BatchNorm2d-14           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-15                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-16                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-17           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-18                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-19                	| [-1, 128, 128, 128] 	| 16384   	|
| BatchNorm2d-20           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-21                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-22                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-23           	| [-1, 160, 128, 128] 	| 320     	|
| ReLU-24                  	| [-1, 160, 128, 128] 	| 0       	|
| Conv2d-25                	| [-1, 128, 128, 128] 	| 20480   	|
| BatchNorm2d-26           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-27                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-28                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-29           	| [-1, 192, 128, 128] 	| 384     	|
| ReLU-30                  	| [-1, 192, 128, 128] 	| 0       	|
| Conv2d-31                	| [-1, 128, 128, 128] 	| 24576   	|
| BatchNorm2d-32           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-33                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-34                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-35           	| [-1, 224, 128, 128] 	| 448     	|
| ReLU-36                  	| [-1, 224, 128, 128] 	| 0       	|
| Conv2d-37                	| [-1, 128, 128, 128] 	| 28672   	|
| BatchNorm2d-38           	| [-1, 128, 128, 128] 	| 256     	|
| ReLU-39                  	| [-1, 128, 128, 128] 	| 0       	|
| Conv2d-40                	| [-1, 32, 128, 128]  	| 36864   	|
| BatchNorm2d-41           	| [-1, 256, 128, 128] 	| 512     	|
| ReLU-42                  	| [-1, 256, 128, 128] 	| 0       	|
| Conv2d-43                	| [-1, 128, 128, 128] 	| 32768   	|
| AvgPool2d-44             	| [-1, 128, 64, 64]   	| 0       	|
| BatchNorm2d-45           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-46                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-47                	| [-1, 128, 64, 64]   	| 16384   	|
| BatchNorm2d-48           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-49                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-50                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-51           	| [-1, 160, 64, 64]   	| 320     	|
| ReLU-52                  	| [-1, 160, 64, 64]   	| 0       	|
| Conv2d-53                	| [-1, 128, 64, 64]   	| 20480   	|
| BatchNorm2d-54           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-55                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-56                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-57           	| [-1, 192, 64, 64]   	| 384     	|
| ReLU-58                  	| [-1, 192, 64, 64]   	| 0       	|
| Conv2d-59                	| [-1, 128, 64, 64]   	| 24576   	|
| BatchNorm2d-60           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-61                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-62                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-63           	| [-1, 224, 64, 64]   	| 448     	|
| ReLU-64                  	| [-1, 224, 64, 64]   	| 0       	|
| Conv2d-65                	| [-1, 128, 64, 64]   	| 28672   	|
| BatchNorm2d-66           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-67                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-68                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-69           	| [-1, 256, 64, 64]   	| 512     	|
| ReLU-70                  	| [-1, 256, 64, 64]   	| 0       	|
| Conv2d-71                	| [-1, 128, 64, 64]   	| 32768   	|
| BatchNorm2d-72           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-73                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-74                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-75           	| [-1, 288, 64, 64]   	| 576     	|
| ReLU-76                  	| [-1, 288, 64, 64]   	| 0       	|
| Conv2d-77                	| [-1, 128, 64, 64]   	| 36864   	|
| BatchNorm2d-78           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-79                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-80                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-81           	| [-1, 320, 64, 64]   	| 640     	|
| ReLU-82                  	| [-1, 320, 64, 64]   	| 0       	|
| Conv2d-83                	| [-1, 128, 64, 64]   	| 40960   	|
| BatchNorm2d-84           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-85                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-86                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-87           	| [-1, 352, 64, 64]   	| 704     	|
| ReLU-88                  	| [-1, 352, 64, 64]   	| 0       	|
| Conv2d-89                	| [-1, 128, 64, 64]   	| 45056   	|
| BatchNorm2d-90           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-91                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-92                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-93           	| [-1, 384, 64, 64]   	| 768     	|
| ReLU-94                  	| [-1, 384, 64, 64]   	| 0       	|
| Conv2d-95                	| [-1, 128, 64, 64]   	| 49152   	|
| BatchNorm2d-96           	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-97                  	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-98                	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-99           	| [-1, 416, 64, 64]   	| 832     	|
| ReLU-100                 	| [-1, 416, 64, 64]   	| 0       	|
| Conv2d-101               	| [-1, 128, 64, 64]   	| 53248   	|
| BatchNorm2d-102          	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-103                 	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-104               	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-105          	| [-1, 448, 64, 64]   	| 896     	|
| ReLU-106                 	| [-1, 448, 64, 64]   	| 0       	|
| Conv2d-107               	| [-1, 128, 64, 64]   	| 57344   	|
| BatchNorm2d-108          	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-109                 	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-110               	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-111          	| [-1, 480, 64, 64]   	| 960     	|
| ReLU-112                 	| [-1, 480, 64, 64]   	| 0       	|
| Conv2d-113               	| [-1, 128, 64, 64]   	| 61440   	|
| BatchNorm2d-114          	| [-1, 128, 64, 64]   	| 256     	|
| ReLU-115                 	| [-1, 128, 64, 64]   	| 0       	|
| Conv2d-116               	| [-1, 32, 64, 64]    	| 36864   	|
| BatchNorm2d-117          	| [-1, 512, 64, 64]   	| 1024    	|
| ReLU-118                 	| [-1, 512, 64, 64]   	| 0       	|
| Conv2d-119               	| [-1, 256, 64, 64]   	| 131072  	|
| AvgPool2d-120            	| [-1, 256, 32, 32]   	| 0       	|
| BatchNorm2d-121          	| [-1, 256, 32, 32]   	| 512     	|
| ReLU-122                 	| [-1, 256, 32, 32]   	| 0       	|
| Conv2d-123               	| [-1, 128, 32, 32]   	| 32768   	|
| BatchNorm2d-124          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-125                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-126               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-127          	| [-1, 288, 32, 32]   	| 576     	|
| ReLU-128                 	| [-1, 288, 32, 32]   	| 0       	|
| Conv2d-129               	| [-1, 128, 32, 32]   	| 36864   	|
| BatchNorm2d-130          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-131                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-132               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-133          	| [-1, 320, 32, 32]   	| 640     	|
| ReLU-134                 	| [-1, 320, 32, 32]   	| 0       	|
| Conv2d-135               	| [-1, 128, 32, 32]   	| 40960   	|
| BatchNorm2d-136          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-137                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-138               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-139          	| [-1, 352, 32, 32]   	| 704     	|
| ReLU-140                 	| [-1, 352, 32, 32]   	| 0       	|
| Conv2d-141               	| [-1, 128, 32, 32]   	| 45056   	|
| BatchNorm2d-142          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-143                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-144               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-145          	| [-1, 384, 32, 32]   	| 768     	|
| ReLU-146                 	| [-1, 384, 32, 32]   	| 0       	|
| Conv2d-147               	| [-1, 128, 32, 32]   	| 49152   	|
| BatchNorm2d-148          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-149                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-150               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-151          	| [-1, 416, 32, 32]   	| 832     	|
| ReLU-152                 	| [-1, 416, 32, 32]   	| 0       	|
| Conv2d-153               	| [-1, 128, 32, 32]   	| 53248   	|
| BatchNorm2d-154          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-155                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-156               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-157          	| [-1, 448, 32, 32]   	| 896     	|
| ReLU-158                 	| [-1, 448, 32, 32]   	| 0       	|
| Conv2d-159               	| [-1, 128, 32, 32]   	| 57344   	|
| BatchNorm2d-160          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-161                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-162               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-163          	| [-1, 480, 32, 32]   	| 960     	|
| ReLU-164                 	| [-1, 480, 32, 32]   	| 0       	|
| Conv2d-165               	| [-1, 128, 32, 32]   	| 61440   	|
| BatchNorm2d-166          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-167                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-168               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-169          	| [-1, 512, 32, 32]   	| 1024    	|
| ReLU-170                 	| [-1, 512, 32, 32]   	| 0       	|
| Conv2d-171               	| [-1, 128, 32, 32]   	| 65536   	|
| BatchNorm2d-172          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-173                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-174               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-175          	| [-1, 544, 32, 32]   	| 1088    	|
| ReLU-176                 	| [-1, 544, 32, 32]   	| 0       	|
| Conv2d-177               	| [-1, 128, 32, 32]   	| 69632   	|
| BatchNorm2d-178          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-179                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-180               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-181          	| [-1, 576, 32, 32]   	| 1152    	|
| ReLU-182                 	| [-1, 576, 32, 32]   	| 0       	|
| Conv2d-183               	| [-1, 128, 32, 32]   	| 73728   	|
| BatchNorm2d-184          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-185                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-186               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-187          	| [-1, 608, 32, 32]   	| 1216    	|
| ReLU-188                 	| [-1, 608, 32, 32]   	| 0       	|
| Conv2d-189               	| [-1, 128, 32, 32]   	| 77824   	|
| BatchNorm2d-190          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-191                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-192               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-193          	| [-1, 640, 32, 32]   	| 1280    	|
| ReLU-194                 	| [-1, 640, 32, 32]   	| 0       	|
| Conv2d-195               	| [-1, 128, 32, 32]   	| 81920   	|
| BatchNorm2d-196          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-197                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-198               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-199          	| [-1, 672, 32, 32]   	| 1344    	|
| ReLU-200                 	| [-1, 672, 32, 32]   	| 0       	|
| Conv2d-201               	| [-1, 128, 32, 32]   	| 86016   	|
| BatchNorm2d-202          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-203                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-204               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-205          	| [-1, 704, 32, 32]   	| 1408    	|
| ReLU-206                 	| [-1, 704, 32, 32]   	| 0       	|
| Conv2d-207               	| [-1, 128, 32, 32]   	| 90112   	|
| BatchNorm2d-208          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-209                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-210               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-211          	| [-1, 736, 32, 32]   	| 1472    	|
| ReLU-212                 	| [-1, 736, 32, 32]   	| 0       	|
| Conv2d-213               	| [-1, 128, 32, 32]   	| 94208   	|
| BatchNorm2d-214          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-215                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-216               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-217          	| [-1, 768, 32, 32]   	| 1536    	|
| ReLU-218                 	| [-1, 768, 32, 32]   	| 0       	|
| Conv2d-219               	| [-1, 128, 32, 32]   	| 98304   	|
| BatchNorm2d-220          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-221                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-222               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-223          	| [-1, 800, 32, 32]   	| 1600    	|
| ReLU-224                 	| [-1, 800, 32, 32]   	| 0       	|
| Conv2d-225               	| [-1, 128, 32, 32]   	| 102400  	|
| BatchNorm2d-226          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-227                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-228               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-229          	| [-1, 832, 32, 32]   	| 1664    	|
| ReLU-230                 	| [-1, 832, 32, 32]   	| 0       	|
| Conv2d-231               	| [-1, 128, 32, 32]   	| 106496  	|
| BatchNorm2d-232          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-233                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-234               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-235          	| [-1, 864, 32, 32]   	| 1728    	|
| ReLU-236                 	| [-1, 864, 32, 32]   	| 0       	|
| Conv2d-237               	| [-1, 128, 32, 32]   	| 110592  	|
| BatchNorm2d-238          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-239                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-240               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-241          	| [-1, 896, 32, 32]   	| 1792    	|
| ReLU-242                 	| [-1, 896, 32, 32]   	| 0       	|
| Conv2d-243               	| [-1, 128, 32, 32]   	| 114688  	|
| BatchNorm2d-244          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-245                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-246               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-247          	| [-1, 928, 32, 32]   	| 1856    	|
| ReLU-248                 	| [-1, 928, 32, 32]   	| 0       	|
| Conv2d-249               	| [-1, 128, 32, 32]   	| 118784  	|
| BatchNorm2d-250          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-251                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-252               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-253          	| [-1, 960, 32, 32]   	| 1920    	|
| ReLU-254                 	| [-1, 960, 32, 32]   	| 0       	|
| Conv2d-255               	| [-1, 128, 32, 32]   	| 122880  	|
| BatchNorm2d-256          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-257                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-258               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-259          	| [-1, 992, 32, 32]   	| 1984    	|
| ReLU-260                 	| [-1, 992, 32, 32]   	| 0       	|
| Conv2d-261               	| [-1, 128, 32, 32]   	| 126976  	|
| BatchNorm2d-262          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-263                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-264               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-265          	| [-1, 1024, 32, 32]  	| 2048    	|
| ReLU-266                 	| [-1, 1024, 32, 32]  	| 0       	|
| Conv2d-267               	| [-1, 128, 32, 32]   	| 131072  	|
| BatchNorm2d-268          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-269                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-270               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-271          	| [-1, 1056, 32, 32]  	| 2112    	|
| ReLU-272                 	| [-1, 1056, 32, 32]  	| 0       	|
| Conv2d-273               	| [-1, 128, 32, 32]   	| 135168  	|
| BatchNorm2d-274          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-275                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-276               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-277          	| [-1, 1088, 32, 32]  	| 2176    	|
| ReLU-278                 	| [-1, 1088, 32, 32]  	| 0       	|
| Conv2d-279               	| [-1, 128, 32, 32]   	| 139264  	|
| BatchNorm2d-280          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-281                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-282               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-283          	| [-1, 1120, 32, 32]  	| 2240    	|
| ReLU-284                 	| [-1, 1120, 32, 32]  	| 0       	|
| Conv2d-285               	| [-1, 128, 32, 32]   	| 143360  	|
| BatchNorm2d-286          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-287                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-288               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-289          	| [-1, 1152, 32, 32]  	| 2304    	|
| ReLU-290                 	| [-1, 1152, 32, 32]  	| 0       	|
| Conv2d-291               	| [-1, 128, 32, 32]   	| 147456  	|
| BatchNorm2d-292          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-293                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-294               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-295          	| [-1, 1184, 32, 32]  	| 2368    	|
| ReLU-296                 	| [-1, 1184, 32, 32]  	| 0       	|
| Conv2d-297               	| [-1, 128, 32, 32]   	| 151552  	|
| BatchNorm2d-298          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-299                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-300               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-301          	| [-1, 1216, 32, 32]  	| 2432    	|
| ReLU-302                 	| [-1, 1216, 32, 32]  	| 0       	|
| Conv2d-303               	| [-1, 128, 32, 32]   	| 155648  	|
| BatchNorm2d-304          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-305                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-306               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-307          	| [-1, 1248, 32, 32]  	| 2496    	|
| ReLU-308                 	| [-1, 1248, 32, 32]  	| 0       	|
| Conv2d-309               	| [-1, 128, 32, 32]   	| 159744  	|
| BatchNorm2d-310          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-311                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-312               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-313          	| [-1, 1280, 32, 32]  	| 2560    	|
| ReLU-314                 	| [-1, 1280, 32, 32]  	| 0       	|
| Conv2d-315               	| [-1, 128, 32, 32]   	| 163840  	|
| BatchNorm2d-316          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-317                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-318               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-319          	| [-1, 1312, 32, 32]  	| 2624    	|
| ReLU-320                 	| [-1, 1312, 32, 32]  	| 0       	|
| Conv2d-321               	| [-1, 128, 32, 32]   	| 167936  	|
| BatchNorm2d-322          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-323                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-324               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-325          	| [-1, 1344, 32, 32]  	| 2688    	|
| ReLU-326                 	| [-1, 1344, 32, 32]  	| 0       	|
| Conv2d-327               	| [-1, 128, 32, 32]   	| 172032  	|
| BatchNorm2d-328          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-329                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-330               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-331          	| [-1, 1376, 32, 32]  	| 2752    	|
| ReLU-332                 	| [-1, 1376, 32, 32]  	| 0       	|
| Conv2d-333               	| [-1, 128, 32, 32]   	| 176128  	|
| BatchNorm2d-334          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-335                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-336               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-337          	| [-1, 1408, 32, 32]  	| 2816    	|
| ReLU-338                 	| [-1, 1408, 32, 32]  	| 0       	|
| Conv2d-339               	| [-1, 128, 32, 32]   	| 180224  	|
| BatchNorm2d-340          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-341                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-342               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-343          	| [-1, 1440, 32, 32]  	| 2880    	|
| ReLU-344                 	| [-1, 1440, 32, 32]  	| 0       	|
| Conv2d-345               	| [-1, 128, 32, 32]   	| 184320  	|
| BatchNorm2d-346          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-347                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-348               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-349          	| [-1, 1472, 32, 32]  	| 2944    	|
| ReLU-350                 	| [-1, 1472, 32, 32]  	| 0       	|
| Conv2d-351               	| [-1, 128, 32, 32]   	| 188416  	|
| BatchNorm2d-352          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-353                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-354               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-355          	| [-1, 1504, 32, 32]  	| 3008    	|
| ReLU-356                 	| [-1, 1504, 32, 32]  	| 0       	|
| Conv2d-357               	| [-1, 128, 32, 32]   	| 192512  	|
| BatchNorm2d-358          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-359                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-360               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-361          	| [-1, 1536, 32, 32]  	| 3072    	|
| ReLU-362                 	| [-1, 1536, 32, 32]  	| 0       	|
| Conv2d-363               	| [-1, 128, 32, 32]   	| 196608  	|
| BatchNorm2d-364          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-365                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-366               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-367          	| [-1, 1568, 32, 32]  	| 3136    	|
| ReLU-368                 	| [-1, 1568, 32, 32]  	| 0       	|
| Conv2d-369               	| [-1, 128, 32, 32]   	| 200704  	|
| BatchNorm2d-370          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-371                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-372               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-373          	| [-1, 1600, 32, 32]  	| 3200    	|
| ReLU-374                 	| [-1, 1600, 32, 32]  	| 0       	|
| Conv2d-375               	| [-1, 128, 32, 32]   	| 204800  	|
| BatchNorm2d-376          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-377                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-378               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-379          	| [-1, 1632, 32, 32]  	| 3264    	|
| ReLU-380                 	| [-1, 1632, 32, 32]  	| 0       	|
| Conv2d-381               	| [-1, 128, 32, 32]   	| 208896  	|
| BatchNorm2d-382          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-383                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-384               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-385          	| [-1, 1664, 32, 32]  	| 3328    	|
| ReLU-386                 	| [-1, 1664, 32, 32]  	| 0       	|
| Conv2d-387               	| [-1, 128, 32, 32]   	| 212992  	|
| BatchNorm2d-388          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-389                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-390               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-391          	| [-1, 1696, 32, 32]  	| 3392    	|
| ReLU-392                 	| [-1, 1696, 32, 32]  	| 0       	|
| Conv2d-393               	| [-1, 128, 32, 32]   	| 217088  	|
| BatchNorm2d-394          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-395                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-396               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-397          	| [-1, 1728, 32, 32]  	| 3456    	|
| ReLU-398                 	| [-1, 1728, 32, 32]  	| 0       	|
| Conv2d-399               	| [-1, 128, 32, 32]   	| 221184  	|
| BatchNorm2d-400          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-401                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-402               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-403          	| [-1, 1760, 32, 32]  	| 3520    	|
| ReLU-404                 	| [-1, 1760, 32, 32]  	| 0       	|
| Conv2d-405               	| [-1, 128, 32, 32]   	| 225280  	|
| BatchNorm2d-406          	| [-1, 128, 32, 32]   	| 256     	|
| ReLU-407                 	| [-1, 128, 32, 32]   	| 0       	|
| Conv2d-408               	| [-1, 32, 32, 32]    	| 36864   	|
| BatchNorm2d-409          	| [-1, 1792, 32, 32]  	| 3584    	|
| ReLU-410                 	| [-1, 1792, 32, 32]  	| 0       	|
| Conv2d-411               	| [-1, 896, 32, 32]   	| 1605632 	|
| AvgPool2d-412            	| [-1, 896, 16, 16]   	| 0       	|
| BatchNorm2d-413          	| [-1, 896, 16, 16]   	| 1792    	|
| ReLU-414                 	| [-1, 896, 16, 16]   	| 0       	|
| Conv2d-415               	| [-1, 128, 16, 16]   	| 114688  	|
| BatchNorm2d-416          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-417                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-418               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-419          	| [-1, 928, 16, 16]   	| 1856    	|
| ReLU-420                 	| [-1, 928, 16, 16]   	| 0       	|
| Conv2d-421               	| [-1, 128, 16, 16]   	| 118784  	|
| BatchNorm2d-422          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-423                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-424               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-425          	| [-1, 960, 16, 16]   	| 1920    	|
| ReLU-426                 	| [-1, 960, 16, 16]   	| 0       	|
| Conv2d-427               	| [-1, 128, 16, 16]   	| 122880  	|
| BatchNorm2d-428          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-429                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-430               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-431          	| [-1, 992, 16, 16]   	| 1984    	|
| ReLU-432                 	| [-1, 992, 16, 16]   	| 0       	|
| Conv2d-433               	| [-1, 128, 16, 16]   	| 126976  	|
| BatchNorm2d-434          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-435                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-436               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-437          	| [-1, 1024, 16, 16]  	| 2048    	|
| ReLU-438                 	| [-1, 1024, 16, 16]  	| 0       	|
| Conv2d-439               	| [-1, 128, 16, 16]   	| 131072  	|
| BatchNorm2d-440          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-441                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-442               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-443          	| [-1, 1056, 16, 16]  	| 2112    	|
| ReLU-444                 	| [-1, 1056, 16, 16]  	| 0       	|
| Conv2d-445               	| [-1, 128, 16, 16]   	| 135168  	|
| BatchNorm2d-446          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-447                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-448               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-449          	| [-1, 1088, 16, 16]  	| 2176    	|
| ReLU-450                 	| [-1, 1088, 16, 16]  	| 0       	|
| Conv2d-451               	| [-1, 128, 16, 16]   	| 139264  	|
| BatchNorm2d-452          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-453                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-454               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-455          	| [-1, 1120, 16, 16]  	| 2240    	|
| ReLU-456                 	| [-1, 1120, 16, 16]  	| 0       	|
| Conv2d-457               	| [-1, 128, 16, 16]   	| 143360  	|
| BatchNorm2d-458          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-459                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-460               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-461          	| [-1, 1152, 16, 16]  	| 2304    	|
| ReLU-462                 	| [-1, 1152, 16, 16]  	| 0       	|
| Conv2d-463               	| [-1, 128, 16, 16]   	| 147456  	|
| BatchNorm2d-464          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-465                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-466               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-467          	| [-1, 1184, 16, 16]  	| 2368    	|
| ReLU-468                 	| [-1, 1184, 16, 16]  	| 0       	|
| Conv2d-469               	| [-1, 128, 16, 16]   	| 151552  	|
| BatchNorm2d-470          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-471                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-472               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-473          	| [-1, 1216, 16, 16]  	| 2432    	|
| ReLU-474                 	| [-1, 1216, 16, 16]  	| 0       	|
| Conv2d-475               	| [-1, 128, 16, 16]   	| 155648  	|
| BatchNorm2d-476          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-477                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-478               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-479          	| [-1, 1248, 16, 16]  	| 2496    	|
| ReLU-480                 	| [-1, 1248, 16, 16]  	| 0       	|
| Conv2d-481               	| [-1, 128, 16, 16]   	| 159744  	|
| BatchNorm2d-482          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-483                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-484               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-485          	| [-1, 1280, 16, 16]  	| 2560    	|
| ReLU-486                 	| [-1, 1280, 16, 16]  	| 0       	|
| Conv2d-487               	| [-1, 128, 16, 16]   	| 163840  	|
| BatchNorm2d-488          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-489                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-490               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-491          	| [-1, 1312, 16, 16]  	| 2624    	|
| ReLU-492                 	| [-1, 1312, 16, 16]  	| 0       	|
| Conv2d-493               	| [-1, 128, 16, 16]   	| 167936  	|
| BatchNorm2d-494          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-495                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-496               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-497          	| [-1, 1344, 16, 16]  	| 2688    	|
| ReLU-498                 	| [-1, 1344, 16, 16]  	| 0       	|
| Conv2d-499               	| [-1, 128, 16, 16]   	| 172032  	|
| BatchNorm2d-500          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-501                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-502               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-503          	| [-1, 1376, 16, 16]  	| 2752    	|
| ReLU-504                 	| [-1, 1376, 16, 16]  	| 0       	|
| Conv2d-505               	| [-1, 128, 16, 16]   	| 176128  	|
| BatchNorm2d-506          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-507                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-508               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-509          	| [-1, 1408, 16, 16]  	| 2816    	|
| ReLU-510                 	| [-1, 1408, 16, 16]  	| 0       	|
| Conv2d-511               	| [-1, 128, 16, 16]   	| 180224  	|
| BatchNorm2d-512          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-513                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-514               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-515          	| [-1, 1440, 16, 16]  	| 2880    	|
| ReLU-516                 	| [-1, 1440, 16, 16]  	| 0       	|
| Conv2d-517               	| [-1, 128, 16, 16]   	| 184320  	|
| BatchNorm2d-518          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-519                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-520               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-521          	| [-1, 1472, 16, 16]  	| 2944    	|
| ReLU-522                 	| [-1, 1472, 16, 16]  	| 0       	|
| Conv2d-523               	| [-1, 128, 16, 16]   	| 188416  	|
| BatchNorm2d-524          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-525                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-526               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-527          	| [-1, 1504, 16, 16]  	| 3008    	|
| ReLU-528                 	| [-1, 1504, 16, 16]  	| 0       	|
| Conv2d-529               	| [-1, 128, 16, 16]   	| 192512  	|
| BatchNorm2d-530          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-531                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-532               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-533          	| [-1, 1536, 16, 16]  	| 3072    	|
| ReLU-534                 	| [-1, 1536, 16, 16]  	| 0       	|
| Conv2d-535               	| [-1, 128, 16, 16]   	| 196608  	|
| BatchNorm2d-536          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-537                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-538               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-539          	| [-1, 1568, 16, 16]  	| 3136    	|
| ReLU-540                 	| [-1, 1568, 16, 16]  	| 0       	|
| Conv2d-541               	| [-1, 128, 16, 16]   	| 200704  	|
| BatchNorm2d-542          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-543                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-544               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-545          	| [-1, 1600, 16, 16]  	| 3200    	|
| ReLU-546                 	| [-1, 1600, 16, 16]  	| 0       	|
| Conv2d-547               	| [-1, 128, 16, 16]   	| 204800  	|
| BatchNorm2d-548          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-549                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-550               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-551          	| [-1, 1632, 16, 16]  	| 3264    	|
| ReLU-552                 	| [-1, 1632, 16, 16]  	| 0       	|
| Conv2d-553               	| [-1, 128, 16, 16]   	| 208896  	|
| BatchNorm2d-554          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-555                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-556               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-557          	| [-1, 1664, 16, 16]  	| 3328    	|
| ReLU-558                 	| [-1, 1664, 16, 16]  	| 0       	|
| Conv2d-559               	| [-1, 128, 16, 16]   	| 212992  	|
| BatchNorm2d-560          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-561                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-562               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-563          	| [-1, 1696, 16, 16]  	| 3392    	|
| ReLU-564                 	| [-1, 1696, 16, 16]  	| 0       	|
| Conv2d-565               	| [-1, 128, 16, 16]   	| 217088  	|
| BatchNorm2d-566          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-567                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-568               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-569          	| [-1, 1728, 16, 16]  	| 3456    	|
| ReLU-570                 	| [-1, 1728, 16, 16]  	| 0       	|
| Conv2d-571               	| [-1, 128, 16, 16]   	| 221184  	|
| BatchNorm2d-572          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-573                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-574               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-575          	| [-1, 1760, 16, 16]  	| 3520    	|
| ReLU-576                 	| [-1, 1760, 16, 16]  	| 0       	|
| Conv2d-577               	| [-1, 128, 16, 16]   	| 225280  	|
| BatchNorm2d-578          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-579                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-580               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-581          	| [-1, 1792, 16, 16]  	| 3584    	|
| ReLU-582                 	| [-1, 1792, 16, 16]  	| 0       	|
| Conv2d-583               	| [-1, 128, 16, 16]   	| 229376  	|
| BatchNorm2d-584          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-585                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-586               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-587          	| [-1, 1824, 16, 16]  	| 3648    	|
| ReLU-588                 	| [-1, 1824, 16, 16]  	| 0       	|
| Conv2d-589               	| [-1, 128, 16, 16]   	| 233472  	|
| BatchNorm2d-590          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-591                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-592               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-593          	| [-1, 1856, 16, 16]  	| 3712    	|
| ReLU-594                 	| [-1, 1856, 16, 16]  	| 0       	|
| Conv2d-595               	| [-1, 128, 16, 16]   	| 237568  	|
| BatchNorm2d-596          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-597                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-598               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-599          	| [-1, 1888, 16, 16]  	| 3776    	|
| ReLU-600                 	| [-1, 1888, 16, 16]  	| 0       	|
| Conv2d-601               	| [-1, 128, 16, 16]   	| 241664  	|
| BatchNorm2d-602          	| [-1, 128, 16, 16]   	| 256     	|
| ReLU-603                 	| [-1, 128, 16, 16]   	| 0       	|
| Conv2d-604               	| [-1, 32, 16, 16]    	| 36864   	|
| BatchNorm2d-605          	| [-1, 1920, 16, 16]  	| 3840    	|
| AdaptiveMaxPool2d-606    	| [-1, 1920, 1, 1]    	| 0       	|
| AdaptiveAvgPool2d-607    	| [-1, 1920, 1, 1]    	| 0       	|
| AdaptiveConcatPool2d-608 	| [-1, 3840, 1, 1]    	| 0       	|
| Flatten-609              	| [-1, 3840]          	| 0       	|
| BatchNorm1d-610          	| [-1, 3840]          	| 7680    	|
| Dropout-611              	| [-1, 3840]          	| 0       	|
| Linear-612               	| [-1, 512]           	| 1966592 	|
| ReLU-613                 	| [-1, 512]           	| 0       	|
| BatchNorm1d-614          	| [-1, 512]           	| 1024    	|
| Dropout-615              	| [-1, 512]           	| 0       	|
| Linear-616               	| [-1, 5]             	| 2565    	|

##### Total params: 20,070,789
##### Trainable params: 2,206,917
##### Non-trainable params: 17,863,872
##### ----------------------------------------------------------------
##### Input size (MB): 3.00
##### Forward/backward pass size (MB): 2294.66
##### Params size (MB): 76.56
##### Estimated Total Size (MB): 2374.23
##### ----------------------------------------------------------------


## Results

First Stage

<img src="https://raw.githubusercontent.com/abhiksark/Diabetic-Retnopathy-Classification-ConvolutionalNeuralNetwork/master/Images/firstiteration.png" width="120%">

Second Stage
<img src="https://raw.githubusercontent.com/abhiksark/Diabetic-Retnopathy-Classification-ConvolutionalNeuralNetwork/master/Images/seconditeration.png" width="120%">

Third Stage
<img src="https://raw.githubusercontent.com/abhiksark/Diabetic-Retnopathy-Classification-ConvolutionalNeuralNetwork/master/Images/thirditeration.png" width="120%">

Confusion Matrix
<img src="https://raw.githubusercontent.com/abhiksark/Diabetic-Retnopathy-Classification-ConvolutionalNeuralNetwork/master/Images/confusionmatrix.png" width="120%">





## References

1. [What is Diabetic Retinopathy?](http://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/basics/definition/con-20023311)
2. [Blur Detection](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)

## Authors

* **[Abhik Sarkar](https://github.com/abhiksark)**
* [Fastai](https://www.fast.ai/)


## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>


