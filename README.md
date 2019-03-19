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
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Preprocessing](#preprocessing)
    * [Download Images to EC2](#download-all-images-to-ec2)
    * [Crop & Resize Images](#crop-and-resize-all-images)
    * [Rotate and Mirror All Images](#rotate-and-mirror-all-images)
4. [CNN Architecture](#neural-network-architecture)
5. [Results](#results)
6. [Next Steps](#next-steps)
7. [References](#references)

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

## Exploratory Data Analysis





## Preprocessing

The preprocessing pipeline is the following:


### Crop and Resize All Images
All images were scaled down to 256 by 256. Despite taking longer to train, the
detail present in photos of this size is much greater then at 128 by 128.

Additionally, 403 images were dropped from the training set. Scikit-Image raised
multiple warnings during resizing, due to these images having no color space.
Because of this, any images that were completely black were removed from the
training data.

### Rotate and Mirror All Images
All images were rotated and mirrored.Images without retinopathy were mirrored;
images that had retinopathy were mirrored, and rotated 90, 120, 180, and 270
degrees.

The first images show two pairs of eyes, along with the black borders. Notice in
the cropping and rotations how the majority of noise is removed.



## Neural Network Architecture



## Results


## Next Steps


## References

1. [What is Diabetic Retinopathy?](http://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/basics/definition/con-20023311)

## Authors

* **[Abhik Sarkar](https://github.com/abhiksark)**
* [Fastai](https://www.fast.ai/)


## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>

