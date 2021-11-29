# Image Segmentation using PSPNet and UNet
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abouttheproject">About The Project</a>
      <ul>
        <li><a href="#Datasource">Data Source - Where is the Data?</a></li>
      </ul>
    </li>
    <li>
      <a href="#gettingstarted">Getting Started</a>
      <ul>
        <li><a href="#projectRequirements">Project Setup Requirements</a></li>
      </ul>
    </li>
    <li><a href="#Problemstobeaddressed">Problems to be addressed</a></li>
    <li><a href="#Potentialpitfalls&challenges">Potential pitfalls & challenges</a></li>
    <li><a href="#BackgroundResearch">Background Research</a></li>
    <li><a href="#AlgorithmsandCodesources">Algorithms and Code sources</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#Projectteammembers">Project team members</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project - Goals & Objectives

![image](https://user-images.githubusercontent.com/13203059/143803576-fe94be69-001e-41ab-a44c-acb72aee54cd.png)



There are several applications that come in the intersection of Computer vision and Deep learning, one such popular application is - Semantic Segmentation in Images. In this project idea, I will be working on an image dataset particularly containing Cars.

This dataset is great for training and testing models for image segmenatation with applications in autonomous driving and also driver assist.The dataset has diverse set of images when it comes to pose variation/background clutter/noise.The problems will be approached by means of employing image segmenation models like Pyramid Scene Parsing Network(PSPNet) and UNet and using Convolutional Neural Network(CNN)

In this portfolio idea some of the objectives that can be acheived through this data are - 
* Can the model be trained to detect and isolate/segment objects in the frame
* Comparing the results of the two segmentation models



<p align="right">(<a href="#top">back to top</a>)</p>

## Data Source - Where is the Data?

For this project the data will be sourced from a Kaggle dataset repository named - ![Cityscapes Image Pairs-
Semantic Segmentation for Improving Automated Driving](https://www.kaggle.com/dansbecker/cityscapes-image-pairs).

About the dataset - 

 The dataset has still images from the original videos, and the semantic segmentation labels are shown in images alongside the original image. This is one of the best datasets around for semantic segmentation tasks.
 
 This dataset has **2975 training images files** and **500 validation image** files. **Each image file is 256x512 pixels**, and each file is a composite with the original photo on the left half of the image, alongside the labeled image (output of semantic segmentation) on the right half.

Acknowledgment

This dataset is the same as what is available ![here](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/) from the Berkeley AI Research group.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Project Setup Requirements

- Requirements: Discovery Cluster, Google Colab Notebook/Jupyter Notebook, Github
- Data Source: Kaggle Dataset - ![Cityscapes Image Pairs-
Semantic Segmentation for Improving Automated Driving](https://www.kaggle.com/dansbecker/cityscapes-image-pairs)


<!-- PROBLEMS TO BE ADDRESSED -->
## Problems to be addressed

For the given dataset of Cars in traffic, using PSPNet and UNet we need to create image segmemtation of the original image, the segmented image gives us a sense of where the objects are in the image. This has particular application autonomous driving to detect neighbouring cars/traffic/pedestrians and navigate accordingly.


## Potential pitfalls & challenges

In this image segmentation problem there are several pitfalls/challenges, namely - 

- For the dataset, producing the segmented image in prediction as close to training data is a challenge. 
- In order to get good results the model training will be computationally expensive and time consuming if done on limited resources.

## Background Research

For the background research I will be skimming through relevant GitHub repositories and also look at these published papers that are relevant to image segmentation - 

    https://arxiv.org/abs/1505.04597 
	
    https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ 
	
    https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0 
	
	https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

## Algorithms and Code sources

In using Keras deep learning models, in the process of building the model for image segmentation I will be looking at the following techniques - 

PSPNet and implementation in Keras

solve the Semantic Segmentation problem using Fully Convolutional Network (FCN) called UNET





## References

Yann LeCun et al., 1998, ![Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

Adit Deshpande, 2016, ![The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

C.-C. Jay Kuo, 2016, ![Understanding Convolutional Neural Networks with A Mathematical Model](https://arxiv.org/pdf/1609.04112.pdf)

https://arxiv.org/abs/1505.04597 
	
https://www.depends-on-the-definition.com/unet-keras-segmenting-images/ 
	
https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0 
	
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

PyTorch & TensorBoard
https://www.youtube.com/watch?v=pSexXMdruFM
 (Links to an external site.)

Hyperparameter Tuning and Experimenting PyTorch & TensorBoard
https://www.youtube.com/watch?v=ycxulUVoNbk

https://keras.io/losses/


## Project team members

Kshitij Zutshi