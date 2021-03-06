Graph based Machine Learning & its Applications

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

![alina-grubnyak-ZiQkhI7417A-unsplash](https://user-images.githubusercontent.com/13203059/142969193-9783a78f-9cdf-4d86-ada2-873c51346d09.jpg)




There are several applications that come in the intersection of Computer vision and Deep learning, one such popular application is - Identifying image Scene features. In this project idea, I will be working on an image dataset particularly containing a set of common scenic objects like - Buildings, forests, glaciers, mountain, sea and street.

This dataset is great for training and testing models for scene classification of multiclass.The dataset has diverse set of images when it comes to pose variation/background clutter/noise. The dataset is also enriched with annotations. The problems will be approached by means of employing Deep learning techniques like Convolutional Neural Networks(CNN).

In this portfolio idea some of the objectives that can be acheived through this data are - 
* Can the model be trained to detect one of the six classes of scene objects?
* Comparing the results using different keras models



<p align="right">(<a href="#top">back to top</a>)</p>

## Data Source - Where is the Data?

For this project the data will be sourced from a Kaggle dataset repository named - ![Intel Image Classification - Image Scene Classification of Multiclass](https://www.kaggle.com/puneet6060/intel-image-classification). This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.

About the dataset - 

In total, the Dataset has 25000 images of size 150 x 150 distributed under 6 categories.
There are about 14000 images in Train, 3000 images in test and 7000 in prediction.

Acknowledgment

Thanks to https://datahack.analyticsvidhya.com for the challenge and Intel for the Data

Photo by Jan B??ttinger on Unsplash

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Project Setup Requirements

- Requirements: Discovery Cluster, Google Colab Notebook/Jupyter Notebook, Github
- Data Source: Kaggle Dataset - ![Intel Image Classification - Image Scene Classification of Multiclass](https://www.kaggle.com/puneet6060/intel-image-classification)


<!-- PROBLEMS TO BE ADDRESSED -->
## Problems to be addressed

In this multiclass image classification problem, given that the CNN model would be trained on 6 or less pre-defined categories of scenes like - Buildings, forests, glaciers, mountain, sea and street. It is expected that the model is able to correctly classify new unseen real world images into one of the 6 categories its trained on.


## Potential pitfalls & challenges

In this image classification problem there are several pitfalls/challenges, namely - 

- The Input image dataset size is verry large 25000+ images, loading these images on currently available hardware would be an issue, the dataset size can be reduced by propotionately reducing the number of images of each class.
- For the given 6 categories, training the model for all 6 classes will be computationally intensive, it could be reduced to say 3 or 4 categories.

## Background Research

For the background research I will be skimming through relevant GitHub repositories and also look at these published papers that are relevant to CNN - 

    Yann LeCun et al., 1998, Gradient-Based Learning Applied to Document Recognition
    Adit Deshpande, 2016, The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)
    C.-C. Jay Kuo, 2016, Understanding Convolutional Neural Networks with A Mathematical Model

## Algorithms and Code sources

In using Keras deep learning models, in the process of building the model for image classification I will be looking at the following techniques - 

- Usages of Activation functions like - Rectified linear unit (ReLU)
TanH
Leaky rectified linear unit (Leaky ReLU)
Parameteric rectified linear unit (PReLU) Randomized leaky rectified linear unit (RReLU)
Exponential linear unit (ELU)
Scaled exponential linear unit (SELU)
S-shaped rectified linear activation unit (SReLU)
Adaptive piecewise linear (APL)

- Usages of different types of cost functions like - Quadratic cost (mean-square error)
Cross-Entropy
Hinge
Kullback???Leibler divergence
Cosine Proximity

- Usages of different gradient estimations like - Stochastic Gradient Descent
Adagrad
RMSProp
ADAMN
AGAdadelta
Momentum




## References

Yann LeCun et al., 1998, ![Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

Adit Deshpande, 2016, ![The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

C.-C. Jay Kuo, 2016, ![Understanding Convolutional Neural Networks with A Mathematical Model](https://arxiv.org/pdf/1609.04112.pdf)

PyTorch & TensorBoard
https://www.youtube.com/watch?v=pSexXMdruFM
 (Links to an external site.)

Hyperparameter Tuning and Experimenting PyTorch & TensorBoard
https://www.youtube.com/watch?v=ycxulUVoNbk

https://keras.io/losses/


## Project team members

Kshitij Zutshi