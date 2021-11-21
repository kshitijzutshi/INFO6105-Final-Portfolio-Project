<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Data source">Where is the Data?</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#projectRequirements">Project Setup Requirements</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project - Goals & Objectives

![image](https://user-images.githubusercontent.com/13203059/142774210-46a968f4-0374-4c18-9cbd-1be222095525.png)


There are several applications that come in the intersection of Computer vision and Deep learning, one such popular application is - Identifying faces/facial features. In this project idea, I will be working on an image dataset particularly of popular celebrities around the world.

This dataset is great for training and testing models for face detection, particularly for recognising facial attributes such as finding people with brown hair, are smiling, or wearing glasses. The dataset has diverse set of images when it comes to pose variation/background clutter/enthnicity. The dataset is also enriched with annotations. The problems will be approached by means of employing Deep learning techniques like Convolutional Neural Networks(CNN) and/or GANs

In this portfolio idea some of the objectives that can be acheived through this data are - 
* Can the model be trained to detect particular facial attributes?
* Which images contain people that are smiling?
* Classifying people with different hair, say - Straight or wavy
* Identifying the people faces as real or fake


<p align="right">(<a href="#top">back to top</a>)</p>

### Where is the Data?

For this project the data will be sourced from a Kaggle dataset repository named - ![CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset). This data was originally collected by researchers at MMLAB, The Chinese University of Hong Kong(specific reference in Acknowledgment).

About the dataset - 

* 202,599 number of face images of various celebrities
* 10,177 unique identities, but names of identities are not given
* 40 binary attribute annotations per image
* 5 landmark locations

Acknowledgment

Original data and banner image source came from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
As mentioned on the website, the CelebA dataset is available for non-commercial research purposes only. For specifics please refer to the website.

The creators of this dataset wrote the following paper employing CelebA for face detection:

S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Project Setup Requirements

- Requirements: Discovery Cluster, Google Colab Notebook/Jupyter Notebook, Github
- Data Source: Kaggle Dataset - ![CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)