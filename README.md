# Missing-Person-Detection_GAN

This project uses GANs to generate realistic images from sketches, improving missing person identification through deep learning techniques.

## Project Overview

This project leverages Generative Adversarial Networks (GANs) to match hand-drawn sketches of missing persons with real-world images from databases. The system enhances identification accuracy using contextual loss functions and deep learning techniques.

## Features

- Sketch-to-Image Matching: Converts sketches into realistic photos  
- GAN-based Approach: Utilizes deep learning for accurate image generation  
- High-Quality Image Synthesis: Improves recognition with contextual loss  
- Performance Evaluation: Uses SSIM and L2-norm for quality assessment  

## Project Structure

Missing-Person-Detection_GAN  
 ├── notebooks  
 │   ├── Predict Image.ipynb  
 │   ├── Testing.ipynb  
 │   ├── Compute SSIM and L2-norm.ipynb  
 │   ├── Data Augmentation.ipynb  
 ├── MISSINGPERSONGAN.pdf  
 ├── Proposed System Diagram for Missing Person Detection.docx  
 ├── README.md  
 ├── requirements.txt  



## Setup Instructions

### Clone the Repository

git clone https://github.com/your-username/Missing-Person-Detection_GAN.git cd Missing-Person-Detection_GAN


### Install Dependencies

pip install -r requirements.txt


### Run the Model

Open and execute the Jupyter Notebooks in the `notebooks/` folder using:


jupyter notebook


## Model Evaluation

We use L2-Norm and Structural Similarity Index (SSIM) to assess the generated images.

| Pixel Loss Weight | Contextual Loss Weight | L2-Norm | SSIM Score |  
|------------------|----------------------|---------|------------|  
| 1.0 | 0.0 | 90.68 | 0.7858 |  
| 0.8 | 0.2 | 92.39 | 0.7712 |  
| 0.5 | 0.5 | 93.45 | 0.7267 |  
| 0.2 | 0.8 | 93.63 | 0.7659 |  

## References

- Generative Adversarial Networks for Sketch-to-Photo Translation: [Link](https://arxiv.org/abs/1803.02077)  
- Face Sketch Recognition with Deep Learning: [Link](https://ieeexplore.ieee.org/document/10306444)  

