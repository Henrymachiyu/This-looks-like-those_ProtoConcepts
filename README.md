This code package implements the prototypical Concepts network (ProtoConcepts)
from the paper 

## "This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations"(NeurIPS 2023), 

Chiyu Ma* (Dartmouth College), Brandon Zhao* (Caltech),
Chaofan Chen (UMaine), and Cynthia Rudin (Duke University)
(* denotes equal contribution).
## Prerequisites
PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)
Recommended hardware: 1 NVIDIA Tesla V-100 GPU or 1 NVIDIA A-5000 GPUs

## Dataset 
Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
3. Unpack CUB_200_2011.tgz
4. Crop the images using information from bounding_boxes.txt (included in the dataset)
5. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
6. Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
7. Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
8. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      "./datasets/cub200_cropped/train_cropped_augmented/"

Dataset Stanford Cars can be downloaded from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

## Running code
Instructions for training a specific type of model are provided in the README file under corresponding model folder. 
