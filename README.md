This code package implements the prototypical Concepts network (ProtoConcepts)
from the paper "This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations"
(to appear at NeurIPS 1013), by Chiyu Ma* (Dartmouth College), Brandon Zhao* (Caltech),
Chaofan Chen (UMaine), and Cynthia Rudin (Duke University)
(* denotes equal contribution).

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)
Recommended hardware: 1 NVIDIA Tesla V-100 GPU or 1 NVIDIA A-5000 GPUs

Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack CUB_200_2011.tgz
3. Crop the images using information from bounding_boxes.txt (included in the dataset)
4. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
5. Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
6. Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
7. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      "./datasets/cub200_cropped/train_cropped_augmented/"

Instructions for training a specific type of model are provided in the README file under corresponding model folder. 
