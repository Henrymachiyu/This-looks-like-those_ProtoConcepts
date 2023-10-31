This code package implements ProtoPNet-Concepts 
The code is based on the other repositories: https://github.com/cfchen-duke/ProtoPNet
This code package was SOLELY developed by the authors at Duke University,
and licensed under MIT License (see LICENSE for more information regarding the use
and the distribution of this code package).

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)
Recommended hardware: 1 NVIDIA Tesla V-100 GPU or 1 NVIDIA Tesla A-5000 GPU

Instructions for training the model:
1. In settings.py, provide the appropriate strings for data_path, train_dir, test_dir,
train_push_dir:
(1) data_path is where the dataset resides
    -- if you followed the instructions for preparing the data, data_path should be "./datasets/cub200_cropped/"
(2) train_dir is the directory containing the augmented training set
    -- if you followed the instructions for preparing the data, train_dir should be data_path + "train_cropped_augmented/"
(3) test_dir is the directory containing the test set
    -- if you followed the instructions for preparing the data, test_dir should be data_path + "test_cropped/"
(4) train_push_dir is the directory containing the original (unaugmented) training set
    -- if you followed the instructions for preparing the data, train_push_dir should be data_path + "train_cropped/"
2. Run main.py

Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-imgdir is the directory containing the image you want to analyze
-img is the filename of the image you want to analyze
-imgclass is the (0-based) index of the correct class of the image


