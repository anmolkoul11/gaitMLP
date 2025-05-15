# Gait Recognition Project 1

In this project, we are building an MLP using hand-crafted features. Our goal is to recognize human gait while walking on hard and soft terrains, and climbing up and down stairs. The input to our models are hand-crafted features from IMU devices.

## Data Context

The data capture correspond to individual wearing an IMU on their shin while walking on Centennial campus. Their gait motion was captured while they were standing, walking on solid even terrain and softer uneven terrain (grass in our case), and climbing up and down stairs. This data was captured to train a gait recognition model to aid on the development of robotic prosthesis.

## Dataset Description

Here is a brief description of the data files:

  - ".x.v" files contain the xyz accelerometers and xyz gyroscope measurements from the lower limb.
  - ".x.t" files contain the time stamps for the accelerometer and gyroscope measurements. The units are in seconds and the sampling rate is 40 Hz.
  - ".y.v" files contain the labels. (0) indicates standing or walking in solid ground, (1) indicates going down the stairs, (2) indicates going up the stairs, and (3) indicates walking on grass.
  - ".y.t" files contain the time stamps for the labels. The units are in seconds and the sampling rates is 10 Hz.

The dataset contents multiple sessions some of which are coming from the same subject. The training folder contains the data files for all the trials considered for training. The data set is imbalanced and we have used SMOTE technique to solve the imbalance issue for this project.

- `ProjC1.1 - EDA.ipynb` - Performs some exploratory analysis on the data.
- `ProjC1.2 - Baseline RF.ipynb` - It trains a simple Random Forest model for classification using the hand-crafted features. We report the results using the validation set.
- `ProjC1.3 - MLP.ipynb` - It uses an multiple layered MLP model for classification.
