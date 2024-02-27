# Smartfarm Project
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## Flow_Chart
![](https://github.com/Roni81/smartfarm/blob/main/info_gram.png)

### Import Library
```python
import os
from glob import glob
```
```python
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
import random
```
```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
```
```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
```
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```
```python
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
```
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```


### Read image DATA
```python
main_path = "/content/drive/MyDrive/growingdata2"

train_imgs = glob(main_path + "/train/*/*/*.jpg") + glob(main_path + "/train/*/*/*.png")
train_imgs = sorted(train_imgs)

test_imgs = glob(main_path + "/test/images/*.jpg")+ glob(main_path + "/test/images/*.png")
test_imgs = sorted(test_imgs)

train_data = glob(main_path + "/train/*/meta/*.csv")
train_data = sorted(train_data)

train_label = glob(main_path + "/train/*/*.csv")
train_label = sorted(train_label)

test_data = test_data = glob(main_path + "/test/meta/*.csv")
test_data = sorted(test_data)
```

### Make preprocess image folders
<pre><code>
main_path = "/content/drive/MyDrive/growingdata2"

preprocessing_train_images = main_path + "/preprocessing_train"
preprocessing_test_images = main_path + "/preprocessing_test"

if not os.path.exists(preprocessing_train_images):
    os.mkdir(preprocessing_train_images)
if not os.path.exists(preprocessing_test_images):
    os.mkdir(preprocessing_test_images)
</code></pre>

### Image Augmentation
```python
def automatic_brightness_and_contrast(image, clip_hist_percent = 0.025):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result)
```

### Data Processing
```python
def get_image_data(dir_in, dir_out):

    ratio_lst = []

    for i in tqdm(dir_in):
        name = i.split("/")[-1] #i.split("/")[-1]
        img = cv2.imread(i,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1000,750))
        brightscale = automatic_brightness_and_contrast(img)
        imgcopy = brightscale.copy()
        hsvimage = cv2.cvtColor(brightscale,cv2.COLOR_BGR2HSV)
        lower = np.array([22,40,0])
        upper = np.array([85,255,245])
        mask = cv2.inRange(hsvimage, lower, upper)
        number_of_white_pix = np.sum(mask == 255)
        number_of_black_pix = np.sum(mask == 0)
        ratio = number_of_white_pix / (number_of_white_pix + number_of_black_pix)
        ratio_lst.append(ratio)
        result = cv2.bitwise_and(imgcopy, imgcopy, mask = mask)
        cv2.imwrite(os.path.join(dir_out, name), result)

    return ratio_lst
```



