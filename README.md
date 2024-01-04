# Smartfarm Project
<hr/>
<img src="https://img.shields.io/badge/background-SmartFarm-blue"/>
<pre><code>
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
import random

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
</code></pre>



###Read image DATA
<pre><code>
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

</code></pre>
