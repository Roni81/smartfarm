# Smartfarm Project
<hr/>
<img src="https://img.shields.io/badge/background-SmartFarm-blue"/>
```
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
```
