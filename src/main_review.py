import torch
import argparse
from config import *
from utils import *
from model import LM
from train_task import train2
import wandb

wandb.init(project = " Elmo ", entity = "roy3" , name = "Test Review Prediction ")

model = train2(TRAIN_DATASET_PATH, VAL_DATASET_PATH, 50)

