import torch
import argparse
from config import *
from utils import *
from model import LM
from train import train
import wandb

wandb.init(project = " Elmo ", entity = "roy3" , name = " Elmo Pre-Training ")

model = train(TRAIN_DATASET_PATH, VAL_DATASET_PATH, 40)
