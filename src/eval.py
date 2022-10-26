import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb 
from torch.utils.data import DataLoader
import argparse
from utils import *
from dataset import Dataset2
from config import *
from model import RoyNet

def evaluate(test_data_filepath = TEST_DATASET_PATH, model_path = NET2_MODEL_PATH,batch_size = BATCH_SIZE):
    """ function to eval the downstream model 
        ARGS : 
            test_data_filepath : test dataset filepath
            model_path : trained model path

        RETURNS :
            Evaluation Metrics
    """
    #wandb.init(project = "Natural Language Understanding ", entity = "roy3", run = "Evaluate")
    
    #get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #dataset
    test_dataset = Dataset2(test_data_filepath)
    test_dataset_size = len(test_dataset)
    print("Size : " + str(test_dataset_size))

    #dataloader 
    test_dataloader = DataLoader(
            test_dataset, batch_size = batch_size, shuffle = True)
    model = RoyNet(device)
    
    #load the model
    model.load_state_dict(torch.load(NET2_MODEL_PATH))
    model = model.float()
    model = model.to(device)
    #class_wise_accuracy = []
    correct = [0,0,0,0,0]
    total = [0,0,0,0,0]
    model.eval()

    for i,example in enumerate(test_dataloader):
        embeddings,labels = example
        if embeddings.shape[0] != BATCH_SIZE:
            continue

        input_vector = embeddings.to(device)
        input_vector = input_vector.permute(1,0,2)
        
        labels = labels.to(device)

        logits = model(input_vector)
        predictions = logits.argmax(dim = 1)
        size = predictions.shape[0]
        print(predictions)
        print(labels)
        for j in range(size):
            prediction = predictions[j]
            label = labels[j]
            if prediction == label :
                d = label 
                correct[label] += 1
            total[label] += 1
        cur_total = sum(total)
        print("Progress : {0} ".format(cur_total/ test_dataset_size))

    print("Per Class Accuracy : ")
    for i in range(5):
        print("Label : {0}   Accuracy : {1}  Total Examples : {2} ".format(i,correct[i]/total[i],total[i]))
    print("Total Accuracy : {0}".format(sum(correct)/sum(total)))



evaluate(TEST_DATASET_PATH)



