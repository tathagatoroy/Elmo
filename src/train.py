import numpy as np
import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import time
# from torch.utils.data import Dataset
import wandb
from torch.utils.data import DataLoader
import argparse
# from torchsummary import summary
from utils import *
from dataset import Dataset
from config import *
from model import LM,LM2

# wandb.init(project = "Natural Language Understanding ", entity = "roy3" , run = "try 1")

def train(train_dataset_filepath, val_dataset_filepath, epochs, model_path = LM2_MODEL_PATH, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, model_saved=False):
    """  function to train the model on language modelling

        ARGS :
            model : the model to be trained
            train_dataset_filepath : path to the training set
            val_dataset_filepath : path to the val set
            criterion : the loss function Eg Cross Entropy
            optimizer : the optimizer used for the weight update : Adam
            epochs : number of epochs
            batch_size : batch size = 32
            model_path : path which stores previously trained model and wbich savaes the currently training model
            model_saved : boolean which stores whether the model is saved or not
            learning_rate  : learning rate eg 0.001
            embedding  :  embedding model returns (4*100,1)

        RETURNS :
        A trained model, along with logging info to the wandb
        """
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs ": epochs,
        "batch_size": batch_size,
        "loss": "CrossEntropy",
        "Optimizer": "Adam",
        "sequence_length": MAX_SENTENCE_SIZE,

        }

    # check if gpu is there
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    if device == "cpu":
        print("No GPU Found")
    else: print("GPU Found")
    # clear cache
    # torch.cuda.empty_cache()

    # build vocabulary
    # print("build the vocabulary")
    # print(train_dataset_filepath)
    # word_to_id , id_to_word = build_vocabulary(train_dataset_filepath)
    # print("vocabulary built")
    # get the train_dataset  and val_dataset
    train_dataset = Dataset(train_dataset_filepath)
    val_dataset = Dataset(val_dataset_filepath)


    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    # get the corresponding dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    # print(vocabulary_size)
    model = LM2(device)
   
    # model loading
    if model_saved:
        model.load_state_dict(torch.load(LM2_MODEL_PATH))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    model = model.float()
    model = model.to(device)


# begin training

    error = {"train": [], "val": []}
    accuracy = {"train" : [], "val" : []}
    #perplexity = {"train": [], "val": []}
    times = {"train": [], "val": []}

    for epoch in range(epochs):
        t1 = time.time()
        print("starting Epoch : " + str(epoch + 1))

        train_samples = 0
        val_samples = 0
        train_loss = 0
        val_loss = 0
        val_accuracy = 0
        train_accuracy = 0
        train_count = 0
        val_count = 0
        t1 = time.time()
        model.train()
        for i, example in enumerate(train_dataloader):
            embeddings,labels = example
            if embeddings.shape[0] != BATCH_SIZE :
                continue
            # print(sequence.shape)
            # print(next_word.shape)
            # print(len(sequence))
            # print(len(next_word))
            # print(sequence)
            # print(next_word)
            # print(input_vector.shape)
            # print(label)
            # print("")
            # if i > 20 :
                # break
            # add the input vector and label to the gpu
            
            input_vector = embeddings.to(device)
            #input_vector = torch.reshape(input_vector, (input_vector.shape[1],input_vector.shape[0],input_vector.shape[2]))
            input_vector = input_vector.permute(1,0,2)
            labels = labels.to(device)

            # forward pass
            logits = model(input_vector)
            #logits = torch.reshape(logits,(logits.shape[1],logits.shape[2],logits.shape[0]))
            logits = logits.permute(1,2,0)
            
            # print("shape of the logits : {0}",format(logits.shape))

            # compute loss
            #print(logits.shape)
            #print(labels.shape)
            #print(torch.max(labels))
            loss = criterion(logits, labels)

            # set the gradients to zero
            optimizer.zero_grad()

            # back.prop
            loss.backward()

            # update parameters
            optimizer.step()

            train_loss += loss.item()
            predictions = logits.argmax(dim=1)

            # update accuracy
            train_accuracy += (predictions == labels).sum()

            train_count += predictions.shape[0] * predictions.shape[1]
            train_samples += predictions.shape[0] 
            print("Progress : {0} \r".format(train_samples * 100 / train_dataset_size))

        t2 = time.time()
        print("Time taken to run training for epoch {0} : {1} ".format(
            epoch, t2 - t1))
        t3 = time.time()

        model.eval()
        for i, example in enumerate(val_dataloader):

            embeddings,labels = example
            if embeddings.shape[0] != BATCH_SIZE:
                continue
            # print(sequence.shape)
            # print(next_word.shape)
            # print(len(sequence))
            # print(len(next_word))
            # print(sequence)
            # print(next_word)
            # print(input_vector.shape)
            # print(label)
            # print("")
            # if i > 20 :
                # break
            # add the input vector and label to the gpu
            input_vector = embeddings.to(device)
            #input_vector = torch.reshape(input_vector, (input_vector.shape[1],input_vector.shape[0],input_vector.shape[2]))
            input_vector = input_vector.permute(1,0,2)


            labels = labels.to(device)

            # forward pass
            logits = model(input_vector)
            #logits = torch.reshape(logits,(logits.shape[1],logits.shape[2],logits.shape[0]))
            logits = logits.permute(1,2,0)

            
            # print("shape of the logits : {0}",format(logits.shape))

            # compute loss
            #print(logits.shape)
            #print(labels.shape)
            #print(torch.max(labels))
            loss = criterion(logits, labels)

            # set the gradients to zero
            #optimizer.zero_grad()

            # back.prop
            #loss.backward()

            # update parameters
            ##optimizer.step()

            val_loss += loss.item()
            predictions = logits.argmax(dim=1)

            # update accuracy
            val_accuracy += (predictions == labels).sum()

            val_count += predictions.shape[0] * predictions.shape[1]
            val_samples += predictions.shape[0] 
            print("Progress : {0} \r".format(val_samples * 100 / val_dataset_size))

        t2 = time.time()
        print("Time taken to run training for epoch {0} : {1} ".format(
            epoch, t2 - t1))
        t3 = time.time()


        t4 = time.time()
        #train_perplexity = get_perplexity(loss)
        #val_perplexity = get_perplexity(loss)
        train_accuracy = train_accuracy / train_count
        val_accuracy = val_accuracy / val_count
        print("Time taken to run validation for epoch {0} : {1} ".format(
            epoch, t4 - t3))
        print("Total Time for epoch {0} is {1}".format(epoch, t4 - t1))
        print("Training Loss for epoch {0} : {1} ".format(epoch + 1, train_loss))
        print("Training Accuracy for epoch {0} : {1}".format(
            epoch + 1, train_accuracy))

        print("Val Loss for epoch {0} : {1} ".format(epoch + 1, val_loss))
        print("Val Accuracy for epoch {0} : {1}".format(epoch + 1, val_accuracy))
        #print("Train perplexity for epoch {0} : {1}".format(
        #    epoch + 1, train_perplexity))
        # print("Val perplexity for epoch {0} : {1}".format(
        #    epoch + 1, val_perplexity))
        error['train'].append(train_loss)
        error['val'].append(val_loss)
        accuracy['train'].append(train_accuracy)
        accuracy['val'].append(val_accuracy)
        times['train'].append(t2 - t1)
        times['val'].append(t4-t3)


        # log into wandb
        wandb.log({
        'train_accuracy': train_accuracy,
        'train_loss': train_loss,
        'train_time': t2 - t1,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'val_time': t4 - t3
        })

        # save the model
        if (epoch + 1) % 1 == 0:
            print("Saving the model ")
            torch.save(model.state_dict(), LM2_MODEL_PATH)
            torch.save(model.Elmo.state_dict(), ELMO2_MODEL_PATH)










"""
model = FNN(VECTOR_SIZE,400)
train_dataset_filepath = TRAIN_DATASET_PATH
val_dataset_filepath = VAL_DATASET_PATH
criterion = torch.nn.CrossEntropy()
epochs = NUM_EPOCHS
model_path = FNN_MODEL_PATH


train(model,train_dataset_filepath,val_dataset_filepath, criterio, epochs , model_path , batch_size = BATCH_SIZE , learning_rate = LEARNING_RATE,model_saved = False)
""" 
