""" implements the bilstm model """
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import random
from utils import *
from config import *


class BiLSTM(nn.Module):
    def __init__(self,device,batch_size = BATCH_SIZE , num_sequences = MAX_SENTENCE_SIZE, input_dimension = EMBEDDING_LAYER_DIMENSION,hidden_size = HIDDEN_DIMENSION, output_dimension = OUTPUT_DIMENSION):
        super(BiLSTM, self).__init__() 
        """ implements the BiLSTM ELmo Model
        
        ARGS :
            input_dimension : the embedding dimension , input is (N,I) where 
                            N is sequence size and I is the input dimension
            num_sequences : max sequence length
            output_dimension : dimension of the output 
            hidden_dimension : the dimension of hidden representations
        
        RETURNS :
            output : tensor of the shape (L,D*Hout)
            h1 : ( 2 , h_out ) all the hidden states for 2nd layers of bilsmt
            h0 : ( 2 , h_out) all the hidden states for 1st layer of bilstm
            c_0 : (2,  hcell) all the cell state of the first layer
            c_1 : (2, hcell) all the cell state of the second layer

        """ 
        self.device = device
        self.input_dimension = input_dimension
        self.num_sequences = num_sequences
        self.hidden_dimension = hidden_size
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.bilstm1 = nn.LSTM(input_size = self.input_dimension, hidden_size = self.hidden_dimension, bidirectional = True)
        self.bilstm2 = nn.LSTM(input_size = 2 * self.output_dimension, hidden_size = self.hidden_dimension, bidirectional = True)   
        self.h0 = torch.rand(2,self.batch_size,self.output_dimension)
        self.c0 = torch.rand(2,self.batch_size,self.hidden_dimension)
        self.Linear = nn.Linear(2*self.output_dimension, 2*self.output_dimension)
        self.a = torch.nn.Parameter(torch.FloatTensor(1,))
        self.b = torch.nn.Parameter(torch.FloatTensor(1,))
        self.w = torch.nn.Parameter(torch.FloatTensor(1,))
        self.h0 = self.h0.to(self.device)
        self.c0 = self.c0.to(self.device)
        #self.a = self.a.to(self.device)
        #self.b = self.b.to(self.device)
        #self.w = self.w.to(self.device)

    def forward(self, x):
        """ performs the forward propagation 
        
        ARGS:
            x = the input vector 
        RETURNS :
            output : tensor of the shape (L,D*Hout)
            h1 : ( 2 , h_out ) all the hidden states for 2nd layers of bilsmt
            h0 : ( 2 , h_out) all the hidden states for 1st layer of bilstm
            c_0 : (2,  hcell) all the cell state of the first layer
            c_1 : (2, hcell) all the cell state of the second layer
        
        """
        output1, (h0, c0) = self.bilstm1(x, (self.h0, self.c0))
        output2, (h1, c1) = self.bilstm2(output1, (self.h0, self.c0))
        #a = torch.nn.Parameter(torch.FloatTensor(1,))
        #b = torch.nn.Parameter(torch.FloatTensor(1,))
        #w = torch.nn.Parameter(torch.FloatTensor(1,))
        embedding = self.w * (self.a * output1 + self.b * output2)
        return output1,output2, h0, h1, c0, c1, embedding  
        

class LM(nn.Module):
    def __init__(self,device,batch_size = BATCH_SIZE , num_sequences = MAX_SENTENCE_SIZE, input_dimension = EMBEDDING_LAYER_DIMENSION,hidden_size = HIDDEN_DIMENSION, output_dimension = OUTPUT_DIMENSION, vocab_size = VOCAB_SIZE):
        super(LM, self).__init__() 
        """ implements the Language Model
        
        ARGS :
            input_dimension : the embedding dimension , input is (N,I) where 
                            N is sequence size and I is the input dimension
            num_sequences : max sequence length
            output_dimension : dimension of the output 
            hidden_dimension : the dimension of hidden representations
        
        RETURNS :
            a tensor of NUM_SEQUENCES * VOCAB_SIZE
            

        """ 
        self.input_dimension = input_dimension
        self.num_sequences = num_sequences
        self.hidden_dimension = hidden_size
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.Elmo = BiLSTM(self.device)
        self.Linear = nn.Linear(2 * self.input_dimension, self.vocab_size)
        
    def forward(self, x):
        """ performs the forward propagation 
        
        ARGS:
            x = the input vector 
            
        RETURNS :
            a tensor of NUM_SEQUENCES * VOCAB_SIZE
        """
        print(x.shape)
        output1,output2, h0, h1, c0, c1, embedding = self.Elmo(x)
        return self.Linear(embedding)
    
        
class RoyNet(nn.Module):
    def __init__(self,batch_size = BATCH_SIZE , num_sequences = MAX_SENTENCE_SIZE, input_dimension = EMBEDDING_LAYER_DIMENSION,hidden_size = HIDDEN_DIMENSION, output_dimension = OUTPUT_DIMENSION, output_size = OUTPUT_SIZE):
        super(RoyNet, self).__init__() 
        """ implements the Classification Model
        
        ARGS :
            input_dimension : the embedding dimension , input is (N,I) where 
                            N is sequence size and I is the input dimension
            num_sequences : max sequence length
            output_dimension : dimension of the output 
            hidden_dimension : the dimension of hidden representations
        
        RETURNS :
            a tensor of NUM_SEQUENCES * 5
            

        """ 
        self.input_dimension = input_dimension
        self.num_sequences = num_sequences
        self.hidden_dimension = hidden_size
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.output_size = output_size
        self.Elmo = BiLSTM()
        self.LSTM = LSMT(self.input_dimension * 2, self.hidden_dimension)
        self.h0 = torch.rand(self.output_dimension)
        self.c0 = torch.rand(self.output_dimension)
        self.Linear = nn.Linear(2 * self.input_dimension, self.vocab_size)
        
    def forward(self, x):
        """ performs the forward propagation 
        
        ARGS:
            x = the input vector 
            
        RETURNS :
            a tensor of NUM_SEQUENCES * VOCAB_SIZE
        """
        output1,output2, h0, h1, c0, c1, embedding = self.Elmo(x)
        size = embedding.shape
        hn = self.h0
        cn = self.c0
        logits = None
        print("Hello ")
        print(size)
        for i in range(size - 1):
            x = embedding[i]
            output, (h_n,c_n) = self.LSTM(x, hn)
            logit = self.Linear(output)
            print(i)
            if i == 0:
                logits = logit
            else:
                logits = torch.stack((logits,logit))
            
            
        return logits
