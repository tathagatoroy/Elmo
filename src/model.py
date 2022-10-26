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
        #print(x.shape)
        output1,output2, h0, h1, c0, c1, embedding = self.Elmo(x)
        return self.Linear(embedding)
    
class LM2(nn.Module):
    def __init__(self,device,batch_size = BATCH_SIZE , num_sequences = MAX_SENTENCE_SIZE, input_dimension = EMBEDDING_LAYER_DIMENSION,hidden_size = HIDDEN_DIMENSION, output_dimension = OUTPUT_DIMENSION, vocab_size = VOCAB_SIZE):
        super(LM2, self).__init__() 
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
        #print(x.shape)
        output1,output2, h0, h1, c0, c1, embedding = self.Elmo(x)
        #print(embedding.shape)
        #return self.Linear(embedding)
        #predicting the ith token should be done from using only i-1 th forward and i+1 th backward token 

        input_tensors = []
        seq_len = embedding.shape[0]
        for i in range(seq_len):
            #for i == 0 , forward token is the input hidden state
            if i == 0:
                l1 = h0[0,:,:]
                l2 = embedding[i+1,:,100:]
                inp = torch.cat((l1,l2),dim = 1)
                input_tensors.append(inp)

            elif i == seq_len - 1:
                l1 = embedding[i-1,:,:100]
                l2 = h0[1,:,:]
                inp = torch.cat((l1,l2),dim = 1)
                input_tensors.append(inp)
            else:
                l1 = embedding[i-1,:,:100]
                l2 = embedding[i+1,:,100:]
                inp = torch.cat((l1,l2), dim = 1)
                input_tensors.append(inp)
        final_input = torch.stack(input_tensors,dim = 0)
        return self.Linear(final_input)



    
               
class RoyNet(nn.Module):
    def __init__(self,device, batch_size = BATCH_SIZE , num_sequences = MAX_REVIEW_SIZE, input_dimension = EMBEDDING_LAYER_DIMENSION,hidden_size = HIDDEN_DIMENSION, output_dimension = OUTPUT_DIMENSION, output_size = OUTPUT_SIZE):
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
        self.device = device
        self.Elmo = BiLSTM(self.device)
        self.LSTM = nn.LSTM(self.input_dimension * 2, self.hidden_dimension)
        self.h0 = torch.rand(1, self.batch_size, self.output_dimension)
        self.c0 = torch.rand(1, self.batch_size, self.output_dimension)
        self.h0 = self.h0.to(self.device)
        self.c0 = self.c0.to(self.device)
        #load the elmo
        self.Elmo.load_state_dict(torch.load(ELMO2_MODEL_PATH))
        #freeze the elmo part of the model
        self.Elmo.requires_grad = False


    
        self.Linear = nn.Linear(self.hidden_dimension * self.num_sequences, self.output_size)
        
    def forward(self, x):
        """ performs the forward propagation 
        
        ARGS:
            x = the input vector 
            
        RETURNS :
            a tensor of NUM_SEQUENCES * VOCAB_SIZE
        """
        #print(x.shape)
        output1,output2, h0, h1, c0, c1, embedding = self.Elmo(x)
        #print(embedding.shape)
        embedding, (h2,c2) = self.LSTM(embedding, (self.h0,self.c0))
        
        #embedding is of the shape (SEQ_LENGTH, BATCH_SIZE, DIMENSION)
        #reshape 
        #first permute
        embedding = torch.permute(embedding, (1,0,2))
        #reshape 
        embedding = embedding.reshape(embedding.shape[0],embedding.shape[1] * embedding.shape[2])
        return self.Linear(embedding)

        #now LST
        
            
            
        return logits
