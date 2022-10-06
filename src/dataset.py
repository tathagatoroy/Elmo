from tkinter.tix import WINDOW
import numpy as np
import pandas as pd
import os 
from utils import *
from config import *
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
import gensim 
import torch 
from nltk.corpus import brown
class FastText:
    def __init__(self,path = FASTTEXT_MODEL_PATH, vector_size = EMBEDDING_LAYER_DIMENSION, window = WINDOW, force_train = False):
        """
        The model which generates Fastext embeddings for words
        
        ARGS:
            path : stores the model in memory
            vector_size : size of the output embeddings
            window : window for training 
            force_train: retrain even if model is saved 
            
        RETURNS:
            a model which generates word embeddings
             
        """
        self.path = path
        self.vector_size = vector_size
        self.window = window
        self.force_train = force_train
        self.sentences = sentence_tokenize(TRAIN_DATASET_PATH)
        
        
        #train if model doesn't exist or forced to train 
        if self.force_train or not os.path.exists(self.path):
            self.model = gensim.models.FastText(vector_size = self.vector_size, window = self.window , min_count = MINIMUM_TRAIN_COUNT)
            print("Building Vocabulary")
            self.model.build_vocab(brown.sents())
            print("Vocabulary built ")
            print("Finetuning FastText")
            self.model.train(brown.sents(), total_examples = len(brown.sents()), epochs = NUMBER_OF_EPOCHS_FOR_FASTTEXT_MODEL)
            self.model.save(self.path)
        else :
            self.model = gensim.models.FastText.load(self.path)
            
    
    def get_vocab(self):
        """ returns the vocabulary generated """
        return self.model.wv.key_to_index
        
    def __call__(self, word):
        """
        given a word, returns the fasttext word embedding 
        
        ARGS : 
            word: input word
        
        
        RETURNS :  
            the embedding vector

        """       
        return self.model.wv[word]

 
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath = TRAIN_DATASET_PATH):
        """
        The Dataset Class which builds the dataset 
        
        ARGS :
            filepath = file path to the dataset csv
            
        PARAMS:
            pad : string represent token
            max_length : max_sequence length
            sentences : all valid sentences
            unknown : id  
            vocab_to_id : do
            
            
        RETURNS : A dataset class which returns a set sentence embedding and the set of labels 
        """
        
        self.path = filepath
        self.pad = "<PAD>"
        self.unk = "<UNK>"
        self.max_length = MAX_SENTENCE_SIZE
        
        self.sentences = filter_sentences(sentence_tokenize(self.path))
        self.embedding = FastText()
        self.vocab_to_id = self.embedding.get_vocab()
        self.id_to_vocab = {}
        for key in self.vocab_to_id.keys():
            self.id_to_vocab[self.vocab_to_id[key]] = key
        current_max_id = max(self.id_to_vocab.keys())
        self.unknown_id = current_max_id + 1
        self.pad_id = current_max_id + 2
        

    def __len__(self):
        """ function which returns the dimension of the dataset ,neccessary for pytorch dataset class
        RETURNS :
            size of the dataset
        """
        return len(self.sentences)
    
    def __getitem__(self,index):
        
        
        """ function which returns one example from the dataset indexed by the index 
        ARGS : 
            index : the index of the example 
 
            
        RETURNS 
        (       list of words =  
                X : A tensor of dimension of Sequence_length * Embedding Dimension
                Y : A Sequence Length of label
                )
        """
        sentence = self.sentences[index]
        words = word_tokenize(sentence)
        embeddings = []
        labels = []
        for word in words:
            if word in self.vocab_to_id.keys():
                embedding = self.embedding(word)
                label = self.vocab_to_id[word]
                embeddings.append(embedding)
                labels.append(label)
            else:
                embedding = self.embedding(self.unk)
                label = self.unknown_id
                embeddings.append(embedding)
                labels.append(label)
        while(len(labels) < MAX_SENTENCE_SIZE):
            labels.append(self.pad_id)
            embeddings.append(self.embedding(self.pad))
        #print(label.shape)
        #print(embedding.shape)
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        #print(labels.shape)
        #print(embeddings.shape)

        return (embeddings,labels)
                 
            
            
class Dataset2(torch.utils.data.Dataset):
    def __init__(self,filepath = TRAIN_DATASET_PATH):
        """
        The dataset class for the 5 way classification 

        ARGS :
            filepath : filepath to the dataset csv 

        PARAMS :
            pad : string pad represent token
            max_length : max review length
        """
        
        self.path = filepath
        self.pad = "<PAD>"
        self.unk = "<UNK>"
        self.embedding = FastText()
        self.max_length = MAX_REVIEW_SIZE
        reviews, labels = generate_reviews(self.path)
        self.reviews = reviews
        self.labels = labels
        self.vocab_to_id = self.embedding.get_vocab()
        self.id_to_vocab = {}
        for key in self.vocab_to_id.keys():
            self.id_to_vocab[self.vocab_to_id[key]] = key
        current_max_id = max(self.id_to_vocab.keys())
        self.unknown_id = current_max_id + 1
        self.pad_id = current_max_id + 2

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,index):

        review = self.reviews[i]
        label = self.label[i]
        embeddings = []
        for word in review:
            if word in self.vocab_to_id.keys():
                embedding = self.embedding(word)
                embeddings.append(embedding)
            else:
                embedding = self.embedding(self.unk)
                embeddings.append(embedding)
            while(len(embeddings) < MAX_REVIEW_SIZE):
                embeddings.append(self.embedding(self.pad))
        embeddings = np.array(embeddings)
        return(embeddings, label)


             





        
    





#embedding = FastText()
#print(embedding("hello"))
