""" contains all the utility functions """

import pandas as pd
import numpy as np
import math 
import random 
import os 
from nltk import word_tokenize , sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from collections import OrderedDict
from config import *


def sentence_tokenize(filepath):
    """ given a filepath for a csv yelp dataset returns a list of sentences """
    sentences = []
    df = pd.read_csv(filepath)
    
    #get just the reviews 
    reviews = df.iloc[:]["text"]
    print("The number of reviews is reviews is {0}".format(len(reviews)))
    #cnt = 0
    for review in reviews:
        #get the sentences
        sents = sent_tokenize(review)
        for sentence in sents:
            
            sentences.append(clean_sentence(sentence))
            #if len(sentences) < 100:
                #print(sentence)
                #print(clean_sentence(sentence))
                #print(word_tokenize(clean_sentence(sentence)))
                #print(word_tokenize(sentence))
                #print(len(word_tokenize(clean_sentence(sentence))))
                #print()
    print("the number of sentences is {0}".format(len(sentences)))
    #writer = open("./../data/train.txt",'w')
    #writer.writelines(sentences)
    return sentences

def generate_reviews(filepath):
    reviews = []
    new_labels = []
    df = pd.read_csv(filepath)
     
    reviews = df.iloc[:]["text"]
    labels = df.iloc[:]["label"]
    labels = labels
    final_reviews = []
    for i,review in enumerate(reviews):
        new_review = ""
        review = review.replace("\\n"," ")
        for c in review :
            if c.isalnum() or c == " " or c == "." :
                new_review += c
        tokenized_review = word_tokenize(new_review)
        #print(new_review)
        #print(len(tokenized_review))
        #print(tokenized_review)
        if len(tokenized_review) > 10 and len(tokenized_review) <= MAX_REVIEW_SIZE:
            #print(new_review.type())
            final_reviews.append(new_review)
            new_labels.append(labels[i])
            #print(len(tokenized_review))
    print("size : {0} {1}".format(len(final_reviews), len(new_labels)))
    return final_reviews, new_labels





""" clean a sentence """
def clean_sentence(sentence):
    #remove \n 
    size = len(sentence)
    new_sentence = ""
    sentence = sentence.replace("\\n"," ")
    for c in sentence:
        #c = sentence[i]
        if c.isalnum() or c == " " or c == "." :
            new_sentence += c
    return new_sentence
        
            
""" tokenize a sentence to words"""
def word_tokenizer(sentence):
    return word_tokenize(sentence)
        

""" get maximum sentence size """
def max_sentence_length():
    train_sentences = sentence_tokenize(TRAIN_DATASET_PATH)
    val_sentences = sentence_tokenize(VAL_DATASET_PATH)
    test_sentences = sentence_tokenize(TEST_DATASET_PATH)
    
    max_train_size = 0
    max_test_size = 0
    max_val_size = 0
    
    train_sentence = None
    val_sentence = None
    test_sentence = None
    train_dist = {}
    test_dist = {}
    val_dist = {}
    
    for sent in train_sentences:
        sentence_size = len(word_tokenizer(sent))
        if sentence_size > max_train_size:
            train_sentence = sent
        max_train_size = max(max_train_size, sentence_size)
        if sentence_size in train_dist.keys():
            train_dist[sentence_size] += 1
        else:
            train_dist[sentence_size] = 1
        

        
    for sent in test_sentences:
        sentence_size = len(word_tokenizer(sent))
        if sentence_size <= 1:
            print(sent)
        if sentence_size > max_test_size:
            test_sentence = sent
        max_test_size = max(max_test_size, sentence_size)
        if sentence_size in test_dist.keys():
            test_dist[sentence_size] += 1
        else:
            test_dist[sentence_size] = 1
    
    for sent in val_sentences:
        sentence_size = len(word_tokenizer(sent))
        if sentence_size > max_val_size:
            val_sentence = sent
        max_val_size = max(max_val_size, sentence_size)
        if sentence_size in val_dist.keys():
            val_dist[sentence_size] += 1
        else:
            val_dist[sentence_size] = 1
    
    
    train_dist = OrderedDict(sorted(train_dist.items()))
    test_dist = OrderedDict(sorted(test_dist.items()))
    val_dist = OrderedDict(sorted(val_dist.items()))    
    print(" TRAIN : {0}   TEST : {1}   VAL : {2} ".format(max_train_size, max_test_size, max_val_size))
    return train_dist, val_dist, test_dist,val_sentence,train_sentence,test_sentence, 

""" filter sentences based on size """
def filter_sentences(sentences):
    new_sentences = []
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        if len(tokenized_sentence) > 2 and len(tokenized_sentence) <= MAX_SENTENCE_SIZE:
            new_sentences.append(sentence)
    return new_sentences 



#max_sentence_length()

#generate_reviews(TEST_DATASET_PATH)
#x = sentence_tokenize(TRAIN_DATASET_PATH)
