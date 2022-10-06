""" file which contains all the constants across the codebase"""

#Dataset Filepath
TRAIN_DATASET_PATH = "./../data/yelp-subset.train.csv"
TEST_DATASET_PATH = "./../data/yelp-subset.test.csv"
VAL_DATASET_PATH = "./../data/yelp-subset.dev.csv"

#Model Filepaths 
FASTTEXT_MODEL_PATH = "./../models/fasttext.mdl"
LM_MODEL_PATH = "./../models/LM.pth"
ELMO_MODEL_PATH = "./../models/elmo.pth"


#VECTOR_DIMENSION
EMBEDDING_LAYER_DIMENSION = 100
MINIMUM_TRAIN_COUNT = 3


NUMBER_OF_EPOCHS_FOR_FASTTEXT_MODEL = 5
WINDOW = 3
MAX_REVIEW_SIZE = 500
MAX_SENTENCE_SIZE = 75
VOCAB_SIZE  = 22341
OUTPUT_DIMENSION = 100
HIDDEN_DIMENSION = 100

BATCH_SIZE = 256
OUTPUT_SIZE = 5
LEARNING_RATE = 0.001 



