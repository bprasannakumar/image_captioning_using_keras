# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:28:41 2018

@author: Kumar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import string
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu



#function definitaion parts
#Step 1: Load a pretrained model
def load_pretrained_model():
    vgg_model = VGG16()    
    #remove the last layer since we do not need the classification result
    vgg_model.layers.pop()    
    #set trainable false for all layers
    for layer in vgg_model.layers:
        layer.trainable = False        
    return Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-1].output)

#Step 2: Extract features from image data using pretrained model
def load_image_features(model, image_directory):
    #image_directory = 'flicker_dataset/Flicker8k_Dataset'
    img_features_dict = dict()
    for file_name in os.listdir(image_directory):
        print(file_name)
        #prepare the full pathname of the image
        img_file_path = os.path.join(image_directory, file_name)    
        #load the image, target size as per the input layer size of the vgg model
        image = load_img(path=img_file_path, target_size=(224, 224))
        #convert the image into a numpy array
        image = img_to_array(image)    
        #reshape the image as per input layer size[batch, height, width, channel]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])    
        #preprocess image as per vgg model using preprocess_input
        image = preprocess_input(image)
        #get the features of the image from the model
        features = model.predict(image)
        #get the image id from the file name
        image_id = file_name.split('.')[0]
        #store features of image into img_feature_dict
        img_features_dict[image_id] = features    
    return img_features_dict   
    
#Step 3: Store Extracted features into file using pickle for later use
def store_data_to_file(data):
    pickle.dump(data, open('image_features.pkl', 'wb'))     

#Step 4: Load descriptions for all images
def load_all_image_descriptions(description_file_name):
    #dictionary holds image and its corresponding description
    img_description_dict = dict()   
    # open the file in read mode
    with open(description_file_name, 'r') as file:
         #read data from the file line by line
         for line in file.readlines():                 
             #split the text by space
             tokens = line.split()
             image_id = tokens[0].split('.')[0]
             #print("image_id: ", image_id)
             img_desc = tokens[1:]
             img_desc = ' '.join(img_desc)             
             #if image_id not in list, then create an empty list, then append description
             if image_id not in img_description_dict:
                 img_description_dict[image_id] = list()
             #else append
             img_description_dict[image_id].append(img_desc)             
    return img_description_dict    
    
#Step 5: Process and clean all description text/convert to lower case, remove puncutation, remove stopwords
        
def process_image_description(description_dict):    
    processed_dict = dict()
    #create a translator to remove punctuations     
    translator = str.maketrans('', '', string.punctuation)    
    for key, values in description_dict.items():
        for line in values:
            words = line.split()            
            #convert to lower case
            descriptions = [word.lower() for word in words]
            #remove punctuations
            descriptions = [word.translate(translator) for word in descriptions]
            #remove single letter words and numeric values
            descriptions = [word for word in descriptions if len(word)>1 & word.isalpha()]           
            #convert list back to string
            descriptions = ' '.join(descriptions)                          
            processed_dict[key]=descriptions
    return processed_dict      

#Step 6: Create vocabulary from the processed image descritpion
def create_vocabulary(processed_desc_dict):
    text = ' '.join(processed_desc_dict.values())
    unique_text = set(text.split())
    return ' '.join(unique_text)


#Step 7: Load image ids from a given file
def load_image_ids_from_file(filename):
    image_id_list = []
    #open the given file in read mode
    with open(filename, 'r') as file:
        #read data from file line by line
        lines = file.readlines()
        for line in lines:
            #separate image id from image extension
            img_id = line.split('.')[0]
            image_id_list.append(img_id)
    return list(set(image_id_list))             
    
#Step 8: Load corresponding features from all features/stored in feature file
def load_image_features_for_ids(filename, image_ids):
    image_feature_dict = dict()         
    features = pickle.load(open(filename, 'rb'))
    for image_id in image_ids:
        image_feature_dict[image_id] = features[image_id]
    return image_feature_dict    
    
#Step 9: Load corresponding descriptions from the processed description/stored file 
def load_image_descriptions_for_ids(all_description_dict, image_ids):
    image_desc_dict = dict()           
    for image_id in image_ids:
        image_desc_dict[image_id] = all_description_dict[image_id]
    return image_desc_dict    

def add_start_end_words_to_data(description_dict):
    new_dict = dict()
    start_word = 'startseq '
    end_word = ' endseq'
    for img_id, img_des in description_dict.items():
        new_desc = start_word + img_des + end_word
        new_dict[img_id] = new_desc
    return new_dict    

#Step 10: Create a tokenizer with fit only
def create_tokenizer(descriptions):    
    all_desc = list()
    for v in descriptions.values():
        all_desc.append(v)               
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer    

#Step 11: Determine vocab size and max length of sequence
def get_max_length(description_dict):
    max_list = list()
    for k, val in description_dict.items():
        #print(k, len(val))
        max_list.append(len(val))
    return max(max_list)               

#Step 12: Create sequence generator for description and image features [X1train, X2train, ytrain]
#this returns features , description input seq, description output seq for input seq per image
def generate_sequence(features, descriptions, tokenizer, vocab_size, max_length):    
    X1, X2, y = list(), list(), list()    
    #tokenize all descriptions
    for img_id, img_des in descriptions.items():
       #print(img_id, img_des)     
        seq = tokenizer.texts_to_sequences([img_des])[0]
        #prepare input and output sequence
        for i in range(1, len(seq)):               
            input_seq = seq[:i]
            output_seq = seq[i]            
            #pad input sequence to make all descriptions with same length, pre padding is good
            input_seq = pad_sequences(sequences=[input_seq], maxlen=max_length, padding='pre')            
            #encode output sequence
            output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]           
            #add values to created list variables
            X1.append(features[img_id][0])
            X2.append(input_seq)
            y.append(output_seq)
    X2 = np.asarray(X2)
    X2 = X2.reshape(X2.shape[0],X2.shape[2])
    return np.asarray(X1), X2, np.asarray(y)                               

#Step 13: Create a data genetor to use in model fit_generator (Optional)
#Step 14: Define a RNN model with LSTM/GRU   
def create_model(vocab_size, max_length):
    #create three layers, 1 for feature processing, 1 for sequence process, 1 decoder
    #create feature processing layer
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    #create sequence processing layer
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
   	#create a merge decoder layer
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    #create model with above three layers
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

#Step 15: generate description for a given image
def generate_word_for_id(tokenizer, word_id):
    for word, index in tokenizer.word_index.items():
        if word_id == index:
            return word
    return None

def generate_description(model, features, tokenizer, max_length):
    seq_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([seq_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		prediction = model.predict([features,sequence], verbose=0)
		index = np.argmax(prediction)
		word = generate_word_for_id(tokenizer, index)
		if word is None:
			break
		seq_text += ' ' + word
		if word == 'endseq':
			break
	return seq_text

#Step 16: Make prediction for new image
def extract_features(filename):
    vmodel = VGG16()
    vmodel.layers.pop()
    model = Model(inputs=vmodel.inputs, outputs=vmodel.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

#Step 17: find accuracy of the model using BLUE score
def evaluate_model(model, features, descriptions, tokenizer, max_length):
    actual, predicted = list(), list()
    #iterate over the descriptions
    for key, desc_list in descriptions.items():
        # generate description
	    pred_descriptions = generate_description(model, features[key], tokenizer, max_length)
        acutal_descriptions = [d.split() for d in desc_list]
        actual.append(acutal_descriptions)
        predicted.append(pred_descriptions.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))