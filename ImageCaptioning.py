import ProcessUtils as pu
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from keras.models import load_model

#Step 1: Load a pretrained model
model = pu.load_pretrained_model()

#Step 2: Extract features from image data using pretrained model
image_directory = 'flicker_dataset/Flicker8k_Dataset'
features = pu.load_image_features(model, image_directory)

#Step 3: Store Extracted features into file using pickle for later use
pu.store_data_to_file(features)

#Step 4: Load descriptions for all images
description_file_name = 'flicker_dataset/Ficker8k_text/Flickr8k.token.txt'
img_descriptions = pu.load_all_image_descriptions(description_file_name)    

#Step 5: Process and clean all description text/convert to lower case, remove puncutation, remove stopwords
processed_description = pu.process_image_description(img_descriptions)

#Step 6: Load ids from training text
training_file_path = 'flicker_dataset/Ficker8k_text/Flickr_8k.trainImages.txt'
training_img_ids = pu.load_image_ids_from_file(training_file_path)

#Step 7: Load corresponding features from all features/stored in feature file
features_filename = 'image_features.pkl'
training_img_features = pu.load_image_features_for_ids(features_filename, training_img_ids)

#Step 8: Load corresponding descriptions from the processed description/stored file 
training_img_desc = pu.load_image_descriptions_for_ids(processed_description, training_img_ids)
#add start and end word to description
training_img_desc = pu.add_start_end_words_to_data(training_img_desc)

#Step 9: Load ids from test text
test_file_path = 'flicker_dataset/Ficker8k_text/Flickr_8k.testImages.txt'
test_img_ids = pu.load_image_ids_from_file(test_file_path)

#Step 10: Load corresponding features from all features/stored in feature file
test_img_features = pu.load_image_features_for_ids(features_filename, test_img_ids)

#Step 11: Load corresponding descriptions from the processed description/stored file 
test_img_desc = pu.load_image_descriptions_for_ids(processed_description, test_img_ids)
#add start and end word to description
test_img_desc = pu.add_start_end_words_to_data(test_img_desc)

#Step 12: Create a tokenizer with fit only
tokenizer = pu.create_tokenizer(processed_description)
print((tokenizer.word_index))

#Step 13: Determine vocab size and max length of sequence
vocab_size = len(tokenizer.word_index)
max_length = pu.get_max_length(processed_description)

#Step 14: Create sequence generator for description and image features [X1train, X2train, ytrain]
#generate sequence for training set
X1train, X2train, ytrain = pu.generate_sequence(training_img_features, training_img_desc, tokenizer, vocab_size, max_length)
#generate sequence for validation set
X1test, X2test, ytest = pu.generate_sequence(test_img_features, test_img_desc, tokenizer, vocab_size, max_length)

#Step 17: Define a RNN model with LSTM/GRU
model = pu.create_model(vocab_size, max_length)

#Step 18: # define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#Step 19: Fit model
model.fit([X1train, X2train], ytrain, epochs=5, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

#Step 20: Make prediction using new image
new_image_path = 'new_image_folder/667626_18933d713e.jpg'
new_image = load_img(new_image_path, target_size=(224, 224))
plt.imshow(new_image)

#load the model from the check point
model = load_model('model-ep002-loss5.264-val_loss6.621.h5')
#get features of the image
new_features = pu.extract_features(new_image_path)

#generate_description for the new image
desc_for_new_image = pu.generate_description(model, new_features, tokenizer, max_length)

#Step 21: Evaluate model with test data
