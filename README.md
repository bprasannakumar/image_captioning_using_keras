This is an image captioning project using deep learning with keras.

The dataset for this project can be objected from the following url.
https://forms.illinois.edu/sec/1713398.

The dataset contains 6,000 training images, 1,000 evaluation images, and 1,000 test images.

A VGG16 pre-trained model is used to extract features from the images.
A sequence model is used to generate sequence from data
A merge/decoder model is used to combine features and generate output.

We use BLEU score to test the accuracy of the model.
This idea is taken from the link https://arxiv.org/abs/1703.09137.



	


