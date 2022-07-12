# Image-Caption-Generator
Generates text (caption) for a picture by implementing the concepts of a CNN and LSTM model.

### Methodology
Feature extraction is done using Xception model- Xception is a convolutional neural network architecture that relies solely on depthwise separable convolution layers.

For Training purpose LSTM model is used- The LSTM (Long Short Term Memory) model takes into consideration the state of the previous cell's output and the present cell's input for the current output

## Dataset
A kaggle dataset consisting of 8000 images is being used. Each image has 5 captions associated with it.

## Technology Stack
Python 3
Keras 2.4.3
Numpy 
Pandas
NLTK
Matplotlib
Pre-trained weights of Xception Model
