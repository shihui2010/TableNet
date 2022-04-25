import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import Image
import pytesseract
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, Add, Activation, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.applications import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.applications.densenet import DenseNet121
from PIL.Image import frombuffer
from numpy.core.fromnumeric import size
import streamlit as st

# Classes for the custom decoders
# table decoder
class table_decoder(tf.keras.layers.Layer):
    '''
        This is the custom layer defined for the table decoder following the above architecture.
    '''
    def __init__(self):
        super().__init__()
        # defining the layers
        self.conv7 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001))
        self.upsamp_pool4 = UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsamp_pool3 = UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsamp_out = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=2, padding='same', activation='softmax')
        
    def call(self, inp, pool3, pool4):
        # passing through convolution
        x = self.conv7(inp)
        # upsampling and concatenating with pool4
        x = self.upsamp_pool4(x)
        x = Concatenate()([x, pool4])
        
        # upsampling and concatenating with pool3
        x = self.upsamp_pool3(x)
        x = Concatenate()([x, pool3])
        
        # further upsampling and output
        x = UpSampling2D((2,2))(x)
        x = UpSampling2D((2,2))(x)
        fin = self.upsamp_out(x)
        
        return fin

# column decoder
class col_decoder(tf.keras.layers.Layer):
    '''
        This custom layer is defined for the Column deocder following the above column decoder architecture. 
    '''
    def __init__(self):
        super().__init__()
        # defining the layers
        self.conv7 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001), activation='relu')
        self.drop = Dropout(rate=0.8)
        self.conv8 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001))
        self.upsamp_pool4 = UpSampling2D((2,2), interpolation='bilinear')
        self.upsamp_pool3 = UpSampling2D((2,2), interpolation='bilinear')
        self.upsamp_out = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=2, padding='same', activation='softmax')
        
    def call(self, inp, pool3, pool4):
        # passing through convolutions
        x = self.conv7(inp)
        x = self.drop(x)
        x = self.conv8(x)
        
        # upsampling and concatenating encoder pool outputs
        x = self.upsamp_pool4(x)
        x = Concatenate()([x, pool4])
        x = self.upsamp_pool3(x)
        x = Concatenate()([x, pool3])
        
        # final upsampling and outputs
        x = UpSampling2D((2,2))(x)
        x = UpSampling2D((2,2))(x)
        fin = self.upsamp_out(x)
        
        return fin

# making the model archtecture
def ModelConstructor():
    '''
        This function makes the tablenet architecture and returns the model object after loading the trained weights.
    '''
    tf.keras.backend.clear_session()
    # making the encoder architecture
    tf.keras.backend.clear_session()
    model_input = Input(shape=(1024,1024,3))
    encoder = DenseNet121(include_top=False, weights='imagenet', input_tensor=model_input)

    # for pool3 and pool4, we are going to use the outputs of the following layers
    # pool4 = pool4_relu
    # pool3 = pool3_relu
    pool3 = encoder.get_layer('pool3_relu').output
    pool4 = encoder.get_layer('pool4_relu').output

    # making all the layers of the encoder untrainable
    for layer in encoder.layers:
        layer.trainable = False

    # continuing the model architecture
    # convolution layers
    conv_6 = Conv2D(filters=512, kernel_size=(1,1), activation='relu', name='block6_conv1',
                    kernel_regularizer=regularizers.l2(0.001))(encoder.output)
    conv6_drop = Dropout(0.2)(conv_6) # this is the dropping probability and in the paper the keep_prop seems to be 0.8.

    conv_7 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', name='block6_conv2',
                    kernel_regularizer=regularizers.l2(0.001))(conv6_drop)
    conv7_drop = Dropout(0.2)(conv_7) # this is the dropping probability and in the paper the keep_prop seems to be 0.8.
    # decoders
    table_mask = table_decoder()
    column_mask = col_decoder()

    table_out = table_mask(conv7_drop, pool3, pool4)
    column_out = column_mask(conv7_drop, pool3, pool4)

    # declaring the model

    tablenet = Model(inputs=model_input, outputs=[table_out, column_out])

    # loading the weights
    tablenet.load_weights('DenseNet-Tablenet.h5')

    return tablenet

# HELPER FUNCTIONS
def decode_image(uploader):
    '''
        This functions takes the uploader object and extracts the image out of it
        and then decodes the image into a numpy array to be used for the model.
        Note: This function does not check if the image is uploaded or not, thus, 
        a manual check is required for checking if the uploader actually contains an image.
    '''
    # getting the bytes from the uploader
    image_bytes = uploader.getvalue()
    # converting the bytes into a numpy array - https://stackoverflow.com/a/49517948/11881261
    image_decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)[:,:,:3]
    
    return image_decoded
    
def predict_masks(image, model):
    '''
        This function takes the image tensor, preprocesses it and predicts the table and column masks from the image.
    '''
    # preprocessing the image
    # resizing
    im = tf.image.resize(image, size=[1024,1024])
    im = tf.cast(im, dtype=tf.float32)/255

    # making a batch
    im = tf.expand_dims(im, axis=0)

    # making prediction using the model
    table_mask, col_mask = model.predict(im)

    return (im, table_mask, col_mask)

def get_mask_image(mask_pred):
    '''
        This function gets the predicted mask image from the masks predicted by the model
    '''
    # taking argmax from both the channels
    mask_pred = tf.argmax(mask_pred, axis=-1)
    # adding a channel axis
    mask_pred = mask_pred[..., tf.newaxis][0]
    
    return mask_pred

def filter_table(image, table_mask):
    '''
        This function turns the image from a matrix to actual image and then uses the table mask to filter out the table from the image.
    '''
    # converting image and mask from matrices to images
    im = tf.keras.preprocessing.image.array_to_img(image)
    mask = tf.keras.preprocessing.image.array_to_img(table_mask)
    # st.text()
    # converting mask to greyscale
    mask = mask.convert('L')

    # changing the alpha values of the image using the table mask
    im.putalpha(mask)
    
    return im

def OCR_Reader(image):
    '''
        This function takes an image as input and uses pytesseract to read and return the textual content in the image.
    '''
    text_data = pytesseract.image_to_string(image)
    return text_data

def image_reader(image_fname):
    im = Image.open(image_fname)
    return numpy.array(im)


if __name__ == "__main__":
    import sys
    tablenet = ModelConstructor()
    # print(tablenet.summary()) 
    im = image_reader(sys.argv[1])
    print(im.shape)
    im, table_mask, col_mask = predict_masks(im, tablenet)
    table_mask_img = get_mask_image(table_mask)
    col_mask_img = get_mask_image(col_mask)
    print(table_mask_img.shape, col_mask_img.shape)
