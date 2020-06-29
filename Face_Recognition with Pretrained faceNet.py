#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.set_printoptions(threshold=np.nan)


# In[5]:


FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# In[6]:


print("Total Params:", FRmodel.count_params())


# In[7]:


def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist =  tf.reduce_sum(tf.square((anchor-positive)), axis=-1)
    neg_dist =  tf.reduce_sum(tf.square((anchor-negative)), axis=-1)
    basic_loss = alpha + pos_dist-neg_dist
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


# In[8]:


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))


# In[9]:


FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


# In[10]:


database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


# In[11]:


# GRADED FUNCTION: verify

def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    a = np.linalg.norm((encoding-database[identity]))
    if a<0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False    
    return a, door_open


# In[12]:


verify("images/camera_0.jpg", "younes", database, FRmodel)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **It's younes, welcome in!**
#         </td>
#         <td>
#            (0.65939283, True)
#         </td>
#     </tr>
# 
# </table>

# In[13]:


verify("images/camera_2.jpg", "kian", database, FRmodel)


# In[19]:


def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 1000
    for (name, db_enc) in database.items():
        dist = np.linalg.norm((encoding-db_enc))
        if dist<min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


# In[20]:


who_is_it("images/camera_0.jpg", database, FRmodel)

