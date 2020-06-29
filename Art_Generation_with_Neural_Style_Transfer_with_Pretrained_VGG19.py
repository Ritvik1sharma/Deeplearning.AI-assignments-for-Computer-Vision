#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)


# In[3]:


content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);


# In[24]:


def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=(m, -1, n_C))
    a_G_unrolled = tf.reshape(a_G , shape=(m, -1, n_C))
    J_content = tf.reduce_sum((tf.multiply((a_C_unrolled-a_G_unrolled), (a_C_unrolled-a_G_unrolled),name=None)), axis=None,name=None)/(4*n_H*n_W*n_C)
    return J_content


# In[25]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))


# In[26]:


style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image);


# In[43]:



def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A, perm=None, name='transpose'),name=None)
    return GA


# In[63]:


# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S =  tf.reshape(tf.transpose(a_S, perm = (3, 1, 2, 0)), shape=(n_C, -1))
    a_G = tf.reshape(tf.transpose(a_G, perm = (3, 1, 2, 0)), shape = (n_C, -1))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer =  tf.reduce_sum((tf.multiply((GG-GS), (GG-GS),name=None)), axis=None,name=None)/(4*n_H*n_H*n_C*n_C*n_W*n_W)
    
    ### END CODE HERE ###
    
    return J_style_layer


# In[44]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = \n" + str(GA.eval()))


# In[64]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J_style_layer**
#         </td>
#         <td>
#            9.19028
#         </td>
#     </tr>
# 
# </table>

# In[65]:


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# In[66]:


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style


# In[67]:


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content+ beta*J_style
    return J


# In[68]:


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


# In[69]:


# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()


# In[70]:


content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)


# In[71]:


style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)


# In[72]:


generated_image = generate_noise_image(content_image)
imshow(generated_image[0]);


# #### Load pre-trained VGG19 model
# Next, as explained in part (2), let's load the VGG19 model.

# In[73]:


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# In[74]:


# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)


# In[75]:


# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


# In[77]:


J = total_cost(J_content, J_style, 10, 40)


# In[78]:


# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)


# In[81]:


def model_nn(sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model["input"].assign(input_image))
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model["input"])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


# In[ ]:


model_nn(sess, generated_image)

