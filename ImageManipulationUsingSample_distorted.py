#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
# Load image to numpy array.
img = PIL.Image.open(r'C:\Users\Sumanth\Desktop\pic2.jpg')
img.load()
img_array = np.array(img)

# Distort an image by cropping it with a different aspect ratio.
def distorted_random_crop(image,
               aspect_ratio_range=(0.5, 0.8)):
   #Defining a 3D tensor to be later passed to the function  
   cropbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
   sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
       tf.shape(image),
       bounding_boxes=cropbox,
       aspect_ratio_range=aspect_ratio_range)
#begin is a Tensor that has the same type as image_size. 1-D, containing [offset_height, offset_width, 0].
#size is a Tensor that the same type as image_size. 1-D, containing [target_height, target_width, -1]. 
#bboxes is a Tensor of type float32. 3-D with shape [1, 1, 4] containing the distorted bounding box.
   begin, size, bboxes = sample_distorted_bounding_box
   # Crop the image to the specified bounding box.
   cropped_image = tf.slice(img_array, begin, size)
   return cropped_image

PIL.Image.fromarray(distorted_random_crop(img_array).numpy())

