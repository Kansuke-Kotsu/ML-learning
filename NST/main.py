import os
import tensorflow as tf

import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time

import func

# define image path
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
# train config
epochs = 1
steps_per_epoch = 10
style_weight=1e-2  # x 0.01
content_weight=1e4 # x 10000

# run sequence
func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    style_weight=style_weight,
    content_weight=content_weight
    )
  

# tf.saved_model.save(model, saved_model_dir)

