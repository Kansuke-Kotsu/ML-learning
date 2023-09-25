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
result_name = "result"
# train config # example
epochs = 10
steps_per_epoch = 100
style_weight=1e-2  # x 0.01
content_weight=1e4 # x 10000

# run sequence
func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=1,
    epochs=10,
    style_weight=1e-2,
    content_weight=1e4,
    result_name="results/result_a_1"
    )
  
func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=10,
    epochs=10,
    style_weight=1e-2,
    content_weight=1e4,
    result_name="result_a_10"
    )

func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=100,
    epochs=10,
    style_weight=1e-2,
    content_weight=1e4,
    result_name="result_a_100"
    )

func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=10,
    epochs=10,
    style_weight=1e-3,
    content_weight=1e3,
    result_name="result_b_10"
    )
  
func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=10,
    epochs=10,
    style_weight=1e-4,
    content_weight=1e4,
    result_name="result_c_10"
    )

func. main_logic(
    content_path=content_path, 
    style_path=style_path,
    steps_per_epoch=10,
    epochs=10,
    style_weight=1,
    content_weight=1,
    result_name="result_d_10"
    )

# tf.saved_model.save(model, saved_model_dir)

