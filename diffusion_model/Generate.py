import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from keras import layers
import numpy as np
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
import func

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train[y_train.squeeze() == 1]
X_train = (X_train / 127.5) - 1.0

# define var
IMG_SIZE = 32     # input image size, CIFAR-10 is 32x32
BATCH_SIZE = 128  # for training batch size
timesteps = 16    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps
epochs = 0
model_save_path = 'outputs/models'
output_directory = 'outputs/images'

# load model
model = func.load_models(model_save_path=model_save_path)

# generate imeges
xs = []
for i in range (1):
    x = func.predict(IMG_SIZE=IMG_SIZE, timesteps=timesteps, model=model)
    func.save_images_as_png(images=x, output_dir=output_directory, epoch=0)

'''
# 画像のインデックスを指定
image_index = 0

# X_train から画像を取得
image = X_train[image_index]

# 画像のサイズを確認
image_shape = image.shape
print("画像のサイズ:", image_shape)

# 画像を表示
plt.imshow(image)
plt.axis('off')  # 軸を表示しない
plt.show()
'''

