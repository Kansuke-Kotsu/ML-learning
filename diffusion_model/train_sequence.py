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
BATCH_SIZE = 10  # for training batch size
timesteps = 16    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps
epochs = 0
model_save_path = 'outputs/models'
output_directory = 'outputs/images'

# load model
model = func.load_models(model_save_path=model_save_path)


def predict(IMG_SIZE, timesteps, model, n_images, x_idx=None):
    x = np.random.normal(size=(n_images, IMG_SIZE, IMG_SIZE, 3))
    for i in trange(timesteps):
        t = i
        x = model.predict([x, np.full((n_images), t)], verbose=0)
    return x

def display_images(x_img, x_a, x_b):
    num_images = len(x_img)

    # 画像のサイズと整列方法を設定
    rows = 3
    cols = num_images
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))

    for i in range(num_images):
        # オリジナル画像
        axes[0, i].imshow(x_img[i] * 0.5 + 0.5)  # ノーマライズを元に戻す
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        # ノイズを乗せた画像
        axes[1, i].imshow(x_a[i] * 0.5 + 0.5)  # ノーマライズを元に戻す
        axes[1, i].set_title('Noisy')
        axes[1, i].axis('off')

        # 復元された画像
        axes[2, i].imshow(x_b[i] * 0.5 + 0.5)  # ノーマライズを元に戻す
        axes[2, i].set_title('Restored')
        axes[2, i].axis('off')

    plt.show()



# Training...
x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
x_ts = func.generate_ts(len(x_img), timesteps=timesteps)
x_a, x_b = func.forward_noise(x_img, x_ts, time_bar=time_bar)
x_r = np.random.normal(size=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
x_g = x_r
for i in trange(timesteps):
    t = i
    x_g = model.predict([x_g, np.full((BATCH_SIZE), t)], verbose=0)

# 以下のコードで表示できます
display_images(x_img=x_img, x_a=x_r, x_b=x_g)




    
