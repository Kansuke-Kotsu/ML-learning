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

# ランダムな画像の数を指定
num_images = 5  # 例として5枚の画像を表示

# ランダムなインデックスを生成
random_indices = np.random.choice(X_train.shape[0], num_images, replace=False)

# ランダムに選んだ画像を表示
plt.figure(figsize=(12, 3))
for i, index in enumerate(random_indices):
    image = X_train[index]
    plt.subplot(1, num_images, i + 1)
    plt.imshow(image)
    plt.axis('off')

plt.show()