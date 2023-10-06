import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import layers
import numpy as np
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import func

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train[y_train.squeeze() == 1]
X_train = (X_train / 127.5) - 1.0

# define var
IMG_SIZE = 32     # input image size, CIFAR-10 is 32x32
BATCH_SIZE = 1  # for training batch size
timesteps = 16    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps
epochs = 1

# main  
t = func.generate_ts(25, timesteps=timesteps)             # random for training data
a, b = func.forward_noise(X_train[:25], t, time_bar=time_bar)
    
model = func.make_model(IMG_SIZE=IMG_SIZE)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
loss_func = tf.keras.losses.MeanAbsoluteError()
model.compile(loss=loss_func, optimizer=optimizer)
func.predict(IMG_SIZE=IMG_SIZE, timesteps=timesteps, model=model)
func.predict_step(IMG_SIZE=IMG_SIZE, timesteps=timesteps, model=model)

for _ in range(epochs):
    func.train(X_train=X_train, BATCH_SIZE=BATCH_SIZE, R=1)
    # reduce learning rate for next training
    model.optimizer.learning_rate = max(0.000001, model.optimizer.learning_rate * 0.9)

    # show result 
    func.predict(IMG_SIZE=IMG_SIZE, timesteps=timesteps, model=model)
    func.predict_step(IMG_SIZE=IMG_SIZE, timesteps=timesteps, model=model)
    plt.show()


    
