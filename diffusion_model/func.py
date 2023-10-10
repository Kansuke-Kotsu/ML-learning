import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import layers
import numpy as np
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from PIL import Image

# normalization image
def cvtImg(img):
    img = img - img.min()
    img = (img / img.max())
    return img.astype(np.float32)

# show image
def show_examples(x):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        img = cvtImg(x[i])
        plt.imshow(img)
        plt.axis('off')

# integrate noise
def forward_noise(x, t, time_bar):
    a = time_bar[t]      # base on t
    b = time_bar[t + 1]  # image for t + 1
    
    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b
    
# generate random vector
def generate_ts(num, timesteps):
    return np.random.randint(0, timesteps, size=num)


def block(x_img, x_ts):
    x_parameter = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_parameter = layers.Activation('relu')(x_parameter)

    time_parameter = layers.Dense(128)(x_ts)
    time_parameter = layers.Activation('relu')(time_parameter)
    time_parameter = layers.Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter
    
    # -----
    x_out = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_out = x_out + x_parameter
    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation('relu')(x_out)
    
    return x_out

def make_model(IMG_SIZE):
    x = x_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='x_input')
    
    x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')
    x_ts = layers.Dense(192)(x_ts)
    x_ts = layers.LayerNormalization()(x_ts)
    x_ts = layers.Activation('relu')(x_ts)
    
    # ----- left ( down ) -----
    x = x32 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x16 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x8 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x4 = block(x, x_ts)
    
    # ----- MLP -----
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, x_ts])
    x = layers.Dense(128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(4 * 4 * 32)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((4, 4, 32))(x)
    
    # ----- right ( up ) -----
    x = layers.Concatenate()([x, x4])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x8])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x16])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x32])
    x = block(x, x_ts)
    
    # ----- output -----
    x = layers.Conv2D(3, kernel_size=1, padding='same')(x)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model


def predict(IMG_SIZE, timesteps, model, x_idx=None):
    x = np.random.normal(size=(32, IMG_SIZE, IMG_SIZE, 3))
    for i in trange(timesteps):
        t = i
        x = model.predict([x, np.full((32), t)], verbose=0)
    #show_examples(x)
    
def predict_step(IMG_SIZE, timesteps, model):
    xs = []
    x = np.random.normal(size=(8, IMG_SIZE, IMG_SIZE, 3))
    for i in trange(timesteps):
        t = i
        x = model.predict([x, np.full((8),  t)], verbose=0)
        if i % 2 == 0:
            xs.append(x[0])
    plt.figure(figsize=(20, 2))
    for i in range(len(xs)):
        plt.subplot(1, len(xs), i+1)
        plt.imshow(cvtImg(xs[i]))
        plt.title(f'{i}')
        plt.axis('off')
    return xs

def train_one(x_img, model, timesteps, time_bar):
    x_ts = generate_ts(len(x_img), timesteps=timesteps)
    x_a, x_b = forward_noise(x_img, x_ts, time_bar=time_bar)
    loss = model.train_on_batch([x_a, x_ts], x_b)
    return loss

def train(X_train, BATCH_SIZE, R, model, timesteps, time_bar):
    bar = trange(R)
    total = 100
    for i in bar:
        for j in range(total):
            x_img = X_train[np.random.randint(len(X_train), size=BATCH_SIZE)]
            loss = train_one(x_img, model=model, timesteps=timesteps, time_bar=time_bar)
            pg = (j / total) * 100
            if j % 5 == 0:
                bar.set_description(f'loss: {loss:.5f}, p: {pg:.2f}%')
 
def save_images_as_png(images, output_dir, epoch):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # imagesは画像のリストです
    for i, image in enumerate(images):
        # 保存するファイルパスを生成
        filename = f"image_{epoch}_{i + 1}.png"
        file_path = os.path.join(output_dir, filename)

        # 画像をPIL Imageに変換
        pil_image = Image.fromarray((image * 255).astype('uint8'))

        # PNGファイルとして保存
        pil_image.save(file_path, format='PNG')
        print(f"Saved {file_path}")
        
def save_models(model_save_path, model):
    # ベースのモデル保存パス
    base_model_save_path = model_save_path
    # ファイル名の初期値
    model_name = 'my_model'
    # 重複しないファイル名を生成
    model_save_path = os.path.join(base_model_save_path, model_name)
    counter = 1
    while os.path.exists(model_save_path):
        model_name = f'my_model_{counter}'
        model_save_path = os.path.join(base_model_save_path, model_name)
        counter += 1

    # モデルを指定したパスに保存
    tf.keras.models.save_model(model, model_save_path)
    print(f"save model as {model_save_path}")
    
def load_models(model_save_path):
    # ベースのモデル保存パス
    base_model_save_path = model_save_path
    # ファイル名の初期値
    model_name = 'my_model_1'
    # 重複しないファイル名を生成
    model_save_path = os.path.join(base_model_save_path, model_name)
    counter = 1
    while os.path.exists(model_save_path):
        model_name = f'my_model_{counter}'
        model_save_path = os.path.join(base_model_save_path, model_name)
        counter += 1
    model_name = f'my_model_{counter - 2}'
    model_save_path = os.path.join(base_model_save_path, model_name)    
    loaded_model = tf.keras.models.load_model(model_save_path)
    print(f"... load {model_name} ...")
    return loaded_model