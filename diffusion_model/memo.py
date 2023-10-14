# 画像のサイズを確認
image = X_train[0]
image_shape = image.shape

# define var
IMG_SIZE = image_shape     # input image size, CIFAR-10 is 32x32

# 画像を表示
plt.imshow(image)
plt.axis('off')  # 軸を表示しない
plt.show()

