import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.compat.v1.InteractiveSession()

# image
image = np.array([[[[1.], [2.], [3.]],
                   [[4.], [5.], [6.]],
                   [[7.], [8.], [9.]]]])
print("image.shape:{}".format(image.shape))
# filter
weight = np.array([[[[1.]], [[1.]]],
                   [[[1.]], [[1.]]]])
print("weight.shape:{}".format(weight.shape))

conv2d = tf.nn.conv2d(image, weight, [1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval(session=sess)
print("conv2d_img.shape:", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 2, i+1)
    plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()
