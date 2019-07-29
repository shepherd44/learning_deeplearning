import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.compat.v1.InteractiveSession()

# image
image = np.array([[[[4.], [3.]],
                   [[2.], [1.]]]], dtype=np.float32)
print("image.shape:{}".format(image.shape))
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[
                      1, 1, 1, 1], padding='SAME')
print(pool.shape)
print(pool.eval(session=sess))