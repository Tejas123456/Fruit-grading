import os
import tensorflow as tf
import cv2
import model

print("hello")
import numpy as np

# DATADIR = "E:/"
img = "84.jpg"
IMG_SIZE = 100
img_array = cv2.imread(os.path.join(os.getcwd(),img), 0)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
x = []
x = np.array(new_array).reshape( IMG_SIZE, IMG_SIZE, 1)
print(len(x))
cv2.imshow("dasd",img_array)
cv2.waitKey()
# x = tf.convert_to_tensor(x)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "save\\model.ckpt")
with sess:
    category = model.y.eval(feed_dict={model.x : [x]})
print(category)







