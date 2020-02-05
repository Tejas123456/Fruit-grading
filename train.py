import os

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

import pickle

pickle_in = open("X.pickle","rb")
xx = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y1 = pickle.load(pickle_in)
yy = []
for i in y1:
    if(i == 0):
        yy.append([1,0,0])
    if (i == 1):
        yy.append([0, 1, 0])
    if (i == 2):
        yy.append([0, 0, 1])
y = np.array(yy)

xx = xx/255.0
print(xx[20][50][50])
print(yy)
xx = tf.convert_to_tensor(xx)
yy = tf.reshape(yy, [-1,3])
# import driving_data
import model

LOGDIR = './save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.y_, labels=model.y)))
# optimizer = tf.train.AdamOptimizer().minimize(cost)

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

hm_epochs = 10
batch_size = 49


# L2NormConst = 0.001
#
# train_vars = tf.trainable_variables()
#
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) # + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-1).minimize(loss)
sess.run(tf.initialize_all_variables())

for epoch in range(hm_epochs):
    epoch_loss = 0
    for _ in range(int(1465 / batch_size)):
        batch_of_x = sess.run(xx[(_ * batch_size):((_ + 1) * batch_size)])
        batch_of_y = sess.run(yy[(_ * batch_size):((_ + 1) * batch_size)])
        print(np.shape(batch_of_y), len(batch_of_y))
        x11 = sess.run(xx[0:49])
        y11 = sess.run(yy[0:49])
        train_step.run(feed_dict={model.x: batch_of_x, model.y_: batch_of_y})
        c = loss.eval(feed_dict={model.x: x11, model.y_: y11})
        # _, c = sess.run([optimizer, -(cost)], feed_dict={model.x: batch_of_x, model.y_: batch_of_y})
        epoch_loss += c
        print(epoch_loss)
        if epoch % batch_size == 0:

            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)

    print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    correct = tf.equal(tf.argmax(model.y, 1), tf.argmax(model.y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({model.x: x11, model.y_: y11}))