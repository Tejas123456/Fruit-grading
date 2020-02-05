import tensorflow as tf

n_classes = 3
# batch_size = 50


def conv2d(x, W):
    return tf.nn.conv2d(tf.cast(x, tf.float32), W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(tf.cast(x, tf.float32), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_img = x
weights = {
    # 5 x 5 convolution, 1 input image, 32 outputs
    'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'W_fc': tf.Variable(tf.random_normal([25 * 25 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
print(weights['W_conv1'])

biases = {
    'b_conv1': tf.Variable(tf.random_normal([32])),
    'b_conv2': tf.Variable(tf.random_normal([64])),
    'b_fc': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

conv1 = tf.nn.relu(conv2d(x_img, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)
print("svgsfbtnn b", conv2)
print(conv2)
fc = tf.reshape(conv2, [-1, 25 * 25 * 64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
print(fc)
y = tf.nn.relu(tf.matmul(fc, weights['out']) + biases['out'])
print(y_)
print(y)
