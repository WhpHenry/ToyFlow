# coding: utf-8

import toyflow as tf
# print(tf.DEFAULT_GRAPH)
import numpy as np 
import matplotlib.pyplot as plt

# random data: y = 3*x
input_x = np.linspace(-1, 1, 100)
input_y = input_x*3 + np.random.randn(input_x.shape[0])*0.5

x = tf.Placeholder()
y_ = tf.Placeholder()

w = tf.Variable([[1.0]], name='weight')
b = tf.Variable(0.0, name='threshold')

# linear regression model
y = x*w + b

loss = tf.reduce_sum(tf.square(y-y_))

train_op = tf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {x: np.reshape(input_x, (-1, 1)), y_: np.reshape(input_y, (-1, 1))}
feed_dict = {x: input_x, y_: input_y}
with tf.Session() as sess:
    for step in range(20):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value/len(input_x)
        
        if step % 1 == 0:
            print('step: {}, loss: {}, mse: {}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))

w_value = float(w_value)
max_x, min_x = np.max(input_x), np.min(input_x)
max_y, min_y = w_value*max_x + b_value, w_value*min_x + b_value

plt.plot([max_x, min_x], [max_y, min_y], color='r')
plt.scatter(input_x, input_y)
plt.show()