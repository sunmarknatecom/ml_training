import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row[0] for x_row in data]
x2 = [x_row[1] for x_row in data]
y_data = [y_row[2] for y_row in data]

a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b =  tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))
y = a1*x1 + a2*x2 + b
rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))
learning_rate = 0.1
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
            sess.run(gradient_descent)
            if step % 100 == 0:
                    print("Epoch: %.f, RMSE = %.04f, 기울기 a1= %.4f, 기울기 a2 = %.4f, y절편 b = %.4f" % (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))