""" Convolutional Neural Network.

用tensorflow 底层基本函数构建卷积神经网络.
下文所提到的节点是指计算图上的各个node

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 设置训练参数
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# 赋值神经网络参数
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# 卷积操作封装函数 卷积核W的输出通道决定了返回值的深度.卷积核移动的跨度(strides)
# ,移动方式(padding)和输入值x的尺寸 三者决定了返回值的尺寸.
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 用池化层降低维度.
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# 建立神经网络计算图
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # 改变输入数据维度.
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第一个卷积层输出conv1值是调用conv2d卷积函数的返回值
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # 对第一个卷积层输出值进行池化.
    conv1 = maxpool2d(conv1, k=2)

    # 第二个卷积层输出值conv2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # 对第二个卷积层输出值进行池化
    conv2 = maxpool2d(conv2, k=2)

    # 将conv2池化后改变维度.将一张图片的数据拉成一维数组.
    # 送入全链层.全链层含一个隐藏层,一个输出层.输入层是卷积层conv2
    # 隐藏层
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # 隐藏层随机丢弃一些神经元,丢弃比例(1-dropout).
    fc1 = tf.nn.dropout(fc1, dropout)

    # 输出层
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 第一个卷积核权重 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 第二个卷积核权重 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 全链隐藏层权重,1024个神经元, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 全链输出层权重10个神经元 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    # 第一卷积层偏置
    'bc1': tf.Variable(tf.random_normal([32])),
    # 第二卷积层偏置
    'bc2': tf.Variable(tf.random_normal([64])),
    # 全链隐藏层偏置
    'bd1': tf.Variable(tf.random_normal([1024])),
    # 输出层偏置
    'out': tf.Variable(tf.random_normal([num_classes]))
}


"""
到此为止所有定义完成.
下面开始创建计算图中各个节点.
"""

# 创建数据图的输入节点.
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# 创建神经网络输出节点Logits,其操作是调用上文构建的神经网络模型函数conv_net
# 维度:[batchsize,10]
logits = conv_net(X, weights, biases, keep_prob)

# 建立损失函数节点loss_op和梯度下降优化操作节点train_op.
# 求出买张图片输出值logits的柔性最大值.在与标签的值组成参数对计算交叉熵.
# 维度:[batchsize,1]
mm = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)

# 计算损失函数,即在样本空间(batchsize)内降维.把所有输入样本的交叉熵计算结果都相加
# 维度[1]
loss_op = tf.reduce_mean(mm)

# 定义优化器节点.使用adam优化器.学习步长等于learning_rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 建立优化操作节点,告诉优化器要优化的对象是损失函数.
train_op = optimizer.minimize(loss_op)


# 模型评估:
# 1.创建节点prediction 计算网络输出的柔性最大值.维度[batchsize,10]
prediction = tf.nn.softmax(logits)
# 2.创建比较节点将较输出的柔性最大值与标签的值进行比较.维度[batchsize,1]
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# 3.将上一步返回的bool值转为float后求平均数,等效于得到正确率百分比.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 创建初始化变量节点.
init = tf.global_variables_initializer()


# 开始训练网络:
# 1.用with创建一个会话
with tf.Session() as sess:

    # 2.在会话中先初始化所有变量.
    sess.run(init)

    # 3.按迭代次数num_steps中设置的参数进行迭代计算
    for step in range(1, num_steps + 1):
        # 获取一批训练数据,值batch_x,标签batch_y 维度分别为[batchsize,786],[batchsize,10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 运行优化器节点.
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        # 如果迭代次数满足一下条件:
        if step % display_step == 0 or step == 1:
            # 计算一次损失函数节点和精度统计节点.并把值传回给python变量loss 和acc
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            # 在python中打印结果.
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
    # 迭代完成,打印完成提示
    print("Optimization Finished!")
    # 在测试数据集上求出最终的精度,并打印.
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
