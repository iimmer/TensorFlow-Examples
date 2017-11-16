""" Auto Encoder Example.


用tenserflow创建一个自动编码器.把图片压缩成一个小尺寸,再解压.
整个网络共5层:第一层是输入层,神经元数量等于图片像素数量,共784个.
             第二层是压缩层,共有256个神经元.
             第三层进一步压缩,共100个神经元.
             第四层开始解压,还原到256个神经元.
             第五层解压回784个神经元作为像素输出.
这里与识别不同,压缩的输出是与输入相比较的.而识别的输出是与图片标签相比较的.

参考论文:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

链接:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 载入数据,地址在程序根目录的/tmp/data/里
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 训练参数:
learning_rate_01 = 0.01           #学习速率
num_steps = 3000               #迭代次数
batch_size_256 = 256               #单次迭代的图片数量

display_step = 1000            #在控制台打印信息的频率
examples_to_show = 10          #示例图片显示数量

# 网络参数
hidden_L2_256 = 256 # 第一隐藏层特征数量
hidden_L3_128 = 128 # 第二隐藏层特征数量 (潜在维度)
input_784 = 784 # MNIST 数据输入层(img shape: 28*28)

# 计算图输入接口(仅输入图片数据)
X = tf.placeholder("float", [None, input_784])

#将每一层的偏置建立成字典形式.
weights = {
    #每层权重数(输入层没有偏置) = 前一层神经元数 * 该层神经元数
    'encoder_L2': tf.Variable(tf.random_normal([input_784, hidden_L2_256])),
    'encoder_L3': tf.Variable(tf.random_normal([hidden_L2_256, hidden_L3_128])),
    'decoder_L4': tf.Variable(tf.random_normal([hidden_L3_128, hidden_L2_256])),
    'decoder_L5': tf.Variable(tf.random_normal([hidden_L2_256, input_784])),
}

biases = {
    #每一层偏置数(输入层没有偏置) = 该层神经元数
    'encoder_L2': tf.Variable(tf.random_normal([hidden_L2_256])),
    'encoder_L3': tf.Variable(tf.random_normal([hidden_L3_128])),
    'decoder_L4': tf.Variable(tf.random_normal([hidden_L2_256])),
    'decoder_L5': tf.Variable(tf.random_normal([input_784])),
}

# 定义编码器函数
def encoder(x):
    # 编码器第二层输出 = sigmoid(第一层输出 * 第二层权重 + 第二层偏置)
    # 使用sigmoid激活函数
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_L2']),
                                   biases['encoder_L2']))
    # 编码器第三层输出 = sigmoid(第二次输出 * 第三层权重 + 第三层偏置)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_L3']),
                                   biases['encoder_L3']))
    # 返回值是第三层输出
    return layer_2


# 定义解码器函数
def decoder(x):
    # 第四层(解码层)输出 = sigmoid(第三层输出 * 第四层权重 + 第四层偏置)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_L4']),
                                   biases['decoder_L4']))
    # 第五层(解码层)输出 = sigmoid(第四层输出 * 第五层权重 + 第五层偏置)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_L5']),
                                   biases['decoder_L5']))
    # 返回第五层输出.
    return layer_2

# 将编码与解码层组合
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# 预测值  Prediction
y_pred = decoder_op
# 目标值 (即标签Labels) 就是输入值.
y_true = X

# 定义代价函数.loss 是指输入的所有图片像素差的平方的平均数.
# 图片像素差 = 输入图片像素 - 解码图片像素
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate_01).minimize(loss)

# 将初始化所有变量的操作赋值给init
init = tf.global_variables_initializer()

# 开始训练
#   打开一个新回话
with tf.Session() as sess:

    # 初始化所有变量
    sess.run(init)

    # 训练
    for i in range(1, num_steps+1):
        # 数据准备
        # 获取一批数据这里抽了256张,只要用到图片,不用标签,所以返回值的标签位置用_代替
        batch_x, _ = mnist.train.next_batch(batch_size_256)

        # 运行两个操作,一是优化 (反向传播) 另一个是代价 (得到每一次迭代的代价,赋值给l)
        # 操作的输入字典是变量X = batch_x
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # 如果迭代数到达显示步长就显示下代价值.
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # 上一步迭代完后所有权重和偏置变量已优化.用这些优化完成的权重和偏置对测试集数据进行
    # 编码和解码,然后显示几张图片的输入前后对比,并返回一个图片经压缩解压后与原图片的误差值.
    n = 4   
    #将n^2张图片合成一张由n*n张图片构成的大图
    canvas_orig = np.empty((28 * n, 28 * n)) 
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # 取得MNIST测试集中的图片.(4张)
        batch_x, _ = mnist.test.next_batch(n)
        # 对图片进行编码后再解码,返回给g
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # 将原始图像放入容器
        for j in range(n):
            # 将原始图片放入容器
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        #  将经过编码再解码的图片放入容器
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

# 图像打印要放在 with tf.Session as sess: 结构之外.
#test
print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

"""
总结:
    编码器的作用是可以压缩信号源,可以加密信号源.

"""