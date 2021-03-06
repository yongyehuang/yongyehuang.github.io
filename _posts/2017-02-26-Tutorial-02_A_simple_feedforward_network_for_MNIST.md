--- 
layout: column_post
title: TensorFlow入门（二）简单前馈网络实现 mnist 分类
column: TensorFlow
description: "实现一个非常简单的两层全连接层来完成MNIST数据的分类问题。"
---

在本教程中，我们来实现一个非常简单的两层全连接网络来完成MNIST数据的分类问题。
输入[-1,28*28], FC1 有 1024 个neurons， FC2 有 10 个neurons。这么简单的一个全连接网络，结果测试准确率达到了 0.98。还是非常棒的！！！


```python
import numpy as np
import tensorflow as tf

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
```

### 1. 导入数据


```python
# 用tensorflow 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
print 'training data shape ', mnist.train.images.shape
print 'training label shape ', mnist.train.labels.shape
```

    training data shape  (55000, 784)
    training label shape  (55000, 10)


### 2.  构建网络


```python
# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# input_layer
X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# FC1
W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X_, W_fc1) + b_fc1)

# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pre = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

```

### 3. 训练和评估


```python
# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
# 2.优化函数：AdamOptimizer, 优化速度要比 GradientOptimizer 快很多
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 3.预测结果评估
#　预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.arg_max(y_, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始运行
sess.run(tf.global_variables_initializer())
# 这大概迭代了不到 10 个 epoch， 训练准确率已经达到了0.98
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    train_step.run(feed_dict={X_: X_batch, y_: y_batch})
    if (i+1) % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={X_: mnist.train.images, y_: mnist.train.labels})
        print "step %d, training acc %g" % (i+1, train_accuracy)
    if (i+1) % 1000 == 0:
        test_accuracy = accuracy.eval(feed_dict={X_: mnist.test.images, y_: mnist.test.labels})
        print "= " * 10, "step %d, testing acc %g" % (i+1, test_accuracy)
```

    step 200, training acc 0.937364
    step 400, training acc 0.965818
    step 600, training acc 0.973364
    step 800, training acc 0.977709
    step 1000, training acc 0.981528
    = = = = = = = = = =  step 1000, testing acc 0.9688
    step 1200, training acc 0.988437
    step 1400, training acc 0.988728
    step 1600, training acc 0.987491
    step 1800, training acc 0.993873
    step 2000, training acc 0.992527
    = = = = = = = = = =  step 2000, testing acc 0.9789
    step 2200, training acc 0.995309
    step 2400, training acc 0.995455
    step 2600, training acc 0.9952
    step 2800, training acc 0.996073
    step 3000, training acc 0.9964
    = = = = = = = = = =  step 3000, testing acc 0.9778
    step 3200, training acc 0.996709
    step 3400, training acc 0.998109
    step 3600, training acc 0.997455
    step 3800, training acc 0.995055
    step 4000, training acc 0.997291
    = = = = = = = = = =  step 4000, testing acc 0.9808
    step 4200, training acc 0.997746
    step 4400, training acc 0.996073
    step 4600, training acc 0.998564
    step 4800, training acc 0.997946
    step 5000, training acc 0.998673
    = = = = = = = = = =  step 5000, testing acc 0.98

本文代码：https://github.com/yongyehuang/Tensorflow-Tutorial

