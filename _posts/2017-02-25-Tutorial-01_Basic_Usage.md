--- 
layout: column_post
title: TensorFlow入门（一）基本用法
column: TensorFlow
description: "介绍 TensorFlow 的基本使用方法。"
---

本例子主要是按照 tensorflow的中文文档来学习 tensorflow 的基本用法。按照文档说明，主要存在的一些问题：
- 1.就是 Session() 和 InteractiveSession() 的用法。后者用 Tensor.eval() 和 Operation.run() 来替代了 Session.run(). 其中更多的是用 Tensor.eval()，所有的表达式都可以看作是 Tensor. 
- 2.另外，tf的表达式中所有的变量或者是常量都应该是 tf 的类型。
- 3.只要是声明了变量，就得用 sess.run(tf.global_variables_initializer()) 或者 x.initializer.run() 方法来初始化才能用。

### 例一：平面拟合
通过本例可以看到机器学习的一个通用过程：1.准备数据  -> 2.构造模型（设置求解目标函数）  -> 3.求解模型


```python
import tensorflow as tf
import numpy as np

# 1.准备数据：使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 2.构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 3.求解模型
# 设置损失函数：误差的均方差
loss = tf.reduce_mean(tf.square(y - y_data))
# 选择梯度下降的方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代的目标：最小化损失函数
train = optimizer.minimize(loss)


############################################################
# 以下是用 tf 来解决上面的任务
# 1.初始化变量：tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.global_variables_initializer()


# 设置tensorflow对GPU的使用按需分配
config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 2.启动图 (graph)
sess = tf.Session(config=config)
sess.run(init)

# 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
```

    0 [[ 0.27467242  0.81889796]] [-0.13746099]
    20 [[ 0.1619305   0.39317462]] [ 0.18206716]
    40 [[ 0.11901411  0.25831661]] [ 0.2642329]
    60 [[ 0.10580806  0.21761954]] [ 0.28916073]
    80 [[ 0.10176832  0.20532639]] [ 0.29671678]
    100 [[ 0.10053726  0.20161074]] [ 0.29900584]
    120 [[ 0.100163    0.20048723]] [ 0.29969904]
    140 [[ 0.10004941  0.20014738]] [ 0.29990891]
    160 [[ 0.10001497  0.20004457]] [ 0.29997244]
    180 [[ 0.10000452  0.20001349]] [ 0.29999167]
    200 [[ 0.10000138  0.2000041 ]] [ 0.29999748]


### 例二：两个数求和


```python
input1 = tf.constant(2.0)
input2 = tf.constant(3.0)
input3 = tf.constant(5.0)

intermd = tf.add(input1, input2)
mul = tf.multiply(input2, input3)

with tf.Session() as sess:
    result = sess.run([mul, intermd])  # 一次执行多个op
    print result
    print type(result)
    print type(result[0])   
```

    [15.0, 5.0]
    <type 'list'>
    <type 'numpy.float32'>


# 1.变量，常量
1.1 用 tensorflow 实现计数器，主要是设计了 在循环中调用加法实现计数


```python
# 创建变量，初始化为0
state = tf.Variable(0, name="counter")

# 创建一个 op , 其作用是时 state 增加 1
one = tf.constant(1) # 直接用 1 也就行了
new_value = tf.add(state, 1)
update = tf.assign(state, new_value)


# 启动图之后， 运行 update op
with tf.Session() as sess:
    # 创建好图之后，变量必须经过‘初始化’ 
    sess.run(tf.global_variables_initializer())
    # 查看state的初始化值
    print sess.run(state)
    for _ in range(3):
        sess.run(update)  # 这样子每一次运行state 都还是1
        print sess.run(state)
```

    0
    1
    2
    3


1.2 用 tf 来实现对一组数求和，再计算平均


```python
h_sum = tf.Variable(0.0, dtype=tf.float32)
# h_vec = tf.random_normal(shape=([10]))
h_vec = tf.constant([1.0,2.0,3.0,4.0])
# 把 h_vec 的每个元素加到 h_sum 中，然后再除以 10 来计算平均值
# 待添加的数
h_add = tf.placeholder(tf.float32)
# 添加之后的值
h_new = tf.add(h_sum, h_add)
# 更新 h_new 的 op
update = tf.assign(h_sum, h_new)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 查看原始值
    print 's_sum =', sess.run(h_sum)
    print "vec = ", sess.run(h_vec)
    
    # 循环添加
    for _ in range(4):
        sess.run(update, feed_dict={h_add: sess.run(h_vec[_])})
        print 'h_sum =', sess.run(h_sum)
    
#     print 'the mean is ', sess.run(sess.run(h_sum) / 4)  # 这样写 4  是错误的， 必须转为 tf 变量或者常量
    print 'the mean is ', sess.run(sess.run(h_sum) / tf.constant(4.0))
```

     s_sum = 0.0
    vec =  [ 1.  2.  3.  4.]
    h_sum = 1.0
    h_sum = 3.0
    h_sum = 6.0
    h_sum = 10.0
    the mean is  2.5



```python
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# 如果不是 assign() 重新赋值的话，每一次 sess.run()都会把 state再次初始化为 0.0
state = tf.Variable(0.0, tf.float32)
# 通过 assign 操作来改变state的值。
add_op = tf.assign(state, state+1)

sess.run(tf.global_variables_initializer())
print 'init state ', sess.run(state)
for _ in xrange(3):
    sess.run(add_op)
    print sess.run(state)
```

    init state  0.0
    1.0
    2.0
    3.0



```python
# 在函数内部用 assign 不会改变外边的值呀
```


```python
def chang_W(W1):
    tf.assign(W1, [1.1, 1.2,1.3])
    
W = tf.get_variable('Weights', initializer=[0.2, 0.3, 0.4])
sess.run(tf.global_variables_initializer())
print 'THE INIT W IS ', sess.run(W)
chang_W(W)
print 'AFTER RUNNING THE FUNC ', sess.run(W)
```

    THE INIT W IS  [ 0.2         0.30000001  0.40000001]
    AFTER RUNNING THE FUNC  [ 0.2         0.30000001  0.40000001]


#   2. InteractiveSession() 的用法
InteractiveSession() 主要是避免 Session（会话）被一个变量持有


```python
a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b

# 下面的两种情况是等价的
with tf.Session():
    print c.eval()
    
sess = tf.InteractiveSession()
print c.eval()
sess.close()
```

    3.0
    3.0



```python
a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.Variable(3.0)
d = a + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

###################
# 这样写是错误的
# print a.run()  
# print d.run()

####################

# 这样才是正确的
print a.eval()   
print d.eval()

# run() 方法主要用来
x = tf.Variable(1.2)
# print x.eval()  # 还没初始化，不能用
x.initializer.run()  # x.initializer 就是一个初始化的 op， op才调用run() 方法
print x.eval()

sess.close()
```

    1.0
    3.0
    1.2


#### 怎样使用 tf.InteractiveSession() 来完成上面 1.2 中 求和 、平均 的操作呢?


```python
h_sum = tf.Variable(0.0, dtype=tf.float32)
# h_vec = tf.random_normal(shape=([10]))
h_vec = tf.constant([1.0,2.0,3.0,4.0])
# 把 h_vec 的每个元素加到 h_sum 中，然后再除以 10 来计算平均值
# 待添加的数
h_add = tf.placeholder(tf.float32)
# 添加之后的值
h_new = tf.add(h_sum, h_add)
# 更新 h_new 的 op
update = tf.assign(h_sum, h_new)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print 's_sum =', h_sum.eval()
print "vec = ", h_vec.eval()
print "vec = ", h_vec[0].eval()


for _ in range(4):
    update.eval(feed_dict={h_add: h_vec[_].eval()})
    print 'h_sum =', h_sum.eval()
sess.close()
```

    s_sum = 0.0
    vec =  [ 1.  2.  3.  4.]
    vec =  1.0
    h_sum = 1.0
    h_sum = 3.0
    h_sum = 6.0
    h_sum = 10.0


# 3.使用feed来对变量赋值
这些需要用到feed来赋值的操作可以通过tf.placeholder()说明，以创建占位符。
<br/> 下面的例子中可以看出 session.run([output], ...) 和 session.run(output, ...) 的区别。前者输出了 output 的类型等详细信息，后者只输出简单结果。


```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.0], input2:[2.0]})
```

    [array([ 14.], dtype=float32)]



```python
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:[7.0], input2:[2.0]})
    print type(result)
    print result
```

    <type 'numpy.ndarray'>
    [ 14.]



```python
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:7.0, input2:2.0})
    print type(result)
    print result
```

    <type 'numpy.float32'>
    14.0



```python
with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.0, 3.0], input2:[2.0, 1.0]})
```

    [array([ 14.,   3.], dtype=float32)]



```python
with tf.Session() as sess:
    print sess.run(output, feed_dict={input1:[7.0, 3.0], input2:[2.0, 1.0]})
```

    [ 14.   3.]


# 4. 矩阵操作
tensorflow 主要的用途就是实现机器学习算法（特别是神经网络），所以跟掌握 numpy 一样，掌握 TensorFlow 的矩阵操作非常重要。


```python
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```


```python
tf.reduce_max?
```


```python
# 1. 内积
mat1 = tf.Variable(tf.random_uniform([3,4], minval=0, maxval=10, dtype=tf.int32)) # default [0,1)
mat2 = tf.Variable(tf.random_uniform([4,2], minval=0, maxval=10, dtype=tf.int32)) # default [0,1)
mat3 = tf.matmul(mat1, mat2)
init = tf.global_variables_initializer()
sess.run(init)
```


```python
m1 = sess.run(mat1)
print 'm1=\n', m1
m2 = sess.run(mat2)
print 'm2=\n',m2
m3 = sess.run(mat3)
print 'm3=\n',m3
```

    m1=
    [[3 2 3 5]
     [2 8 4 9]
     [5 4 5 9]]
    m2=
    [[0 1]
     [4 7]
     [3 0]
     [6 7]]
    m3=
    [[ 47  52]
     [ 98 121]
     [ 85  96]]



```python
# 2. reduce 操作
sum_all = tf.reduce_sum(m3)  # 不提供 axis ，则对整个矩阵求和
sum_0 = tf.reduce_sum(m3, axis=0) # axis 是哪个维度，哪个维度的长度就变
sum_1 = tf.reduce_sum(m3, reduction_indices=1)  # reduction_indices 是 axis 的旧名字
sess.run(init)
```


```python
s_all = sess.run(sum_all)
print 's_all=', s_all
s_0 = sess.run(sum_0)
print 's_0=', s_0
s_1 = sess.run(sum_1)
print 's_1=', s_1
```

    s_all= 499
    s_0= [230 269]
    s_1= [ 99 219 181]

