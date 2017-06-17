# coding:utf-8
import warnings
warnings.filterwarnings("ignore")


from tensorflow.examples.tutorials.mnist import input_data

# 60000行的训练数据集（mnist.train）
# 10000行的测试数据集（mnist.test）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist.train.images 是一个形状为 [60000, 784] 的张量，
# 第一个维度数字用来索引图片，
# 第二个维度数字用来索引每张图片中的像素点, 784=28x28
# 张量里的每个元素，都表示某张图片里的某个像素的灰度值[0,1]

# mnist.train.labels 是一个 [60000, 10] 的数字矩阵。
# one_hot=True: 比如标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。 



import tensorflow as tf

# x是一个占位符，不是特定的值
# None表示第一个维度可以是任意长度
x = tf.placeholder("float", [None, 784])


# Variable表示可以修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# 使用softmax分类器
# tf.matmul表示矩阵乘法
y = tf.nn.softmax(tf.matmul(x, W) + b)


# y_表示正确的标签
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# 计算cross entropy
# reduce_sum计算张量的所有元素的总和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


# learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 初始化操作
init = tf.initialize_all_variables()

# 使用Seesion启动model，并进行初始化
sess = tf.Session()
sess.run(init)

for i in range(10):
	print i
	batch_xs, batch_ys = mnist.train.next_batch(32)
	
	# y_ should be provided, not y
	# otherwise: Shape [-1,10] has negative dimensions
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



# tf.argmax给出tensor某一维的最大值的索引
correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))



	# y_ should be provided, not y
print sess.run(accuracy, 
			feed_dict={x: mnist.test.images, y_: mnist.test.labels})

