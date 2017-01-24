import tensorflow as tf  
import numpy as np
import pdb
np.random.seed(1006)
from tensorflow.python.training import moving_averages

nObs = 1000
decay = 0.2

def varSummary(variable):
	with tf.name_scope("summary"):
		mean = tf.reduce_mean(variable)
		tf.summary.scalar('mean',mean)
		std = tf.sqrt(tf.reduce_mean(tf.square(variable-mean)))
		tf.summary.scalar('std',std)
		tf.summary.histogram('mean',mean)
		tf.summary.histogram('std',std)


with tf.variable_scope("input"):
	X = tf.placeholder(tf.float32,[nObs,1])
	varSummary(X)
with tf.variable_scope("ma"):
	mean, variance = tf.nn.moments(X, [0], name='moments')
	moving_mean = tf.get_variable("mean_ma",[1],tf.float32,initializer=tf.constant_initializer(1.0,tf.float32),trainable=False)
	moving_variance = tf.get_variable("var_ma",[1],tf.float32,initializer=tf.constant_initializer(1.0,tf.float32),trainable=False)
	moving_mean = moving_averages.assign_moving_average( moving_mean, mean, decay)
	moving_variance = moving_averages.assign_moving_average( moving_variance, variance, decay)


merged = tf.summary.merge_all()
with tf.Session() as sess:
	writer = tf.summary.FileWriter("varEg/",sess.graph)
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()
	for i in xrange(5):
		mu = i
		std = 0.3+i*1.1
		x = np.random.normal(loc = mu, scale = std, size=nObs).reshape(nObs,1)
		summary, meanC_, varianceC_, mean_,var_ = sess.run([merged,mean,variance,moving_mean,moving_variance],feed_dict={X:x})
		print("Mean = {0},{1}, var = {2},{3} ".format(meanC_,mean_,varianceC_,var_))
		writer.add_summary(summary,i)

