import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Activation, Reshape, Dropout, MaxPooling2D, BatchNormalization, Conv2D
from keras import backend as K
from keras.models import Model
import numpy as np
import re

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        """with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        """
        tf.summary.histogram('histogram', var)

def tf_get_uninitialized_variables(sess):
    '''A bit of a hack from
    https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
    to get a list of all uninitialized Variable objects from the
    graph
    '''

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars

regex = re.compile('[^A-Za-z0-9_.]')

difficult_indices = [0, 3, 13, 21, 32, 42, 45, 52, 61, 62, 65, 68, 70, 84, 95]
easy_indices = [4, 5, 8, 10, 11, 23, 28, 43, 56, 66, 67, 69, 71, 76, 89]
difficult_indices2 = [109, 110, 146, 332, 158, 338, 192, 194, 210, 216, 248, 264, 274, 280, 298]
easy_indices2 = [100, 101, 102, 114, 121, 139, 140, 142, 150, 176, 201, 205, 206, 207, 214]

tf.reset_default_graph()
sess = tf.Session()
K.set_session(sess)

with tf.name_scope('inputs'):
    x = Input(shape = (784,), name='x-input')
    y_ = Input(shape = (10,), name='y-input')

with tf.name_scope('layer1'):
    xx = Dense(124, kernel_initializer='he_normal', activation='relu')(x)

with tf.name_scope('layer2'):
    xx = Dense(512, kernel_initializer='he_normal', activation='relu')(xx)

with tf.name_scope('layer3'):
    xx = Dense(64, kernel_initializer='he_normal', activation='relu')(xx)

with tf.name_scope('layer4'):
    xx = Dense(32, kernel_initializer='he_normal', activation='relu')(xx)

with tf.name_scope('prob'):
    logits = Dense(10, kernel_initializer='he_normal')(xx)

with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
    loss = tf.reduce_mean(diff, name = 'loss')
    tf.summary.scalar('loss', loss)

model = Model(input=x, output=logits)

opt = tf.train.AdamOptimizer(0.0001)
grads_and_vars = opt.compute_gradients(loss, model.trainable_weights, gate_gradients=tf.train.Optimizer.GATE_GRAPH)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

with tf.name_scope('grads'):
    for grad, var in grads_and_vars:
        varname = regex.sub('', var.name)
        with tf.name_scope(varname) as scope:
            grad_norm = tf.norm(grad)
            variable_summaries(grad_norm)

train_step = opt.apply_gradients(grads_and_vars)

with tf.name_scope('accuracy') as scope:
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1), name = 'correct_prediction')
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name='accuracy')

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

    
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/train',sess.graph)
easy_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/easy')
hard_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/hard')
test_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/test')
easy2_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/easy2')
hard2_writer = tf.summary.FileWriter('tmp/tensorflow/mnist/logs/model_igrads_v2/mnist_with_summaries' + '/hard2')

uninitialized_vars = tf_get_uninitialized_variables(sess)
init_missed_vars = tf.variables_initializer(uninitialized_vars, 'init_missed_vars')
sess.run(init_missed_vars)

for i in range(15000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.train.images[difficult_indices], y_: mnist.train.labels[difficult_indices]})
        hard_writer.add_summary(summary, i)
        print("step %s, difficult accuracy %s"%(i, test_accuracy))
        
        summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.train.images[easy_indices], y_: mnist.train.labels[easy_indices]})
        easy_writer.add_summary(summary, i)
        print("step %s, easy accuracy %s"%(i, test_accuracy))

        summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.train.images[difficult_indices2], y_: mnist.train.labels[difficult_indices2]})
        hard2_writer.add_summary(summary, i)
        print("step %s, difficult2 accuracy %s"%(i, test_accuracy))
        
        summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.train.images[easy_indices2], y_: mnist.train.labels[easy_indices2]})
        easy2_writer.add_summary(summary, i)
        print("step %s, easy2 accuracy %s"%(i, test_accuracy))
        
        summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_writer.add_summary(summary, i)
        print("step %s, test accuracy %s"%(i, test_accuracy))
    else:
        summary, _ = sess.run([merged, train_step], feed_dict={x:batch[0], y_: batch[1]})
        train_writer.add_summary(summary, i)
print("Final test accuracy %s"% (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
