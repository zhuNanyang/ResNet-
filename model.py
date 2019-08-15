from config import Config
import tensorflow as tf
import math

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tflearn.layers.conv import global_avg_pool
class ResNet_model(Config):
    def __init__(self):

        self.input_x = tf.placeholder(tf.float32, [None, Config.sequence, Config.word_dim], name="input_x")
        self.input_p1 = tf.placeholder(tf.int32, [None, Config.sequence], name="input_p1")

        self.input_p2 = tf.placeholder(tf.int32, [None, Config.sequence], name="input_p2")
        self.input_y = tf.placeholder(tf.float32, [None, Config.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.position_size = 5
        self.filter_sizes = [3]
        self.embedding_size = Config.word_dim
        self.num_filters = 128
        self.num_block = 6
        self.l2_reg_lambda = 0.0001
        self.learning_rate = 0.001

        self.MOVING_AVERAGE_DECAY = 0.9997
        self.RESNET_VARIABLES = "resnet_variables"

        self.BN_DECAY = self.MOVING_AVERAGE_DECAY
        self.UPDATE_OPS_COLLECTION = "renet_update_ops"
        self.BN_EPSILON = 0.001



    def build_model(self):
        with tf.variable_scope("position_embedding"):
            #W_position_1 = tf.get_variable(name='pos_e1_embedding', shape=[62, 5])
            #W_position_2 = tf.get_variable(name="pos_e2_embedding", shape=[62, 5])
            W = tf.Variable(tf.random_uniform([62, 5],
                                              minval=-math.sqrt(6 / (3 * self.position_size + 3 * Config.word_dim)),
                                              maxval=math.sqrt(6 / (3 * self.position_size + 3 * Config.word_dim))), name="W")
            input_x_p1 = tf.nn.embedding_lookup(W, self.input_p1)
            input_x_p2 = tf.nn.embedding_lookup(W, self.input_p2)
            input = tf.concat([self.input_x, input_x_p1, input_x_p2], 2)
            input_expanded = tf.expand_dims(input, -1)
        pooled_outputs = []
        for i, filer_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filer_size):
                filter_shape = [filer_size, self.embedding_size + 2 * self.position_size, 1, self.num_filters]

                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b1")
                conv1 = tf.nn.conv2d(input_expanded,
                                    W1,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                c_ = {"use_bias": True,
                      "is_training": tf.cond(self.dropout_keep_prob < 1.0, lambda :True, lambda: False)}
                conv1 = self.bn(conv1, c_, "first-bn")
                beta1 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='first-swish')

                x1 = tf.nn.bias_add(conv1, b1)
                h1 = x1 * tf.nn.sigmoid(x1 * beta1)
                for i in range(self.num_block):
                    h2 = self.block(self.num_filters, h1, i)
                    h1 = h2 + h1
                pooled = tf.nn.max_pool(
                    h1,
                    ksize=[1, self.sequence - filer_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_avg = tf.nn.avg_pool(
                    h1,
                    ksize=[1, self.sequence - filer_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                pooled_outputs.append(pooled_avg)

        num_filters_total = self.num_filters * 2
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name="hidden_feature")
        with tf.variable_scope("MLP"):
            W2 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b2")
            h3 = tf.nn.relu(tf.nn.xw_plus_b(h_pool_flat, W2, b2))
            W3 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b3")
            h3_2 = tf.nn.relu(tf.nn.xw_plus_b(h3, W3, b3))

        with tf.variable_scope("dropout"):

            h3_3 = tf.nn.dropout(h3_2, self.dropout_keep_prob)
        with tf.variable_scope("output"):
            W3_1 = tf.get_variable(
                "W3_1",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b3_1 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b3_1")
            logits = tf.nn.xw_plus_b(h3_3, W3_1, b3_1, name="logits")
            probability = tf.nn.softmax(logits, 1)
            return logits, probability
    def train_loss(self):
        logits, probability = self.build_model()
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            l2_loss_in = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                weights_list=tf.trainable_variables())
            loss = tf.reduce_mean(losses) + l2_loss_in
            tf.summary.scalar("final_loss", loss)
            return loss, logits, probability

    def train(self):
        loss, logits, probability = self.train_loss()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return logits, probability, loss, train_op

    def block(self, num_filters, h, i, has_se=False):
        W2 = tf.get_variable(
            "W2_" + str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2_" + str(i))
        conv2 = tf.nn.conv2d(
            h,
            W2,
            strides=[1, 1, 1, 1],
            padding="SAME")

        c_ = {'use_bias': True, 'is_training': tf.cond(self.dropout_keep_prob < 1.0, lambda: True, lambda: False)}
        conv2 = self.bn(conv2, c_, str(i) + '-conv2')

        beta2 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-2'.format(i))
        x2 = tf.nn.bias_add(conv2, b2)
        h2 = x2 * tf.nn.sigmoid(x2 * beta2)

        W3 = tf.get_variable(
            "W3_" + str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b3 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b3_" + str(i))
        conv3 = tf.nn.conv2d(
            h2,
            W3,
            strides=[1, 1, 1, 1],
            padding="SAME")

        conv3 = self.bn(conv3, c_, str(i) + '-conv3')

        beta3 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-3'.format(i))
        x3 = tf.nn.bias_add(conv3, b3)
        h3 = x2 * tf.nn.sigmoid(x3 * beta3)

        if has_se:
            h3 = self.Squeeze_excitation_layer(h3, num_filters, 16, 'se-block-' + str(i))

        return h3


    def bn(self, x, c, name):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        if c["use_bias"]:
            bias = self._get_variable("bn_bias_{}".format(name),params_shape,
                                      initializer=tf.zeros_initializer)
            return x + bias
        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable('bn_beta_{}'.format(name),
                                  params_shape,
                                  initializer=tf.zeros_initializer)
        gamma = self._get_variable('bn_gamma_{}'.format(name),
                                   params_shape,
                                   initializer=tf.ones_initializer)
        moving_mean = self._get_variable('bn_moving_mean_{}'.format(name), params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)
        moving_variance = self._get_variable('bn_moving_variance_{}'.format(name),
                                             params_shape,
                                             initializer=tf.ones_initializer,
                                             trainable=False)
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, self.BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.BN_DECAY)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self.BN_EPSILON)
        return x
    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype="float",
                      trainable=True):
        if weight_decay > 0:

            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = self.Global_Average_Pooling(input_x)

            excitation = self.Fully_connected(squeeze, units=out_dim / ratio,
                                              layer_name=layer_name + '_fully_connected1')
            excitation = self.Relu(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim,
                                              layer_name=layer_name + '_fully_connected2')
            excitation = self.Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale
    def Global_Average_Pooling(self, x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Relu(self, x):
        return tf.nn.relu(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def Fully_connected(self, x, units, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=True, units=units)



