'''
Data: 2019-5-1 22:27
Writer: dongzw


'''
import tensorflow as tf
'''
hyperparameter:
learning_rate
weights_regularize_lambda
three_class_regularize_lambda
keep_prob
lambda_level1 
lambda_level2 
lambda_level3 
lambda_label
'''


class Top_V1():
    def __init__(self, purpose):
        '''
        purpose: 0 for train, 1 for valid, 2 for test
        '''
        self.weights_regularize_lambda = tf.Variable(0.00001, trainable=False)
        self.three_class_regularize_lambda = tf.Variable(0.00001, trainable=False)
        self.keep_prob = tf.Variable(0.95, trainable=False)
        self.purpose = purpose  
        self.is_training = False
        if self.purpose is 0:
            self.is_training = True

        self.image_input = tf.placeholder(dtype=tf.float32, shape=[None, 180, 180, 3])
        self.level_output = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.label_output = tf.placeholder(dtype=tf.int32, shape=[None])
        self.softmax1, self.softmax2, self.softmax3, self.softmax4 = self._inference(self.image_input)

        if self.purpose is 0 or 1:
            self.loss, self.accuracy = self._loss(self.softmax1, self.softmax2, self.softmax3, self.softmax4, self.level_output, self.label_output)
        
        if self.purpose is 0:
            self.train_op = self._get_optimizer()

    def _inference(self, X):
        with tf.variable_scope('top-net-v1'):
            transform_size = 128
            with tf.variable_scope('data_input'):
                concat = []
                x = self._conv(X, 5, 36, name="x_conv")
                x = self._pool(x, 2, 2, name="x_maxpool")
                concat.append(x)

            with tf.variable_scope('block_1'):
                conv_depth = 128

                block_input1 = x

                block_conv1 = self._conv(block_input1, conv_size=3, conv_depth=conv_depth, name='blk1_conv1')
                block_conv1 = self._conv(block_conv1, conv_size=3, conv_depth=conv_depth, name='blk1_conv2')
                block_conv1 = self._conv(block_conv1, conv_size=3, conv_depth=conv_depth, name='blk1_conv3', strides=[1, 2, 2, 1])
            
            with tf.variable_scope('block_2'):
                conv_depth = 256

                block_input2 = self._conv(x, 3, transform_size, name='tmp_for_blk2', strides=[1, 2, 2, 1])
                block_input2 = tf.concat([block_input2, block_conv1], axis=3)

                block_conv2 = self._conv(block_input2, conv_size=3, conv_depth=conv_depth, name='blk1_conv1')
                block_conv2 = self._conv(block_conv2, conv_size=3, conv_depth=conv_depth, name='blk1_conv2')
                block_conv2 = self._conv(block_conv2, conv_size=3, conv_depth=conv_depth, name='blk1_conv3', strides=[1, 2, 2, 1])
            
            with tf.variable_scope('block_3'):
                conv_depth = 512

                block_input3 = self._conv(block_input2, 3, transform_size, name='tmp_for_blk3', strides=[1, 2, 2, 1])
                block_input3 = tf.concat([block_input3, block_conv2], axis=3)

                block_conv3 = self._conv(block_input3, conv_size=3, conv_depth=conv_depth, name='blk3_conv1')
                block_conv3 = self._conv(block_conv3, conv_size=3, conv_depth=conv_depth, name='blk3_conv2')
                block_conv3 = self._conv(block_conv3, conv_size=3, conv_depth=conv_depth, name='blk3_conv3', strides=[1, 2, 2, 1])
            
            with tf.variable_scope('block_4'):
                conv_depth = 512

                block_input4 =  self._conv(block_input3, 3, transform_size, name='tmp_for_blk4', strides=[1, 2, 2, 1])
                block_input4 = tf.concat([block_input4, block_conv3], axis=3)      

                block_conv4 = self._conv(block_input4, conv_size=3, conv_depth=conv_depth, name='blk4_conv1')
                block_conv4 = self._conv(block_conv4, conv_size=3, conv_depth=conv_depth, name='blk4_conv2')
                block_conv4 = self._conv(block_conv4, conv_size=3, conv_depth=conv_depth, name='blk4_conv3', strides=[1, 2, 2, 1])

            with tf.variable_scope('block_5'):
                conv_depth = 1024

                block_input5 = self._conv(block_input4, 3, transform_size, name='tmp_for_blk5', strides=[1, 2, 2, 1])
                block_input5 = tf.concat([block_input5, block_conv4], axis=3)

                block_conv5 = self._conv(block_input5, conv_size=3, conv_depth=conv_depth, name='blk5_conv1')
                block_conv5 = self._conv(block_conv5, conv_size=3, conv_depth=conv_depth, name='blk5_conv2')
                block_conv5 = self._conv(block_conv5, conv_size=3, conv_depth=conv_depth, name='blk5_conv3')
                
            with tf.variable_scope('fully_connect'):
                pool = tf.nn.avg_pool(block_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

                #######################################################
                fc_tmp = self._fc(pool, 2019 * 2, 'fc_tmp', relu=True, flatten=True, is_bn=True)
                # if self.is_training is True:
                #     fc20 = tf.nn.dropout(fc20, keep_prob=self.keep_prob)        
                # fc_tmp = self._batch_norm_layer(fc_tmp)
                fc_1 = self._fc(fc_tmp, 5, 'fc_1', relu=False, flatten=False, is_bn=False)
                fc_2 = self._fc(fc_tmp, 14, 'fc_2', relu=False, flatten=False, is_bn=False)
                fc_3 = self._fc(fc_tmp, 9, 'fc_3', relu=False, flatten=False, is_bn=False)
                fc_4 = self._fc(fc_tmp, 2019, 'fc_4', relu=False, flatten=False, is_bn=False)

                softmax1 = tf.nn.softmax(fc_1)
                softmax2 = tf.nn.softmax(fc_2)
                softmax3 = tf.nn.softmax(fc_3)
                softmax4 = tf.nn.softmax(fc_4)

        return softmax1, softmax2, softmax3, softmax4

    
    def _loss(self, softmax1, softmax2, softmax3, softmax4, level_logits, label_logits):
        with tf.variable_scope('entropy'):
            label = tf.one_hot(label_logits, 2019)
            level1 = tf.one_hot(level_logits[:, 0], 5)
            level2 = tf.one_hot(level_logits[:, 1], 14)
            level3 = tf.one_hot(level_logits[:, 2], 9)

            ent1 = self._ent(level1, softmax1)
            ent2 = self._ent(level2, softmax2)
            ent3 = self._ent(level3, softmax3)
            ent4 = self._ent(label, softmax4)

            self.lambda_level1 = tf.Variable(0.1, name='lambda_level1', trainable=False)
            self.lambda_level2 = tf.Variable(0.1, name='lambda_level2', trainable=False)
            self.lambda_level3 = tf.Variable(0.1, name='lambda_level3', trainable=False)
            self.lambda_label = tf.Variable(0.7, name='lambda_label', trainable=False)

            ent = self.lambda_level1 * ent1 + self.lambda_level2 * ent2 + self.lambda_level3 * ent3 + self.lambda_label * ent4
        
        with tf.variable_scope('accuracy'):
            softmax_v, softmax_i = tf.math.top_k(softmax4, k=3, sorted=True)
            softmax_i = tf.cast(softmax_i, tf.int32)
            self.softmax_v = softmax_v
            self.softmax_i = softmax_i
            tf.summary.histogram('softmax_v', self.softmax_v)
            three_class_regularizer = (tf.reduce_sum(softmax_v, axis=1) - 1)
            three_class_regularizer = self.three_class_regularize_lambda * tf.reduce_mean(three_class_regularizer * three_class_regularizer)
            tf.add_to_collection('three_class_regularizer', three_class_regularizer)
            _la = tf.stack([label_logits, label_logits, label_logits], axis=1)
            _equal = tf.cast(tf.equal(softmax_i, _la), tf.float32)
            accuracy = tf.reduce_mean(tf.reduce_sum(_equal, axis=1))   

            argmax = tf.cast(tf.math.argmax(self.softmax4, axis=1), tf.int32)
            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(argmax, label_logits), tf.float32)
            )
            tf.summary.scalar('acc', self.acc)

        with tf.variable_scope('loss'):
            regularization = tf.add_n(tf.get_collection("fc_regularizer")) +  tf.add_n(tf.get_collection('three_class_regularizer'))
            loss = ent + regularization
        return loss, accuracy

    def _ent(self, label, softmax):
        ent = -tf.reduce_mean(
            tf.reduce_sum(label * tf.log(tf.clip_by_value(softmax, 1e-7, 1)), axis=1)
            )
        return ent
    
    def _get_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.Variable(0.01, trainable=False, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        tf.summary.scalar('learning_rate', self.learning_rate)
        return train_op

    def _dense_block(self, input_layer, conv_size, conv_depth, name, k=32):
        with tf.variable_scope(name):
            bn = self._batch_norm_layer(input_layer, name=name+'1x1_bn')
            relu = tf.nn.relu(bn, name=name+'1x1_relu')
            conv = self._conv_v0(relu, 1, 4*k, name=name+'1x1_conv')
            bn = self._batch_norm_layer(conv, name=name+'3x3_bn')
            relu = tf.nn.relu(bn, name=name+'3x3_relu')
            conv = self._conv_v0(relu, conv_size, conv_depth, name=name+'3x3_conv')
            return conv

    def _dense_block_V2(self, input_layer, conv_size, conv_depth, name, k=32):
        with tf.variable_scope(name):
            bn = self._batch_norm_layer(input_layer, name=name+'1x1_bn')
            relu = tf.nn.relu(bn, name=name+'1x1_relu')
            conv = self._conv_v0(relu, 1, 4*k, name=name+'1x1_conv')
            bn = self._batch_norm_layer(conv, name=name+'3x3_bn')
            relu = tf.nn.relu(bn, name=name+'3x3_relu')
            conv = self._conv_v0(relu, conv_size, conv_depth, strides=[1, 2, 2, 1], name=name+'3x3_conv')
            return conv

    def _conv_v0(self, x, conv_size, conv_depth, name, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            conv_channels = x.get_shape().as_list()[-1]
            filter_weights = tf.get_variable(
                'filter_weights', shape=[conv_size, conv_size, conv_channels, conv_depth],
                initializer=initializer
            )
            biases = tf.get_variable('biases', shape=[conv_depth], initializer=tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(input=x, filter=filter_weights, strides=strides, padding=padding)
            conv = tf.nn.bias_add(conv, bias=biases)
            return conv

    def _conv(self, x, conv_size, conv_depth, name, strides=[1, 1, 1, 1], padding='SAME', relu=True):
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            conv_channels = x.get_shape().as_list()[-1]
            filter_weights = tf.get_variable(
                'filter_weights', shape=[conv_size, conv_size, conv_channels, conv_depth],
                initializer=initializer
            )
            biases = tf.get_variable('biases', shape=[conv_depth], initializer=tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(input=x, filter=filter_weights, strides=strides, padding=padding)
            conv = tf.nn.bias_add(conv, bias=biases)
            conv = self._batch_norm_layer(conv, name='bn')
        if relu is True:
            conv = tf.nn.relu(conv)
        return conv

    def _pool(self, x, filter_size, stride_size, name, padding='SAME'):
        with tf.variable_scope(name):
            pool = tf.nn.max_pool(
                x, ksize=[1, filter_size, filter_size, 1],
                strides=[1, stride_size, stride_size, 1],
                padding=padding
            )
        return pool

    def _fc(self, x, num_out, name, relu=True, flatten=False, is_bn=True, is_regularized=True):
        x_shape = x.get_shape()
        num_in = x_shape[-1]
        if flatten is True:
            num_in = x_shape[1] * x_shape[2] * x_shape[3]
            x = tf.reshape(x, shape=[-1, num_in])

        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:
            initializer = tf.contrib.layers.variance_scaling_initializer()
            weights = tf.get_variable('weights', shape=[num_in, num_out], initializer=initializer, trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True, initializer=tf.constant_initializer(0.01))

            if is_regularized is True:
                tf.add_to_collection("fc_regularizer", tf.contrib.layers.l2_regularizer(self.weights_regularize_lambda)(weights))
            fc = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            if is_bn is True:
                fc = self._batch_norm_layer(fc, name=name+'_bn')
        if relu:
            fc = tf.nn.relu(fc)
        return fc

    def _batch_norm_layer(self, layer,  name='batch_norm'):
        with tf.variable_scope(name):
            bn = tf.contrib.layers.batch_norm(inputs=layer, decay=0.99, epsilon=0.0001, updates_collections=None, is_training=self.is_training)
        return bn
