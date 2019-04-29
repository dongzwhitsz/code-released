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


class Dongzw():
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
        with tf.variable_scope('dongzw-net'):
            concat = []
            x = self._conv_v0(X, 5, 36, name="x_conv", strides=[1, 2, 2, 1])
            x = self._pool(x, 2, 2, name="x_maxpool")
            concat.append(x)

            with tf.variable_scope('block1'):
                db1 = self._dense_block(x, 3, 64, name='db1')
                concat.append(db1)

                db2_input = tf.concat(concat, axis=3)
                db2 = self._dense_block(db2_input, 3, 64, name='db2')
                concat.append(db2)

                db3_input = tf.concat(concat, axis=3)
                db3 = self._dense_block(db3_input, 3, 64, name='db3')
                concat.append(db3)

                db4_input = tf.concat(concat, axis=3)
                db4 = self._dense_block(db4_input, 3, 64, name='db4')
                concat.append(db4)
            #########################################################
            with tf.variable_scope('block2'):
                db5_input = tf.concat(concat, axis=3)
                db5 = self._dense_block(db5_input, 3, 128, name='db5')
                concat.append(db5)

                db6_input = tf.concat(concat, axis=3)
                db6 = self._dense_block(db6_input, 3, 128, name='db6')
                concat.append(db6)

                db7_input = tf.concat(concat, axis=3)
                db7 = self._dense_block(db7_input, 3, 128, name='db7')
                concat.append(db7)

                db8_input = tf.concat(concat, axis=3)
                db8 = self._dense_block(db8_input, 3, 128, name='db8')
                concat.append(db8)

            with tf.variable_scope('block3'):
                db9_input = tf.concat(concat, axis=3)
                db9 = self._dense_block(db9_input, 3, 256, name='db9')
                concat.append(db9)

                db10_input = tf.concat(concat, axis=3)
                db10 = self._dense_block(db10_input, 3, 256, name='db10')
                concat.append(db10)

                db11_input = tf.concat(concat, axis=3)
                db11 = self._dense_block(db11_input, 3, 256, name='db11')
                concat.append(db11)

                db12_input = tf.concat(concat, axis=3)
                db12 = self._dense_block(db12_input, 3, 256, name='db12')
                concat.append(db12)

                db13_input = tf.concat(concat, axis=3)
                db13 = self._dense_block(db13_input, 3, 256, name='db13')
            #     concat.append(db13)

            # with tf.variable_scope('block4'):
            #     db14_input = tf.concat(concat, axis=3)
            #     db14 = self._dense_block(db14_input, 3, 512, name='db14')
            #     concat.append(db14)

            #     db15_input = tf.concat(concat, axis=3)
            #     db15 = self._dense_block(db15_input, 3, 512, name='db15')
            #     concat.append(db15)

            #     db16_input = tf.concat(concat, axis=3)
            #     db16 = self._dense_block(db16_input, 3, 512, name='db16')
            #     concat.append(db16)

            #     db17_input = tf.concat(concat, axis=3)
            #     db17 = self._dense_block(db17_input, 3, 512, name='db17')
            #     concat.append(db17)

            #     db18_input = tf.concat(concat, axis=3)
            #     db18 = self._dense_block(db18_input, 3, 512, name='db18')
            #     concat.append(db18)
            
            with tf.variable_scope('fully_connect'):
                db19 = self._batch_norm_layer(db13, name="db19_batch_norm")
                db19 = tf.nn.relu(db19)
                pool19 = tf.nn.avg_pool(db19, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool19')

                #######################################################
                fc20 = self._fc(pool19, 512, 'fc20', relu=True, flatten=True, is_bn=True)
                if self.is_training is True:
                    fc20 = tf.nn.dropout(fc20, keep_prob=self.keep_prob)                
                fc_1 = self._fc(fc20, 5, 'fc_1', relu=False, flatten=False, is_bn=False)
                fc_2 = self._fc(fc20, 14, 'fc_2', relu=False, flatten=False, is_bn=False)
                fc_3 = self._fc(fc20, 9, 'fc_3', relu=False, flatten=False, is_bn=False)
                fc_4 = self._fc(fc20, 2019, 'fc_4', relu=False, flatten=False, is_bn=False)

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
            bn = tf.contrib.layers.batch_norm(inputs=layer, decay=0.9, updates_collections=None, is_training=self.is_training)
        return bn
