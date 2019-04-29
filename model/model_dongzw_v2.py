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


class Dongzw_V2():
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
        with tf.variable_scope('dongzw_v2_net'):
            conv1 = self._conv(X, 3, 64, 'conv1', relu=True)
            pool2 = self._pool(conv1, 2, 2, 'pool2')

            with tf.variable_scope('block1'):
                conv3 = self._conv(pool2, 3, 64, 'conv3', relu=True)
                conv4 = self._conv(conv3, 3, 64, 'conv4', relu=False)
                conv4 = tf.nn.relu(conv4 + pool2)

            with tf.variable_scope('block2'):
                conv5 = self._conv(conv4, 3, 64, 'conv5', relu=True)
                conv6 = self._conv(conv5, 3, 64, 'conv6', relu=False)
                conv6 = tf.nn.relu(conv6 + conv4)
            
            with tf.variable_scope('block3'):
                conv7 = self._conv(conv6, 3, 64, 'conv7', relu=True)
                conv8 = self._conv(conv7, 3, 64, 'conv8', relu=False)
                conv8 = tf.nn.relu(conv8 + conv6)
        #######################################################
            with tf.variable_scope('block4'):
                conv9 = self._conv(conv8, 3, 128, 'conv9', relu=True)
                conv10 = self._conv(conv9, 3, 128, 'conv10', relu=False)
                conv10 = tf.nn.relu(conv10 + self._conv(conv8, 1, 128, 'conv10_tmp'))

            with tf.variable_scope('block5'):
                conv11 = self._conv(conv10, 3, 128, 'conv11', relu=True)
                conv12 = self._conv(conv11, 3, 128, 'conv12', relu=False)
                conv12 = tf.nn.relu(conv12 + conv10)

            with tf.variable_scope('block6'):
                conv13 = self._conv(conv12, 3, 128, 'conv13', relu=True)
                conv14 = self._conv(conv13, 3, 128, 'conv14', relu=False)
                conv14 = tf.nn.relu(conv14 + conv12)
        ###############################################
            # with tf.variable_scope('block7'):
            #     conv15 = self._conv(conv14, 3, 256, 'conv15', relu=True)
            #     conv16 = self._conv(conv15, 3, 256, 'conv16', relu=False)
            #     conv16 = tf.nn.relu(conv16 + self._conv(conv14, 1, 256, 'conv16_tmp'))

            # with tf.variable_scope('block8'):
            #     conv17 = self._conv(conv16, 3, 256, 'conv17', relu=True)
            #     conv18 = self._conv(conv17, 3, 256, 'conv18', relu=False)
            #     conv18 = tf.nn.relu(conv18 + conv16)

            # with tf.variable_scope('block9'):
            #     conv19 = self._conv(conv18, 3, 256, 'conv19', relu=True)
            #     conv20 = self._conv(conv19, 3, 256, 'conv20', relu=False)
            #     conv21 = tf.nn.relu(conv20 + conv18)

        ###############################################
            # with tf.variable_scope('block10'):
            #     conv22 = self._conv(conv21, 3, 512, 'conv22', relu=True)
            #     conv23 = self._conv(conv22, 3, 512, 'conv23', relu=False)
            #     conv23 = tf.nn.relu(conv23 + self._conv(conv21, 1, 512, 'conv23_tmp'))

            # with tf.variable_scope('block11'):
            #     conv24 = self._conv(conv23, 3, 512, 'conv24', relu=True)
            #     conv25 = self._conv(conv24, 3, 512, 'conv25', relu=False)
            #     conv25 = tf.nn.relu(conv25 + conv23)

            # with tf.variable_scope('block12'):
            #     conv26 = self._conv(conv25, 3, 512, 'conv26', relu=True)
            #     conv27 = self._conv(conv26, 3, 512, 'conv27', relu=False)
            #     conv27 = tf.nn.relu(conv27 + conv25)

            pool28 = tf.nn.avg_pool(conv14, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool28')

            with tf.variable_scope('fully_connect'):
                #######################################################
                fc29 = self._fc(pool28, 256, 'fc29', relu=True, flatten=True, is_bn=True)
                if self.is_training is True:
                    fc29 = tf.nn.dropout(fc29, keep_prob=self.keep_prob)                
                fc_1 = self._fc(fc29, 5, 'fc_1', relu=False, flatten=False, is_bn=False)
                fc_2 = self._fc(fc29, 14, 'fc_2', relu=False, flatten=False, is_bn=False)
                fc_3 = self._fc(fc29, 9, 'fc_3', relu=False, flatten=False, is_bn=False)
                fc_4 = self._fc(fc29, 2019, 'fc_4', relu=False, flatten=False, is_bn=False)

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
            bn = tf.contrib.layers.batch_norm(inputs=layer, decay=0.9, updates_collections=None, is_training=self.is_training)
        return bn
