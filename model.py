import tensorflow as tf
import numpy as np


class Text_to_image_gan_model():
    def __init__(self, **args):
        self.args = args['args']
        print(self.args)


    def text_encoder(self, text_input, reuse=False):
        with tf.variable_scope("text_encoder") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = tf.layers.conv2d(inputs=text_input,
                                     filters=384,
                                     kernel_size=[4, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 198 x 384 dimension
            conv1_max_pool = tf.nn.max_pool(value=conv1,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 66 x 1 x 384 dimension

            conv2 = tf.layers.conv2d(inputs=conv1_max_pool,
                                     filters=512,
                                     kernel_size=[4, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 63 x 1 x 512 dimension
            conv2_max_pool = tf.nn.max_pool(value=conv2,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 21 x 1 x 512 dimension

            conv3 = tf.layers.conv2d(inputs=conv2_max_pool,
                                     filters=self.args.cnn_represent_dim,
                                     kernel_size=[5, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 18 x 1 x 1024 dimension
            conv3_max_pool = tf.nn.max_pool(value=conv3,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 2, 1, 1],
                                            padding='VALID') # 8 x 1 x 1024 dimension
            cnn_final = tf.squeeze(conv3_max_pool, [2]) # 8 x 1024 dimension

            cnn_final_sequence = tf.split(cnn_final, 8, axis=1)
            cnn_final_list = list()
            for temporal in cnn_final_sequence:
                cnn_final_list.append(tf.squeeze(temporal, [1]))

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.args.cnn_represent_dim, activation=tf.nn.relu, reuse=reuse)
            outputs, state = tf.nn.static_rnn(rnn_cell, cnn_final_list, dtype=tf.float32)

            embedded_code = tf.reduce_mean(outputs, axis=0)

        if reuse == False:
            print('DS_SJE Architecture')
            print(np.shape(conv1))
            print(np.shape(conv1_max_pool))
            print(np.shape(conv2))
            print(np.shape(conv2_max_pool))
            print(np.shape(conv3))
            print(np.shape(conv3_max_pool))
            print(np.shape(cnn_final))
            print(np.shape(state))
            print(np.shape(outputs))
            print(np.shape(embedded_code))

        return embedded_code


    def generator(self, z_input, txt_embed_input, reuse=False, is_train=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            with tf.variable_scope("text_regression"):
                txt_embed = tf.layers.dense(txt_embed_input,
                                            units=self.args.txt_embed_size,
                                            activation=tf.nn.leaky_relu)
                concat_input = tf.concat([z_input, txt_embed], axis=-1)
                concat_input = tf.expand_dims(concat_input, axis=1)
                concat_input = tf.expand_dims(concat_input, axis=1)

            with tf.variable_scope("G1024"):
                feature_map1 = tf.layers.conv2d_transpose(concat_input,
                                                          kernel_size=4,
                                                          filters=self.args.num_init_filter,
                                                          strides=1,
                                                          padding='VALID',
                                                          activation=None)
                feature_map1_batchnorm = tf.contrib.layers.batch_norm(feature_map1,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)

            with tf.variable_scope("G1024_residual"):
                feature_map1_1 = tf.layers.conv2d(feature_map1_batchnorm,
                                                  kernel_size=1,
                                                  filters=int(self.args.num_init_filter / 4),
                                                  strides=1,
                                                  padding='VALID',
                                                  activation=None)
                feature_map1_1_batchnorm = tf.contrib.layers.batch_norm(feature_map1_1,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map1_1_batchnorm = tf.nn.relu(feature_map1_1_batchnorm)

                feature_map1_2 = tf.layers.conv2d(feature_map1_1_batchnorm,
                                                  kernel_size=3,
                                                  filters=int(self.args.num_init_filter / 4),
                                                  strides=1,
                                                  padding='SAME',
                                                  activation=None)
                feature_map1_2_batchnorm = tf.contrib.layers.batch_norm(feature_map1_2,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_2',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map1_2_batchnorm = tf.nn.relu(feature_map1_2_batchnorm)

                feature_map1_3 = tf.layers.conv2d(feature_map1_2_batchnorm,
                                                  kernel_size=3,
                                                  filters=self.args.num_init_filter,
                                                  strides=1,
                                                  padding='SAME',
                                                  activation=None)
                feature_map1_3_batchnorm = tf.contrib.layers.batch_norm(feature_map1_3,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_3',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map1_residual = feature_map1_3_batchnorm + feature_map1_batchnorm
                feature_map1_residual = tf.nn.relu(feature_map1_residual)

            with tf.variable_scope("G512"):
                feature_map2 = tf.layers.conv2d_transpose(feature_map1_residual,
                                                          kernel_size=4,
                                                          filters=int(self.args.num_init_filter / 2),
                                                          strides=2,
                                                          padding='SAME',
                                                          activation=None)
                feature_map2_batchnorm = tf.contrib.layers.batch_norm(feature_map2,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)

            with tf.variable_scope("G512_residual"):
                feature_map2_1 = tf.layers.conv2d(feature_map2_batchnorm,
                                                  kernel_size=1,
                                                  filters=int(self.args.num_init_filter / 8),
                                                  strides=1,
                                                  padding='VALID',
                                                  activation=None)
                feature_map2_1_batchnorm = tf.contrib.layers.batch_norm(feature_map2_1,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map2_1_batchnorm = tf.nn.relu(feature_map2_1_batchnorm)

                feature_map2_2 = tf.layers.conv2d(feature_map2_1_batchnorm,
                                                  kernel_size=3,
                                                  filters=int(self.args.num_init_filter / 8),
                                                  strides=1,
                                                  padding='SAME',
                                                  activation=None)
                feature_map2_2_batchnorm = tf.contrib.layers.batch_norm(feature_map2_2,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_2',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map2_2_batchnorm = tf.nn.relu(feature_map2_2_batchnorm)

                feature_map2_3 = tf.layers.conv2d(feature_map2_2_batchnorm,
                                                  kernel_size=3,
                                                  filters=int(self.args.num_init_filter / 2),
                                                  strides=1,
                                                  padding='SAME',
                                                  activation=None)
                feature_map2_3_batchnorm = tf.contrib.layers.batch_norm(feature_map2_3,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_3',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map2_residual = feature_map2_3_batchnorm + feature_map2_batchnorm
                feature_map2_residual = tf.nn.relu(feature_map2_residual)

            with tf.variable_scope("G256"):
                feature_map3 = tf.layers.conv2d_transpose(feature_map2_residual,
                                                          kernel_size=4,
                                                          filters=int(self.args.num_init_filter / 4),
                                                          strides=2,
                                                          padding='SAME',
                                                          activation=None)
                feature_map3_batchnorm = tf.contrib.layers.batch_norm(feature_map3,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)
                feature_map3_batchnorm = tf.nn.relu(feature_map3_batchnorm)

            with tf.variable_scope("G128"):
                feature_map4 = tf.layers.conv2d_transpose(feature_map3_batchnorm,
                                                          kernel_size=4,
                                                          filters=int(self.args.num_init_filter / 8),
                                                          strides=2,
                                                          padding='SAME',
                                                          activation=None)
                feature_map4_batchnorm = tf.contrib.layers.batch_norm(feature_map4,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)
                feature_map4_batchnorm = tf.nn.relu(feature_map4_batchnorm)

            with tf.variable_scope("G_final"):
                feature_map5 = tf.layers.conv2d_transpose(feature_map4_batchnorm,
                                                          kernel_size=4,
                                                          filters=self.args.num_channel,
                                                          strides=2,
                                                          padding='SAME',
                                                          activation=tf.nn.tanh)

        if not reuse:
            print('Generator Architecture')
            print(np.shape(txt_embed))
            print(np.shape(concat_input))
            print(np.shape(feature_map1_batchnorm))
            print(np.shape(feature_map1_1_batchnorm))
            print(np.shape(feature_map1_2_batchnorm))
            print(np.shape(feature_map1_residual))
            print(np.shape(feature_map2_batchnorm))
            print(np.shape(feature_map2_1_batchnorm))
            print(np.shape(feature_map2_2_batchnorm))
            print(np.shape(feature_map2_residual))
            print(np.shape(feature_map3_batchnorm))
            print(np.shape(feature_map4_batchnorm))
            print(np.shape(feature_map5))


        return feature_map5


    def discriminator(self, x_input, txt_embed_input, reuse=False, is_train=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            with tf.variable_scope("D128"):
                feature_map1 = tf.layers.conv2d(x_input,
                                                kernel_size=4,
                                                filters=int(self.args.num_init_filter / 8),
                                                strides=2,
                                                padding='SAME',
                                                activation=tf.nn.leaky_relu)

            with tf.variable_scope("D256"):
                feature_map2 = tf.layers.conv2d(feature_map1,
                                                kernel_size=4,
                                                filters=int(self.args.num_init_filter / 4),
                                                strides=2,
                                                padding='SAME',
                                                activation=None)
                feature_map2_batchnorm = tf.contrib.layers.batch_norm(feature_map2,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)
                feature_map2_batchnorm = tf.nn.leaky_relu(feature_map2_batchnorm)

            with tf.variable_scope("D512"):
                feature_map3 = tf.layers.conv2d(feature_map2_batchnorm,
                                                kernel_size=4,
                                                filters=int(self.args.num_init_filter / 2),
                                                strides=2,
                                                padding='SAME',
                                                activation=None)
                feature_map3_batchnorm = tf.contrib.layers.batch_norm(feature_map3,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)

            with tf.variable_scope("D1024"):
                feature_map4 = tf.layers.conv2d(feature_map3_batchnorm,
                                                kernel_size=4,
                                                filters=self.args.num_init_filter,
                                                strides=2,
                                                padding='SAME',
                                                activation=None)
                feature_map4_batchnorm = tf.contrib.layers.batch_norm(feature_map4,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)

            with tf.variable_scope("D1024_residual"):
                feature_map4_1 = tf.layers.conv2d(feature_map4_batchnorm,
                                                  kernel_size=1,
                                                  filters=int(self.args.num_init_filter / 4),
                                                  strides=1,
                                                  padding='VALID',
                                                  activation=None)
                feature_map4_1_batchnorm = tf.contrib.layers.batch_norm(feature_map4_1,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map4_1_batchnorm = tf.nn.leaky_relu(feature_map4_1_batchnorm)

                feature_map4_2 = tf.layers.conv2d(feature_map4_1_batchnorm,
                                                  kernel_size=3,
                                                  filters=int(self.args.num_init_filter / 4),
                                                  strides=2,
                                                  padding='SAME',
                                                  activation=None)
                feature_map4_2_batchnorm = tf.contrib.layers.batch_norm(feature_map4_2,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_2',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map4_2_batchnorm = tf.nn.leaky_relu(feature_map4_2_batchnorm)

                feature_map4_3 = tf.layers.conv2d(feature_map4_2_batchnorm,
                                                  kernel_size=3,
                                                  filters=self.args.num_init_filter,
                                                  strides=2,
                                                  padding='SAME',
                                                  activation=None)
                feature_map4_3_batchnorm = tf.contrib.layers.batch_norm(feature_map4_3,
                                                                        decay=0.9,
                                                                        updates_collections=None,
                                                                        epsilon=1e-5,
                                                                        scale=True,
                                                                        scope='batch_norm_3',
                                                                        reuse=reuse,
                                                                        is_training=is_train)
                feature_map4_residual = feature_map4_3_batchnorm + feature_map4_batchnorm
                feature_map4_residual = tf.nn.leaky_relu(feature_map4_residual)

            with tf.variable_scope("text_regression"):
                txt_embed = tf.layers.dense(txt_embed_input,
                                            units=self.args.txt_embed_size,
                                            activation=tf.nn.leaky_relu)
                txt_embed_replicate = tf.tile(txt_embed, [1, 16])
                txt_embed_reshaped = tf.reshape(txt_embed_replicate, [-1, 4, 4, self.args.txt_embed_size])

            with tf.variable_scope("concat"):
                feature_map_concat = tf.concat([feature_map4_residual, txt_embed_reshaped], axis=3)

                feature_map5 = tf.layers.conv2d(feature_map_concat,
                                                kernel_size=1,
                                                filters=self.args.num_init_filter,
                                                strides=1,
                                                padding='VALID',
                                                activation=None)
                feature_map5_batchnorm = tf.contrib.layers.batch_norm(feature_map5,
                                                                      decay=0.9,
                                                                      updates_collections=None,
                                                                      epsilon=1e-5,
                                                                      scale=True,
                                                                      scope='batch_norm',
                                                                      reuse=reuse,
                                                                      is_training=is_train)
                feature_map5_batchnorm = tf.nn.leaky_relu(feature_map5_batchnorm)

            with tf.variable_scope("final"):
                logit = tf.layers.conv2d(feature_map5_batchnorm,
                                         kernel_size=4,
                                         filters=1,
                                         strides=1,
                                         padding='VALID',
                                         activation=None)

            if not reuse:
                print('Discriminator Architecture')
                print(np.shape(feature_map1))
                print(np.shape(feature_map2_batchnorm))
                print(np.shape(feature_map3_batchnorm))
                print(np.shape(feature_map4_batchnorm))
                print(np.shape(feature_map4_1_batchnorm))
                print(np.shape(feature_map4_2_batchnorm))
                print(np.shape(feature_map4_residual))
                print(np.shape(txt_embed))
                print(np.shape(txt_embed_replicate))
                print(np.shape(txt_embed_reshaped))
                print(np.shape(feature_map_concat))
                print(np.shape(feature_map5_batchnorm))
                print(np.shape(logit))

            return tf.squeeze(logit)