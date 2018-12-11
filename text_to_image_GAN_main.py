import tensorflow as tf
from glob import glob
import os
import numpy as np
from utils import append_nparr
from utils import save_images
import cv2
from model import Text_to_image_gan_model


class Text_to_image_GAN():
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.z_sample_format = tf.placeholder(tf.float32, [None, 2, self.args.z_input_size])


    def train(self):
        cv2.setNumThreads(0)

        self.input_pipeline_setup()
        self.network_setup()
        self.weight_setup()
        self.loss_setup()
        self.optimizer_setup()

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver_txt = tf.train.Saver(var_list=self.txt_vars)
        saver_gen = tf.train.Saver(var_list=self.gen_vars, max_to_keep=50)
        saver_dis = tf.train.Saver(var_list=self.dis_vars, max_to_keep=50)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.args.write_summary_path)

        with tf.Session() as sess:
            sess.run(init)
            saver_txt.restore(sess, self.args.text_encoder_ckpt)

            for cur_epoch in range(self.args.num_epoch):
                sess.run(self.train_init_op)

                total_loss_D = 0
                total_loss_G = 0

                idx = 0
                while True:
                    try:
                        _, _, loss_D, loss_G, generated_image, summary = sess.run([self.opti_D, self.opti_G,
                                                                                   self.D_loss, self.G_loss,
                                                                                   self.G_z_test, merged])
                        total_loss_D += loss_D
                        total_loss_G += loss_G

                        if idx % 100 == 0 and idx != 0:
                            save_images(generated_image,
                                        '{}/generated_{:02d}_train.png'.format(self.args.write_generated_img_path,
                                                                               cur_epoch))
                            train_writer.add_summary(summary, cur_epoch)
                        idx += 1
                    except tf.errors.OutOfRangeError:
                        sess.run(self.test_init_op)
                        generated_image = sess.run(self.G_z_test)
                        save_images(generated_image,
                                    '{}/generated_{:02d}_test.png'.format(self.args.write_generated_img_path,
                                                                          cur_epoch))
                        print("[EPOCH_{%02d}] d_loss: %.8f, g_loss: %.8f" % (cur_epoch,
                                                                             total_loss_D / idx,
                                                                             total_loss_G / idx))
                        saver_gen.save(sess, self.args.write_gen_model_path, global_step=cur_epoch)
                        saver_dis.save(sess, self.args.write_dis_model_path, global_step=cur_epoch)

                        if cur_epoch % self.args.learning_rate_decay_every == 0:
                            if not cur_epoch == 0:
                                self.args.D_learning_rate = self.args.D_learning_rate * self.args.learning_rate_decay
                                self.args.G_learning_rate = self.args.G_learning_rate * self.args.learning_rate_decay
                                print(self.args.D_learning_rate)
                        break


    def read_train_data(self, img_file_name, txt_object, wrong_txt_object):
        img_file_name = img_file_name.decode("utf-8")
        image = cv2.imread(img_file_name)
        image = np.float32(image)

        image = image / 255
        image = image - 0.5
        image = image * 2

        width = image.shape[0]
        height = image.shape[1]

        image_list = list()

        if width > height:
            width = int(width * 74 / height)
            height = 74

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:64, 0:64])  # left-up
            image_list.append(image[0:64, -64:])  # right-up
            image_list.append(image[-64:, 0:64])  # left-down
            image_list.append(image[-64:, -64:])  # right-down
            image_list.append(image[int(height / 2) - 32:int(height / 2) + 32,
                              int(width / 2) - 32:int(width / 2) + 32])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:64, 0:64])  # left-up
            image_list.append(image[0:64, -64:])  # right-up
            image_list.append(image[-64:, 0:64])  # left-down
            image_list.append(image[-64:, -64:])  # right-down
            image_list.append(image[int(height / 2) - 32:int(height / 2) + 32,
                              int(width / 2) - 32:int(width / 2) + 32])  # center

        elif height >= width:
            height = int(height * 74 / width)
            width = 74

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:64, 0:64])  # left-up
            image_list.append(image[0:64, -64:])  # right-up
            image_list.append(image[-64:, 0:64])  # left-down
            image_list.append(image[-64:, -64:])  # right-down
            image_list.append(image[int(height / 2) - 32:int(height / 2) + 32,
                              int(width / 2) - 32:int(width / 2) + 32])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:64, 0:64])  # left-up
            image_list.append(image[0:64, -64:])  # right-up
            image_list.append(image[-64:, 0:64])  # left-down
            image_list.append(image[-64:, -64:])  # right-down
            image_list.append(image[int(height / 2) - 32:int(height / 2) + 32,
                              int(width / 2) - 32:int(width / 2) + 32])  # center

        image_list = np.float32(image_list)
        np.random.shuffle(image_list)
        z_sample = np.random.uniform(-1, 1, size=[self.args.z_input_size]).astype(np.float32)

        np.random.shuffle(txt_object)
        np.random.shuffle(wrong_txt_object)

        return image_list[0], np.float32(txt_object[0]), np.float32(wrong_txt_object[0]), z_sample


    def read_test_data(self, txt_object):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        z_sample = np.random.uniform(-1, 1, size=[self.args.z_input_size]).astype(np.float32)

        return image, np.float32(txt_object[0]), np.float32(txt_object[0]), z_sample


    def input_pipeline_setup(self):
        read_list = list()
        test_list = list()

        with open(self.args.train_meta_path, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                read_list.append(line)

        with open(self.args.test_meta_path, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                test_list.append(line)

        img_file_name_list = list()
        train_txt_list = None
        for class_name in read_list:
            class_name = class_name.replace("\n", "")
            new_img_file_name_list = sorted(glob(os.path.join(self.args.train_img_data_path,
                                                              class_name,
                                                              self.args.train_img_data_type)))
            img_file_name_list += new_img_file_name_list

            new_txt_file_name_list = sorted(glob(os.path.join(self.args.train_txt_data_path,
                                                              class_name,
                                                              self.args.train_txt_data_type)))
            new_txt_list = None
            for txt_file_name in new_txt_file_name_list:
                new_txt = np.load(txt_file_name)
                new_txt = new_txt[0:self.args.num_caption]
                new_txt = np.int8(new_txt)
                new_txt = np.expand_dims(new_txt, axis=0)
                new_txt_list = append_nparr(new_txt_list, new_txt)

            train_txt_list = append_nparr(train_txt_list, new_txt_list)
            print(np.shape(train_txt_list))

        wrong_txt_list = train_txt_list.copy()
        np.random.shuffle(wrong_txt_list)

        test_txt_list = None
        for class_name in test_list:
            class_name = class_name.replace("\n", "")
            new_txt_file_name_list = sorted(glob(os.path.join(self.args.train_txt_data_path,
                                                              class_name,
                                                              self.args.train_txt_data_type)))
            new_txt_list = None
            for txt_file_name in new_txt_file_name_list:
                new_txt = np.load(txt_file_name)
                new_txt = np.int8(new_txt)
                np.random.shuffle(new_txt)
                new_txt = new_txt[0:self.args.num_caption]
                new_txt = np.expand_dims(new_txt, axis=0)
                new_txt_list = append_nparr(new_txt_list, new_txt)

            test_txt_list = append_nparr(test_txt_list, new_txt_list)

        dataset = tf.data.Dataset.from_tensor_slices((img_file_name_list,
                                                      train_txt_list,
                                                      wrong_txt_list))
        dataset = dataset.shuffle(buffer_size=self.args.buffer_size)
        dataset = dataset.map(lambda filename, txt, wrongtxt:
                              tuple(tf.py_func(self.read_train_data,
                                               [filename, txt, wrongtxt],
                                               [tf.float32, tf.float32, tf.float32, tf.float32])),
                              num_parallel_calls=10)
        dataset = dataset.prefetch(self.args.batch_size * self.args.prefetch_multiply)
        dataset = dataset.batch(self.args.batch_size)

        dataset_test = tf.data.Dataset.from_tensor_slices(test_txt_list)
        dataset_test = dataset_test.map(lambda txt:
                                        tuple(tf.py_func(self.read_test_data,
                                                         [txt],
                                                         [tf.float32, tf.float32, tf.float32, tf.float32])),
                                        num_parallel_calls=10)
        dataset_test = dataset_test.batch(64)

        self.iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                        (tf.TensorShape([None, 64, 64, 3]),
                                                         tf.TensorShape([None, 201, 1, self.args.alpha_size]),
                                                         tf.TensorShape([None, 201, 1, self.args.alpha_size]),
                                                         tf.TensorShape([None, self.args.z_input_size])))

        self.train_init_op = self.iterator.make_initializer(dataset)
        self.test_init_op = self.iterator.make_initializer(dataset_test)

        self.img_batch, self.txt_batch, self.wrong_txt_batch, self.z_batch = self.iterator.get_next()


    def network_setup(self):
        model = Text_to_image_gan_model(args=self.args)

        # selected_img = self.img_batch[:, tf.squeeze(tf.random.uniform(shape=[1], maxval=10, dtype=tf.int32)), :, :, :]
        # selected_txt = self.txt_batch[:, tf.squeeze(tf.random.uniform(shape=[1],
        #                                                               maxval=self.args.num_caption,
        #                                                               dtype=tf.int32)), :, :, :]
        # selected_wrong_txt = self.wrong_txt_batch[:, tf.squeeze(tf.random.uniform(shape=[1],
        #                                                                           maxval=self.args.num_caption,
        #                                                                           dtype=tf.int32)), :, :, :]

        # txt encoder
        txt_embed = model.text_encoder(self.txt_batch)
        wrong_txt_embed = model.text_encoder(self.wrong_txt_batch, reuse=True)
        # int_txt_embed = (txt_embed + wrong_txt_embed) / 2

        # G network setup
        self.G_z = model.generator(z_input=self.z_batch,
                                   txt_embed_input=txt_embed)
        # self.G_z_int = model.generator(z_input=self.z_batch[:, 1, :],
        #                                txt_embed_input=int_txt_embed,
        #                                reuse=True)
        self.G_z_test = model.generator(z_input=self.z_batch,
                                        txt_embed_input=txt_embed,
                                        reuse=True,
                                        is_train=False)

        # D network setup
        self.D_x_right_logit = model.discriminator(x_input=self.img_batch,
                                                   txt_embed_input=txt_embed)
        self.D_x_wrong_logit = model.discriminator(x_input=self.img_batch,
                                                   txt_embed_input=wrong_txt_embed,
                                                   reuse=True)
        self.D_Gz_right_logit = model.discriminator(x_input=self.G_z,
                                                    txt_embed_input=txt_embed,
                                                    reuse=True)
        # self.D_Gz_int_logit = model.discriminator(x_input=self.G_z_int,
        #                                           txt_embed_input=int_txt_embed,
        #                                           reuse=True)


    def weight_setup(self):
        self.total_vars = tf.global_variables()

        self.txt_vars = [var for var in self.total_vars if var.name.startswith('text_encoder')]
        self.gen_vars = [var for var in self.total_vars if var.name.startswith('generator')]
        self.dis_vars = [var for var in self.total_vars if var.name.startswith('discriminator')]

        var_cnt = 0
        for var in self.gen_vars:
            tf.summary.histogram('histogram_g%d' % var_cnt, var)

        var_cnt = 0
        for var in self.dis_vars:
            tf.summary.histogram('histogram_d%d' % var_cnt, var)


    def loss_setup(self):
        D_x_right_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_x_right_logit,
                                                                 labels=tf.ones_like(self.D_x_right_logit))
        D_x_wrong_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_x_wrong_logit,
                                                                 labels=tf.zeros_like(self.D_x_wrong_logit))
        D_Gz_right_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gz_right_logit,
                                                                  labels=tf.zeros_like(self.D_Gz_right_logit))

        G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gz_right_logit,
                                                         labels=tf.ones_like(self.D_Gz_right_logit))
        # self.G_int_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_Gz_int_logit,
        #                                                           labels=tf.zeros_like(self.D_Gz_int_logit))

        self.D_loss = tf.reduce_mean(D_x_right_loss + (D_x_wrong_loss + D_Gz_right_loss) / 2)
        self.G_loss = tf.reduce_mean(G_loss)

        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss', self.G_loss)


    def optimizer_setup(self):
        self.opti_D = tf.train.AdamOptimizer(learning_rate=self.args.D_learning_rate,
                                             beta1=0.5).minimize(self.D_loss, var_list=[self.dis_vars])
        self.opti_G = tf.train.AdamOptimizer(learning_rate=self.args.G_learning_rate,
                                             beta1=0.5).minimize(self.G_loss, var_list=[self.gen_vars])