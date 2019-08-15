import tensorflow as tf
import numpy as np
from model import ResNet_model

from read_data import Read
from config import Config
import random

tf.flags.DEFINE_string("content_path", "H:/biebiecompetition/competion/open_data/sent_train.txt", "content path")
tf.flags.DEFINE_string("label_path", "H:/biebiecompetition/competion/open_data/sent_relation_train.txt", "label_path")
tf.flags.DEFINE_string("relation_label_path", "H:/biebiecompetition/competion/open_data/relation2id.txt", "relation label path")
tf.flags.DEFINE_boolean("shuffle", True, "whether shuffle")
tf.flags.DEFINE_integer("batch_size", 16, "batch size")

FLAGS = tf.flags.FLAGS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def build_model():

    read_model = Read(FLAGS.content_path, FLAGS.relation_label_path, FLAGS.label_path)
    data = read_model.read_data()
    label = read_model.read_label()
    length = len(data)
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(data)
    random.seed(randnum)
    random.shuffle(label)
    train_data = data[:900]
    train_label = label[:900]
    dev_data = data[900:]
    dev_label = label[900:]
    Model = ResNet_model()

    logits,probability, loss, train_op = Model.train()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(Config.epoch):
            batcher = read_model.batcher_yield(train_data, train_label, FLAGS.batch_size)
            for step, (train_con, train_p1, train_p2, l) in enumerate(batcher):
                feed_dict = {Model.input_x: train_con,
                             Model.input_p1: train_p1,
                             Model.input_p2: train_p2,
                             Model.input_y: l,
                             Model.dropout_keep_prob: 0.6}

                logit, Loss, _ = sess.run([logits, loss, train_op], feed_dict=feed_dict)
                if step % 1000 == 0:
                    print("epoch:{}, step:{}, loss:{}".format(epoch, step, Loss))










if __name__ == "__main__":
    build_model()