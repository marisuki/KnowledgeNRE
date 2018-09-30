import numpy as np
import tensorflow as tf
from ctypes import *
import json
import network

cdll = windll.LoadLibrary
lib = cdll('init.dll')

export_path = './data/'

config_file = open(export_path + 'config')
config = json.load(config_file)
config_file.close()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('nbatch_kg', 100, 'entity numbers of each training time')
tf.flags.DEFINE_float('margin', 0.1, 'entity numbvers used each training time')
tf.flags.DEFINE_float('learning_rate_kg', 0.001, 'learning rate for kg')
tf.flags.DEFINE_float('ent_total', lib.getEntityTotal(), 'total entity number')
tf.flags.DEFINE_float('rel_total', lib.getRelationTotal(), 'total relation number')
tf.flags.DEFINE_float('tri_total', lib.getTripleTotal(), 'total triple number')
tf.flags.DEFINE_float('katt_flag', 0, '1: katt, 0:att')
tf.flags.DEFINE_string('model', 'cnn', 'kind of models for sentence encode, cnn implemented, GAN waited')
tf.flags.DEFINE_float('max_length', config['fixlen'], 'maximum of number of words in one sentence')
tf.flags.DEFINE_float('pos_num', config['maxlen'] * 2 + 1, 'number of position embedding vectors')
tf.flags.DEFINE_float('num_class', config['textual_rel_total'], 'maximum of relations')
tf.flags.DEFINE_float('hidden_size', 230, 'hidden feature size')
tf.flags.DEFINE_float('pos_size', 5, 'position embedding size')
tf.flags.DEFINE_float('max_epoch', 30, 'maximum of training epochs')
tf.flags.DEFINE_float('batch_size', 131 * 2, 'entity numbers used each training time')
tf.flags.DEFINE_float('learning_rate', 0.1, 'entity numbers used each training time')
tf.flags.DEFINE_float('weight_decay', 0.00001, 'weight_decay')
tf.flags.DEFINE_float('keep_prob', 1.0, 'dropout rate')
tf.flags.DEFINE_float('test_batch_size', 131 * 2, 'entity numbers used each test time')
tf.flags.DEFINE_string('checkpoint_path', './model/', 'path to store model')


class nreapi(object):
    def __init__(self):
        print("NRE\tSetup: CNN NRE based on joint KG\nWaiting...")
        print("Word embedding...")
        self.word_vec = np.load(export_path + 'vec.npy')
        print("embedding finished.")
        print('relations		: %d' % FLAGS.num_classes)
        print('word size		: %d' % (len(self.word_vec[0])))
        print('position size 	: %d' % FLAGS.pos_size)
        print('hidden size		: %d' % FLAGS.hidden_size)
        print("reading pre-data finished.")
        print("network building:")
        self.sess = tf.Session()
        self.model = network.CNN(is_training=False, word_embeddings=self.word_vec)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.dict_word = {}

    def step(self, head_index, tail_index, word, pos1, pos2, mask, leng, label_index, label, scope):
        self.feed_dict = {
            self.model.tail_index: tail_index,
            self.model.head_index: head_index,
            self.model.pos1: pos1,
            self.model.word: word,
            self.model.pos2: pos2,
            self.model.label_index: label_index,
            self.model.mask: mask,
            self.model.max_length: leng,
            self.model.label: label,
            self.model.scope: scope
        }
        output = self.sess.run(self.model.test_output, feed_dict=self.feed_dict)
        return output


