#from yaml import load
import tensorflow as tf
import numpy as np
import json
import logging
import os
import scipy.io as sio
import scipy
from scipy import sparse
import json 
import math
from pathlib import Path
from operator import itemgetter

#For glove, get mean and SD from existing embeddings
GLOVE_MEAN = np.array([-0.12920061, -0.28866239, -0.01224894, -0.05676689, -0.20211109,
       -0.08389026,  0.33359737,  0.16045146,  0.03867495,  0.17833092,
        0.0469662 , -0.00285779,  0.29099851,  0.04613723, -0.20923842,
       -0.066131  , -0.06822448,  0.07665885,  0.31339918,  0.17848512,
       -0.12257719, -0.09916928, -0.07495973,  0.06413206,  0.14441256,
        0.608946  ,  0.17463101,  0.05335403, -0.01273826,  0.03474108,
       -0.81239567, -0.04688727,  0.20193533,  0.20311115, -0.03935654,
        0.06967518, -0.01553655, -0.03405275, -0.06528025,  0.12250092,
        0.13992005, -0.17446305, -0.08011841,  0.08495219, -0.01041645,
       -0.13704901,  0.20127088,  0.10069294,  0.00653007,  0.0168515 ])

GLOVE_SD = np.array([0.6367036 , 0.61399522, 0.61246498, 0.62420189, 0.63991811,
       0.66558709, 0.62343921, 0.60837914, 0.64596959, 0.6192167 ,
       0.56026841, 0.64560634, 0.63535613, 0.65003734, 0.60408052,
       0.55042996, 0.59963162, 0.59922248, 0.63446751, 0.60850917,
       0.63192796, 0.5854551 , 0.60097065, 0.58522762, 0.59460321,
       0.64659922, 0.664701  , 0.53545429, 0.61339997, 0.58462912,
       0.81344494, 0.59899052, 0.58130035, 0.64292357, 0.57981031,
       0.53389309, 0.53456518, 0.6020573 , 0.61489106, 0.60126015,
       0.57000732, 0.60827673, 0.60869018, 0.62945291, 0.62749929,
       0.62078889, 0.55848591, 0.61550812, 0.60006644, 0.62328338])


def load_data(mat_file_name, is_to_dense=True):

    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain'].transpose()
    test_data = data['wordsTest'].transpose()


    word_embeddings = data['embeddings']
    voc = data['vocabulary']


    voc = [v[0][0] for v in voc]

    if is_to_dense:
        if sparse.isspmatrix(train_data):
            train_data = train_data.toarray()
        train_data = train_data.astype('float32')

        if sparse.isspmatrix(test_data):
            test_data = test_data.toarray()
        test_data = test_data.astype('float32')

    print(type(word_embeddings), word_embeddings.shape)
    return train_data, test_data, word_embeddings, voc

def load_glove_model(modelpath, words):
    embeds = {}
    for word in words: 
        embeds[word] = np.random.normal(GLOVE_MEAN, GLOVE_SD)

    with open(modelpath) as f: 
        for line in f: 
            tokens = line.split()
            word = tokens[0]
            embed = np.array(tokens[1:], dtype=np.float64) 
            if word in embeds: 
                embeds[word] = embed
    
    
    word_embeddings = np.array(list(embeds.values()))
    print(word_embeddings.shape, len(words))
    assert(len(words) == word_embeddings.shape[0])

    return word_embeddings

def load_data_modified(
    model_path="glove",
    is_to_dense=True,
    train_dtm_path="train.dtm.npz",
    test_dtm_path="test.dtm.npz",
    embeds_path=None, 
    vocab_path="vocab.json",
    ):
    
    """
    Load data to pass into the nstm model. 

    returns -- train_data, test_data, word_embeddings, vec
    """
    
    #load document topic matrices
    train_data = scipy.sparse.load_npz(train_dtm_path)
    test_data = scipy.sparse.load_npz(test_dtm_path)
    

    #load vocabulary 
    with open(vocab_path) as f :
        vocab = json.load(f)
    
    voc = list(vocab.keys())
    
    # get embeddings 
    if embeds_path.exists(): 
        word_embeddings = np.load(embeds_path)
        print("loaded embeddings")
    else : 
        print("Creating embeddings") 
        word_embeddings = load_glove_model(model_path, voc)
        Path(embeds_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeds_path, word_embeddings)
        print("Created and saved embeddings")

    assert(len(voc) == word_embeddings.shape[0])


    if is_to_dense:
        if sparse.isspmatrix(train_data):
            train_data = train_data.toarray()
        train_data = train_data.astype('float32')

        if sparse.isspmatrix(test_data):
            test_data = test_data.toarray()
        test_data = test_data.astype('float32')

    #print(type(word_embeddings), word_embeddings.shape)
    return train_data, test_data, word_embeddings, voc





def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
    """Define a linear connection."""
    with tf.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix_initializer = tf.constant_initializer(0)
        else:
            matrix_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        if bias_start_zero:
            bias_initializer = tf.constant_initializer(0)
        else:
            bias_initializer = None
        input_size = inputs.get_shape()[1].value

        if weights is not None:
            matrix = weights
        else:
            matrix = tf.get_variable('Matrix', [input_size, output_size], initializer=matrix_initializer)

        output = tf.matmul(inputs, matrix)
        if not no_bias:
            bias_term = tf.get_variable('Bias', [output_size],
                                        initializer=bias_initializer)
            output = output + bias_term
    return output


def mlp(inputs,
        mlp_hidden=[],
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
    """Define an MLP."""
    with tf.variable_scope(scope or 'Linear'):
        mlp_layer = len(mlp_hidden)
        res = inputs
        for l in range(mlp_layer):
            res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l' + str(l)))
        return res

def myrelu(features):
    return tf.maximum(features, 0.0)

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end



def set_logger(save_dir=None, logger_name='nstm', log_file_name='log.txt'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, log_file_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def save_flags(results_dir):
    FLAGS = tf.flags.FLAGS
    train_params = json.dumps({k: v.value
                               for k, v in FLAGS._flags().items()}, sort_keys=True)
    with open(os.path.join(results_dir, 'params.txt'), 'a') as f:
        f.writelines(str(train_params))
        f.write('\n')


def get_doc_topic(sess, doc_topic_tf, doc_word_tf, doc_word, K, other_param_tf=None, batch_size=200):
    N = np.shape(doc_word)[0]
    nb_batches = int(math.ceil(float(N) / batch_size))
    assert nb_batches * batch_size >= N

    import scipy.sparse

    doc_topic = np.zeros((N, K))
    for batch in range(nb_batches):
        start, end = batch_indices(batch, N, batch_size)
        X = doc_word[start:end]

        if scipy.sparse.issparse(X):
            X = X.todense()
            X = X.astype('float32')

        feed_dict = {doc_word_tf: X}
        if other_param_tf is not None:
            feed_dict.update(other_param_tf)

        temp = sess.run(doc_topic_tf, feed_dict)

        doc_topic[start:end] = temp

    return doc_topic

def print_topics(topic_word_mat, voc, doc_topic_mat=None, sample_doc_word_mat=None, top_words_N=50, top_docs_N=10,
                 printer=None):
    if printer == None:
        printer = print

    K = np.shape(topic_word_mat)[0]

    V = np.shape(topic_word_mat)[1]

    if doc_topic_mat is not None:
        rank = np.argsort(np.sum(doc_topic_mat, axis=0))[::-1]
    else:
        rank = list(range(K))

    assert V == len(voc)

    for k in rank:

        top_word_idx = np.argsort(topic_word_mat[k, :])[::-1]

        top_word_idx = top_word_idx[0:top_words_N]

        top_words = itemgetter(*top_word_idx)(voc)

        printer('topic_%d:\t[%s]\n' % (k, ', '.join(map(str, top_words))))

        if doc_topic_mat is not None:

            doc_rank = np.argsort(doc_topic_mat[:, k])[::-1]

            doc_rank = doc_rank[0:top_docs_N]

            for i in doc_rank:
                doc_words_idx = np.nonzero(sample_doc_word_mat[i, :])[0]

                top_words = itemgetter(*doc_words_idx)(voc)

                printer('*******doc words: [%s]' % ', '.join(map(str, top_words)))
    printer('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
        elif prefix in varname:
            ret_list.append(var)
    return ret_list
