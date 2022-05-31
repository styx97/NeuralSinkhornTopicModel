from cgi import test
from email.mime import base
import math
from pydoc_data.topics import topics

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import time
from scipy import sparse
import configargparse
import shutil 
from pathlib import Path

base_dir = "./"
sys.path.insert(1, base_dir)

import utils
from auto_diff_sinkhorn import sinkhorn_tf



from utils import load_data, batch_indices, load_data_modified, print_topics, set_logger, save_flags, get_doc_topic

# flags = tf.app.flags
# flags.DEFINE_float('sh_epsilon', 0.001, 'sinkhorn epsilon')
# flags.DEFINE_integer('sh_iterations', 50, 'sinkhorn iterations')
# flags.DEFINE_string('dataset', 'TMN', 'dataset')
# flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
# flags.DEFINE_integer('batch_size', 200, 'batch_size')
# flags.DEFINE_integer('K', 100, 'num topics')
# flags.DEFINE_integer('random_seed', 1, 'random_seed')
# flags.DEFINE_integer('n_epochs', 50, 'n_epochs')
# flags.DEFINE_float('rec_loss_weight', 0.07, 'rec_loss_weight')
# flags.DEFINE_float('sh_alpha', 20, 'sh_alpha')
# FLAGS = flags.FLAGS

def run_ntsm(
    sh_epsilon=0.001, 
    sh_iterations=50, 
    output_dir=None,
    learning_rate=0.001,
    batch_size=200, 
    num_topics=100,
    random_seed=42, 
    num_epochs=50, 
    rec_loss_weight=0.07, 
    sh_alpha=20,
    train_data=None,
    test_data=None,
    word_embeddings=None, 
    voc=None
):


    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    #output_dir = os.path.join(base_dir, 'save', 'dataset%s_K%d_RW%0.3f_RS%d_L%0.3f' %
    #                        (FLAGS.dataset, num_topics, rec_loss_weight, random_seed, sh_alpha))

    #os.makedirs(output_dir, exist_ok=True)

    #save_flags(output_dir)

    logger = set_logger(output_dir)

    #data_dir = os.path.join(base_dir, 'datasets')
    #data_dir = '%s/%s' % (data_dir, FLAGS.dataset)

    #train_data, test_data, word_embeddings, voc = load_data('%s/data.mat' % data_dir, True)

    L = word_embeddings.shape[1]

    V = train_data.shape[1]
    N = train_data.shape[0]

    doc_word_ph = tf.placeholder(dtype=tf.float32, shape=[None, V])

    doc_word_tf = tf.nn.softmax(doc_word_ph)

    with tf.variable_scope('encoder'):

        doc_topic_tf = utils.mlp(doc_word_ph, [200], utils.myrelu)
        doc_topic_tf = tf.nn.dropout(doc_topic_tf, 0.75)
        doc_topic_tf = tf.contrib.layers.batch_norm(utils.linear(doc_topic_tf, num_topics, scope='mean'))
        doc_topic_tf = tf.nn.softmax(doc_topic_tf)

    with tf.variable_scope('cost_function'):

        topic_embeddings_tf = tf.get_variable(name='topic_embeddings', shape=[num_topics, L],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, seed=random_seed))
        word_embeddings_ph = tf.placeholder(dtype=tf.float32, shape=[V, L])

        topic_embedding_norm = tf.nn.l2_normalize(topic_embeddings_tf, dim=1)
        word_embedding_norm = tf.nn.l2_normalize(word_embeddings_ph, dim=1)
        topic_word_tf = tf.matmul(topic_embedding_norm, tf.transpose(word_embedding_norm))
        M = 1 - topic_word_tf


    sh_loss = sinkhorn_tf(M, tf.transpose(doc_topic_tf), tf.transpose(doc_word_tf), lambda_sh = sh_alpha)

    sh_loss = tf.reduce_mean(sh_loss)

    rec_log_probs = tf.nn.log_softmax(tf.matmul(doc_topic_tf, topic_word_tf))
    rec_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(rec_log_probs, doc_word_ph), 1))

    fullvars = tf.trainable_variables()
    enc_vars = utils.variable_parser(fullvars, 'encoder')
    cost_function_vars = utils.variable_parser(fullvars, 'cost_function')

    rec_loss_weight = tf.placeholder(tf.float32, ())

    joint_loss = rec_loss_weight * rec_loss + sh_loss
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(joint_loss, var_list=[enc_vars + cost_function_vars])

    saver = tf.train.Saver()

    is_stop = False

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    with session as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        nb_batches = int(math.ceil(float(N) / batch_size))
        assert nb_batches * batch_size >= N

        rec_losses = []

        sh_losses = []

        joint_losses = []

        running_times = []

        for epoch in range(num_epochs):

            logger.info('epoch: %d' % epoch)

            idxlist = np.random.permutation(N)

            rec_loss_avg, sh_loss_avg, joint_loss_avg = 0., 0., 0.

            for batch in tqdm(range(nb_batches)):

                start, end = batch_indices(batch, N, batch_size)
                X = train_data[idxlist[start:end]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                batch_start_time = time.time()
                _, rec_loss_batch, sh_rec_loss_batch, joint_loss_batch = \
                    sess.run([trainer, rec_loss, sh_loss, joint_loss],
                             feed_dict={doc_word_ph: X, word_embeddings_ph: word_embeddings, rec_loss_weight: rec_loss_weight})
                running_times.append(time.time() - batch_start_time)
                if np.isnan(joint_loss_batch):
                    is_stop = True

                rec_loss_avg += rec_loss_batch
                sh_loss_avg += sh_rec_loss_batch
                joint_loss_avg += joint_loss_batch

                rec_losses.append(rec_loss_batch)
                sh_losses.append(sh_rec_loss_batch)
                joint_losses.append(joint_loss_batch)

                
            logger.info('joint_loss: %f' % (joint_loss_avg / nb_batches))

            if is_stop:
                logger.info('early stop because of NaN at epoch %d' % epoch)
                break



        [topic_embeddings, topic_word_mat] = sess.run([topic_embeddings_tf, topic_word_tf], feed_dict={word_embeddings_ph: word_embeddings})

        train_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, train_data, num_topics)

        test_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, test_data, num_topics)

        
        with open(os.path.join(output_dir, 'topics.txt'), 'a') as vis_file:
            print_topics(topic_word_mat, voc, printer=vis_file.write)
        

        train_doc_topic_path = output_dir / "train.theta.npy"
        test_doc_topic_path = output_dir / "test.theta.npy"
        beta_path = output_dir / "beta.npy"
        joint_losses_path = output_dir / "joint_losses.npy"
        topic_embeddings_path = output_dir / "topic_embeddings.npy"     

        np.save(beta_path, topic_word_mat)
        np.save(topic_embeddings_path, topic_embeddings)
        np.save(train_doc_topic_path, train_doc_topic)
        np.save(test_doc_topic_path, test_doc_topic)
        np.save(joint_losses_path, joint_losses)

        # import scipy.io
        # scipy.io.savemat(os.path.join(output_dir, 'save.mat'), {'phi': topic_word_mat,
        #                                                         'train_theta': train_doc_topic,
        #                                                         'test_theta': test_doc_topic,
        #                                                         'topic_embeddings': topic_embeddings,
        #                                                         'joint_losses': joint_losses})
        print("saved files")


if __name__ == '__main__':

    #list of params 
    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add("--run_embeddings_only", action="store_true", default=False)
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", required=True, default=None)
    parser.add("--output_dir", required=True, default=None)
    parser.add(
        "--jsonl_text_key", default="text", help="The key that has the document in it"
    )

    parser.add("--dtm_path", default="train.dtm.npz")
    #parser.add("--text_path", default="train.metadata.jsonl")
    parser.add("--embeds_path", default=None)
    parser.add("--vocab_path", default="vocab.json")
    parser.add("--topic_word_init_path", default=None)
    parser.add("--topic_word_prior_path", default=None)
    parser.add("--embeddings_model", default="glove")


    parser.add("--num_topics", default=50, type=int)  # n_components
    parser.add("--dropout", default=0.2, type=float)

    parser.add("--batch_size", default=64, type=int)
    parser.add("--num_epochs", default=100, type=int)
    parser.add("--learning_rate", default=2e-3, type=float)
    parser.add("--momentum", default=0.99, type=float)
    parser.add("--random_seed", default=42, type=int)


    #sinkhorn stuff 
    parser.add("--sh_alpha", default=20, type=int)
    parser.add("--sh_epsilon", default=0.001, type=float)
    parser.add("--sh_iterations", default=50, type=50)
    parser.add("--rec_loss_weight", default=0.07, type=float)

    args = parser.parse_args()

    #organize input paths 
    base_input_dir = Path(args.input_dir)
    train_dtm_path = base_input_dir / "train.dtm.npz"
    test_dtm_path = base_input_dir / "test.dtm.npz"
    vocab_path = base_input_dir / "vocab.json"

    #model inputs 
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_topics = args.num_topics 
    random_seed = args.random_seed
    batch_size = args.batch_size


    if args.embeds_path is None: 
        prefix = Path(args.dtm_path).stem.rstrip(".dtm") #train/test
        embeds_path = Path(args.input_dir, "embeddings", args.embeddings_model, f"{prefix}.embeds.npy")

    train_data, test_data, word_embeddings, voc = load_data_modified(
        model_path = "/workspace/.cache/glove/glove.6B.50d.txt", 
        is_to_dense=True, 
        train_dtm_path=train_dtm_path, 
        test_dtm_path=test_dtm_path, 
        embeds_path=embeds_path, 
        vocab_path=vocab_path, 
    )

    tf.app.run(run_ntsm(
        sh_epsilon=args.sh_epsilon, 
        sh_iterations=args.sh_iterations, 
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size, 
        num_topics=num_topics,
        random_seed=random_seed, 
        num_epochs=num_epochs, 
        rec_loss_weight=args.rec_loss_weight, 
        sh_alpha=args.sh_alpha,
        train_data=train_data,
        test_data=test_data,
        word_embeddings=word_embeddings, 
        voc=voc
    ))
