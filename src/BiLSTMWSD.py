import tensorflow as tf
from DataProcessing import get_datasets, generate_batches, serialize, deserialize, reverse_dictionary
from tqdm import tqdm
import EvaluationAndStats as eval
import datetime as date
from enum import Enum


class NeuralUsage(Enum):
    """
    The enumeration represents the possible situations when using the neural architecture,
    for training, evaluating or testing.
    """
    TRAIN = 0
    EVALUATE = 1
    TEST = 2


def write_predictions_testing(instanceid_prediction):
    """
    Given a list of tuples (intance_id, predicted_sense) it writes them in a file.
    :param instanceid_prediction: list((String, String)) - list of tuples (intance_id, predicted_sense)
    """
    with open('1593974_test_answer.txt', 'a') as the_file:
        for instance, prediction in instanceid_prediction:
            the_file.write(instance + '\t' + prediction + '\n')


def arg_max_at_indicies(source, indxs):
    """
    Given a list of indicies 'indxs' the function isolates the values pointed by those indicies in 'source' list
    and then computes the maximum of those values, the function returns the index of the maximum value among the isolated ones.
    :param source: list(float) - the probability distribution over the senses.
    :param indxs: list(int) - list of indicies of the senses that make a certain word ambiguous.
    :return: (int, float) - the touple containing the index having maximum probability and the probability associated to that index
    """

    values = []
    for ind in indxs:
        values.append((source[ind], ind))

    max_prob = -1
    max_ind = -1
    for val, ind in values:
        if val > max_prob:
            max_prob = val
            max_ind = ind

    return max_ind, max_prob

# ------------------------------------------- NEURAL NETWORK SETUP ----------------------------------------------------

# specify the modality of accessing the neural architecture
MODALITY = NeuralUsage.EVALUATE

# specify whether or not we need BabelNet integration of the MFS's
BABELNET_INTEGRATION = False

# file of a stored model
MODEL_DIRECTORY = "/tmp/model20180610195415-5.ckpt"

# gets now date and time
DATE = str(date.datetime.now().strftime("%Y%m%d%H%M%S"))

# Do I really want to discard the NN prediction?
THRESHOLDING_CLASSIFICATION = False

# Generates 1) the list od sentences of the training set 2) - of the dev set 3) - of of the test set
#           4) the dictionary mapping senses to their IDs 5) the dictionary mapping a word to the senses that make it ambiguous
#           6) the embedding matrix
flat_tr_corpus, flat_dev_corpus, flat_test_corpus, sense_id, dictionary_ambiguitis, embeddings_matrix = get_datasets()
# deserialize()

# The hyper-parameters
# the size of the latent representation of the context of a word
HIDDEN_SIZE = 128
# the size of each batch
BATCH_SIZE = 4
# the learning rate for the optimizer
LR = 2
# the number of epochs
EPOCHS = 10
# mean acceptance probability for a prediction
MEAN_ACC = 0.0878

# size of the embedding vectors
EMEBDDINGS_SIZE = 300
# possible number of senses to make predictions
N_SENSES = len(sense_id.keys())
# number of embedding vectors (number of distinct words appearing in the training set)
W_EMEBDDINGS = len(embeddings_matrix)


# ------------------------------------------- START TENSORFLOW GRAPH  --------------------------------------------------
# Described on the report

# shape = [BATCH_SIZE, TIME-STEP]
input_words_ids = tf.placeholder(tf.int32, shape=[None, None], name="words_ids")

# shape = [BATCH_SIZE]
sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="seq_lengths")

# shape = [VOCAB-SIZE, EMEBDDINGS_SIZE]
lookup = tf.Variable(embeddings_matrix, dtype=tf.float32, trainable=False)

# shape = [BATCH_SIZE, TIME-STEP, EMEBDDINGS_SIZE]
pretrained_emb = tf.nn.embedding_lookup(lookup, input_words_ids)

# LSTM layer 1
cell_fw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
# LSTM layer 2
cell_bw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
(ofw, obw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pretrained_emb, sequence_length=sequence_lengths,
                                                dtype=tf.float32)

# shape = [BATCH_SIZE, TIME-STEP, 2*HIDDEN_SIZE]
# For each batch for each sentence for each word we compute its context representation
context_rep = tf.concat([ofw, obw], axis=-1)

# -------------------------------------------- DECODE CONTEXT ---------------------------------------------------------

W = tf.get_variable("W", shape=[2 * HIDDEN_SIZE, N_SENSES], dtype=tf.float32)
b = tf.get_variable("b", shape=[N_SENSES], dtype=tf.float32, initializer=tf.zeros_initializer())

context_rep_flat = tf.reshape(context_rep, [-1, 2 * HIDDEN_SIZE])

predictions = tf.matmul(context_rep_flat, W) + b
predictions_scores = tf.reshape(predictions, [-1, tf.shape(context_rep)[1], N_SENSES])

# shape = [BATCH_SIZE, TIME-STEP, N_SENSES]
# Probability distribution over the senses
softmax_score = tf.nn.softmax(predictions_scores)

# shape = [BATCH_SIZE, TIME-STEP]
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

# measures the probability error
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_scores, labels=labels)
loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(sequence_lengths)))

optimizer = tf.train.GradientDescentOptimizer(LR)

train_op = optimizer.minimize(loss)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# --------------------------------------------- END TENSORFLOW GRAPH  --------------------------------------------------



if MODALITY == NeuralUsage.TRAIN:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # --------------------------------------------- TRAIN  ---------------------------------------------------

        print("Start training!")

        for epoch in range(EPOCHS):
            print('Epoch:', epoch, '/', EPOCHS)
            BATCHES_MAX = int(len(flat_tr_corpus) / BATCH_SIZE)
            loss_sum = 0

            for i in tqdm(range(BATCHES_MAX)):

                # Generates a list of size BATCH_SIZE of padded sentences containing encoded 1) words 2) senses. 3) A list of sequences, true lenghts of the words
                batchx, batchy, sl, _ = generate_batches(flat_tr_corpus, BATCH_SIZE, i)
                _, loss_value = sess.run([train_op, loss], feed_dict={input_words_ids: batchx, labels: batchy, sequence_lengths: sl}) # Invocation of the TRAINING operation train_op
                loss_sum += loss_value

            # evaluate()  # Invocation to an intermediate evaluation, not reported for clearity

            print('avg loss:', loss_sum / BATCHES_MAX)

        save_path = saver.save(sess, "/tmp/model" + DATE + ".ckpt")
        print("Model saved in path: %s" % save_path)
        print("End of training!")



elif MODALITY == NeuralUsage.EVALUATE:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_DIRECTORY)

        # --------------------------------------------- EVALUATE  ---------------------------------------------------

        print("Start evaluating!")
        evaluation_batch = 1

        counters = {'global': (0, 0, 0), 'senseval2': (0, 0, 0), 'senseval3': (0, 0, 0), 'semeval2007': (0, 0, 0), 'semeval2013': (0, 0, 0), 'semeval2015': (0, 0, 0)}

        for i in range(int(len(flat_dev_corpus) / evaluation_batch)):

            # Generates a list of size 1 of padded sentences containing encoded 1) words 2) senses
            # 3) A list of sequences, true lenghts of the words 4) a list of size 1 of sentences containing explicit words objects, with several accessible fields (see XMLDocument.py).
            batchxD, batchyD, slD, batchD_extended = generate_batches(flat_dev_corpus, evaluation_batch, i)
            predictions_forinst = sess.run(softmax_score, feed_dict={input_words_ids: batchxD, sequence_lengths: slD})  # Invocation of the EVALUATION operation

            for j in range(slD[0]):  # Iterate over the words in the input sequence

                # Access to the extended version of an input word
                X_ext = batchD_extended[0][j]
                X_lem_enc = X_ext.word_encoding
                X_sen_enc = X_ext.meaning_encoding
                Y_truth = batchyD[0][j]

                # If the word should be disambiguated
                if X_ext.type == 'instance':

                    # Word is an instance to disambiguate, count it
                    eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS)

                    # Word is an instance to disambiguate and the neural network can disambiguate it because:
                    # 1) the word is not UNK 2) the sense is not UNK 3) the word is in the dictionary of the ambiguities
                    if not (X_lem_enc == 0 or X_sen_enc == 0) and X_lem_enc in dictionary_ambiguitis.keys():

                        # Word is an instance that I can disambiguate count it
                        eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUABLE)

                        # A list of encoded senses sorted by the most frequent within the training set
                        Xambs, _ = zip(*dictionary_ambiguitis[X_lem_enc].most_common())

                        # Y = index of the sense having the highest probability within the senses in Xambs
                        # acceptance_probability = probability that motivated the choice of Y
                        Y, acceptance_probability = arg_max_at_indicies(predictions_forinst[0][j], Xambs)

                        # We are ready to give a prediction...

                        # If I allow the BabelNet integration and the classication based on MEAN_ACC
                        if THRESHOLDING_CLASSIFICATION and BABELNET_INTEGRATION:
                            # The meaning hase been chosen with high enough probability
                            if acceptance_probability >= MEAN_ACC:
                                if Y == Y_truth:
                                    eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUATED_CORRECT)

                            # The meaning hase been chosen with low probability, use MFS on BabelNet
                            else:
                                # If the MFS is correct count it as correct
                                if X_ext.meaning == X_ext.most_frequent_sense:
                                    eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUATED_CORRECT)

                        # I don't allow BabelNet integration and the classication based on MEAN_ACC, I predict with NN
                        else:
                            if Y == Y_truth:
                                eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUATED_CORRECT)

                    # The word is ambiguous but the NN couln't train the word it's sense
                    else:
                        if BABELNET_INTEGRATION:
                            # Word is an instance that I can disambiguate count it
                            eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUABLE)

                            # If the MFS is correct count it as correct
                            if X_ext.meaning == X_ext.most_frequent_sense:
                                eval.increment_counters(counters, X_ext, eval.EvaluationCase.AMBIGUOUS_EVALUATED_CORRECT)

        # Print stats
        eval.print_recall_precion_f1(counters)



elif MODALITY == NeuralUsage.TEST:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_DIRECTORY)

        # ------------------------------------------------ TEST  ---------------------------------------------------

        print("Start testing!")
        testing_batch = 1

        instanceid_precition = []
        for i in range(int(len(flat_test_corpus) / testing_batch)):

            # Generates a list of size 1 of padded sentences containing encoded 1) words
            # 3) A list of sequences, true lenghts of the words 4) a list of size 1 of sentences containing explicit words objects, with several accessible fields (see XMLDocument.py).
            batchxT, _, slT, batchT_extended = generate_batches(flat_test_corpus, testing_batch, i)
            predictions_forinst = sess.run(softmax_score, feed_dict={sequence_lengths: slT, input_words_ids: batchxT})  # Invocation of the EVALUATION operation

            for j in range(slT[0]):  # Iterate over the words in the input sequence

                X_ext = batchT_extended[0][j]
                X_lem_enc = X_ext.word_encoding
                X_sen_enc = X_ext.meaning_encoding

                # If the word should be disambiguated
                if X_ext.type == 'instance':

                    # Word is an instance to disambiguate and the neural network can disambiguate it because:
                    # 1) the word is not UNK 2) the sense is not UNK 3) the word is in the dictionary of the ambiguities
                    if not (X_lem_enc == 0 or X_sen_enc == 0) and X_lem_enc in dictionary_ambiguitis.keys():

                        # A list of encoded senses sorted by the most frequent within the training set
                        Xambs, _ = zip(*dictionary_ambiguitis[X_lem_enc].most_common())

                        # Y = index of the sense having the highest probability within the senses in Xambs
                        # acceptance_probability = probability that motivated the choice of Y
                        Y, acceptance_probability = arg_max_at_indicies(predictions_forinst[0][j], Xambs)

                        # We are ready to give a prediction...

                        # If I allow the BabelNet integration and the classication based on MEAN_ACC
                        if THRESHOLDING_CLASSIFICATION and BABELNET_INTEGRATION:
                            # The meaning hase been chosen with high enough probability
                            if acceptance_probability >= MEAN_ACC:
                                instanceid_precition.append((X_ext.idd, reverse_dictionary(sense_id)[Y]))

                            # The meaning hase been chosen with low probability, use MFS on BabelNet
                            else:  instanceid_precition.append((X_ext.idd, X_ext.most_frequent_sense))

                         # I don't allow BabelNet integration and the classication based on MEAN_ACC, I simply predict with NN
                        else: instanceid_precition.append((X_ext.idd, reverse_dictionary(sense_id)[Y]))

                    else:
                        if BABELNET_INTEGRATION:
                            instanceid_precition.append((X_ext.idd, X_ext.most_frequent_sense))

        # Write the results of the test as output
        write_predictions_testing(instanceid_precition)