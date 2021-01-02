import tensorflow as tf
from DataProcessing import get_datasets, generate_batches
from tqdm import tqdm

from enum import Enum

# ------------------------------------------- NEURAL NETWORK SETUP ----------------------------------------------------

# Generates 1) the list od sentences of the training set 6) the embedding matrix
flat_tr_corpus, _, _, sense_id, _, embeddings_matrix = get_datasets()
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

with tf.variable_scope('LAYER-1'):

    cell_fw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
    cell_bw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
    (ofw, obw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pretrained_emb, sequence_length=sequence_lengths, dtype=tf.float32)

    context_rep = tf.concat([ofw, obw], axis=-1)


with tf.variable_scope('LAYER-2'):
    cell_fw_l2 = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
    cell_bw_l2 = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)
    (ofw_2, obw_2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_l2, cell_bw_l2, context_rep, dtype=tf.float32)

    context_rep_l2 = tf.concat([ofw_2, obw_2], axis=-1)

W = tf.get_variable("W", shape=[2*HIDDEN_SIZE, N_SENSES], dtype=tf.float32)
b = tf.get_variable("b", shape=[N_SENSES], dtype=tf.float32, initializer=tf.zeros_initializer())

context_rep_flat = tf.reshape(context_rep_l2, [-1, 2*HIDDEN_SIZE])

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

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print("End of training!")
