
from enum import Enum


class EvaluationCase(Enum):
    """
    The enumeration represents the possible situations when evaluating the model.
    """
    AMBIGUOUS = 1                       # COUNT total ambiguous words (denominator Recall)
    AMBIGUOUS_EVALUABLE = 2             # COUNT total ambiguous evaluable words (denominator Precision)
    AMBIGUOUS_EVALUATED_CORRECT = 0     # COUNT total ambiguous well classified (numerator Precision and Recall)

def stats_dataset(corpus):
    """
    Given a corpus object (from XMLDocument.py file) print some stats like the number of texts, the number of sentences,
    the number of words, the number of wf, and the number of instances whithin the corpus.
    :param corpus: Corpus object - the object representing the corpus either of the training dataset, testing dataset or development dataset.
    """
    n_texts = 0
    n_sentences = 0
    n_words = 0
    n_wf = 0
    n_instances = 0

    for t in corpus.texts:  # Iterates over the texts of the corpus
        n_texts += 1
        for s in t.sentences:  # Iterates over the sentences of the corpus
            n_sentences += 1
            for wd in s.words:  # Iterates over the words of the corpus
                n_words += 1
                if wd.type == 'wf':
                    n_wf += 1
                elif wd.type == 'instance':
                    n_instances += 1

    print('n_texts:', n_texts, 'n_sentences:', n_sentences, 'n_words:', n_words, 'n_wf:', n_wf, 'n_instances:', n_instances)


def increment_counters(counters, X_ext, case):
    """
    This function is responsible for incrementing the counters for computing the statistics. Civen a dictionary of counters,
    it increments the counter per corpus, depending on tehe word object.
    :param counters: a dictionary of counters
    :param X_ext: a Word object
    :param case: an EvaluationCase to understand what event triggered increment of the counter
    """

    # take the name of the corpus of the word and then th relative counter
    tuple = counters[X_ext.corpus]
    # take the global counter
    g_tup = counters['global']

    # increment either the global counter or the counter or the corpus
    if case.value == 0:  # correct
        counters[X_ext.corpus] = (tuple[0]+1, tuple[1], tuple[2])
        counters['global'] = (g_tup[0]+1, g_tup[1], g_tup[2])

    elif case.value == 1:  # total ambiguous
        counters[X_ext.corpus] = (tuple[0], tuple[1]+1, tuple[2])
        counters['global'] = (g_tup[0], g_tup[1]+1, g_tup[2])

    elif case.value == 2:   # total ambiguous possibly disambiguable
        counters[X_ext.corpus] = (tuple[0], tuple[1], tuple[2]+1)
        counters['global'] = (g_tup[0], g_tup[1], g_tup[2]+1)


def print_recall_precion_f1(counters):
    """
    Given a dictionary of counters it prints the staistics in a readble way.
    """
    print('\nStats now')

    for key in counters.keys():
        co = counters[key]

        R_c = co[0] / co[1]
        P_c = co[0] / co[2]
        F1_c = 2 * P_c * R_c / (P_c + R_c)

        print()
        print('Statistics for', key)
        print("Recall:", R_c, "(" + str(co[0]) + "/" + str(co[1]) + ")")
        print("Precision:", P_c, "(" + str(co[0]) + "/" + str(co[2]) + ")")
        print("F1:", F1_c, "(" + str(2 * P_c * R_c) + "/" + str(P_c + R_c) + ")")
        print()
