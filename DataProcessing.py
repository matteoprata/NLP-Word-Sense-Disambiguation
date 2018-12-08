import xml.etree.ElementTree as et
from XMLDocument import Corpus, Text, Sentence, Word
from enum import Enum
import PretrainedWordEmbeddingsLoader as pret_embs
import pickle
import numpy as np
from datetime import datetime
from collections import Counter
from EvaluationAndStats import stats_dataset

# Directories of train datasets
TRAIN_CORPUS_DIR = 'data/semcor.data.xml'
TRAIN_CLASSES_DIR = 'data/semcor.gold.key.bnids.txt'

# Directories of dev datasets
DEV_CORPUS_DIR = 'data/ALL.data.xml'
DEV_CLASSES_DIR = 'data/ALL.gold.key.bnids.txt'

# Directory of dev MFS
DEV_MFS = 'data/annotated_dev.txt'

# Directory of test dataset
TEST_CORPUS_DIR = 'data/test_data.txt'

# Directory of test MFS
TEST_MFS = 'data/annotated_test.txt'


# Directores of pre-trained wordemeddings
PRE_TRAINED_EMB_DIR_GOOGLE = 'data/GoogleNews-vectors-negative300.bin'
PRE_TRAINED_EMB_DIR_FASTTEXT = 'data/wiki-news-300d-1M.vec'
PRE_TRAINED_EMB_DIR_GLOVE = 'data/glove'

# How to map the UNK token in a pre-trained word embedding
UNK = 'UNK'


# ----------------------------------------------- ENUMERATORS ----------------------------------------------------------
class DatasetType(Enum):
    """
    The enumeration represents the possible datasets available.
    """
    TRAIN = (TRAIN_CORPUS_DIR, TRAIN_CLASSES_DIR)
    DEVELOPER = (DEV_CORPUS_DIR, DEV_CLASSES_DIR)
    TEST = TEST_CORPUS_DIR


class CorpusName(Enum):
    """
    The enumeration represents the possible corpus names.
    """
    SEM2 = 'senseval2'
    SEM3 = 'senseval3'
    SEM07 = 'semeval2007'
    SEM13 = 'semeval2013'
    SEM15 = 'semeval2015'


class EmbeddingsFamily(Enum):
    """
    The enumeration represents the possible embeddings matricies.
    """
    GLOVE = PRE_TRAINED_EMB_DIR_GLOVE
    GOOGLE = PRE_TRAINED_EMB_DIR_GOOGLE
    FASTTEXT = PRE_TRAINED_EMB_DIR_FASTTEXT


# ----------------------------------------------------- UTIL.S ---------------------------------------------------------

def serialize(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def reverse_dictionary(dictionary):
    reversed_dictionary = dict()
    for k in dictionary.keys():
        reversed_dictionary[dictionary[k]] = k
    return reversed_dictionary


def timeout(func_name):
    start_time = datetime.now()
    func_name()
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# -------------------------------------------------- PARSING  ----------------------------------------------------------
def parse_train_dev_corpus(corpus_type, limit=-1):
    """
    Given an enumerator representing the corpus we want to parse from an XML file, the function creates and returns
    a Corpus object.
    :param corpus_type: DatasetType enum - the corpus we want to parse.
    :return: Corpus object - representing the XML file.
    """

    tree = et.parse(corpus_type.value[0])  # generates the tree of the XML file stored in data_dir
    t_corpus = tree.getroot()

    # Parse the tree and create the relative objects defined the XMLDocument.py
    corpus = Corpus(source=t_corpus.attrib['source'], language=t_corpus.attrib['lang'])  # creates a Corpus object
    for t_text in t_corpus:
        text = Text(id=t_text.attrib['id'], source=t_text.attrib['source']) \
            if corpus.source == 'semcor' else Text(id=t_text.attrib['id'], source="")  # creates a Text object
    
        # to limit the size of the tdataset
        t_text = t_text[:limit] if limit > 0 else t_text  

        for t_sentence in t_text:
            sentence = Sentence(id=t_sentence.attrib['id'])  # creates a Sentence object

            for t_word in t_sentence:
                idd = t_word.attrib['id'] if t_word.tag == 'instance' else None

                word = Word(word=t_word.text, lemma=t_word.attrib['lemma'], pos=t_word.attrib['pos'],
                            type=t_word.tag, corpus=text.id.split('.')[0], idd=idd)  # creates a Word object

                sentence.add_word(word)

            text.add_sentence(sentence)
        corpus.add_text(text)

    # ADD LABELS TO THE CORPUS NOW
    # Once a Corpus object has been created, map each word to it's class, it's sense 
    
    # Fill a dictionary mapping a word id to it's meaning
    # while reading fromt the file
    meanings = dict()
    with open(corpus_type.value[1]) as file:  # read the file containing the classes
        for line in file.readlines():
            meanings[line.split()[0]] = line.split()[1]

    # For each instance word in the corpus associate to it its meaning read from the file
    # if the word is not a instance but a wf, associate the lemma as the meaning 
    for text in corpus.texts:
        for sentence in text.sentences:
            for word in sentence.words:
                if word.type == 'instance':
                    word.meaning = meanings[word.idd]
                else:
                    word.meaning = word.lemma  # if it's a wf, tag it with its lemma

    print('\nLoaded corpus:', str(corpus_type))
    stats_dataset(corpus)
    print()
    return corpus


def parse_test_corpus():
    """
    It produces the corpus of the test set.
    """

    corpus = Corpus(source='test_set', language='eng')
    text = Text(id='1', source='test_set')

    # Read the test set
    with open(TEST_CORPUS_DIR) as f:
        content = f.readlines()
    
    # Generate a more vague corpus (less deatils in the test dataset provided)
    for sent in content:
        words = sent.split()
        sentence = Sentence(id='none')

        for wd in words:
            comps = wd.split('|')
            
            # a chunk is an instance if is composed of 4 parts, else it is wf
            type = 'instance' if len(comps) == 4 else 'wf'
            id_ass = comps[3] if type == 'instance' else None

            wod = Word(word=comps[0], lemma=comps[1], pos=comps[2], type=type, corpus='test', idd=id_ass)

            sentence.add_word(wod)
        text.add_sentence(sentence)
    corpus.add_text(text)

    return corpus


def export_instances_to_file(corpus, fname):
    """
    Given a Corpus object and a file name, the funcion writes fname with all the informations of the instances of the corpus, 
    in order to get MFS from BableNet of those instances. The geneate file is then used by the Java script I attached. 
    """
    li = [] # Append here all the words objects that are instances
    
    # Iterate over the corpus to get all the instances
    for t in corpus.texts:
        for s in t.sentences:
            for wd in s.words:
                if wd.type == 'instance':
                    li.append(wd)

    # Write in the file instances in the from word_id|lemma|pos
    with open(fname, 'a') as the_file:
        for wd in li:
            the_file.write(wd.idd + "|" + wd.lemma + "|" + wd.pos + "\n")


# --------------------------- CREAZIONE DEI DIZIONARI -------------------------------

def dictionary_senses(corpus):
    """
    This generates the vocabulary of the senses within the training set.
    It returns a dictionary mapping a sense to its id. A special token UNK is mapped to 0.
    """
    
    dictionary_y = dict()  # SENSE - INDEX
    dictionary_y[UNK] = 0

    cls = 1
    for t in corpus.texts:
        for s in t.sentences:
            for wd in s.words:
                if not (wd.meaning in dictionary_y.keys()):  # If it wasn't already in the vocabulary
                    dictionary_y[wd.meaning] = cls           # add it and increment the counter of the id 
                    cls += 1

    return dictionary_y


def dictionary_words(corpus):
    """
    This generates the vocabulary of the words within the training set.
    It returns a dictionary mapping a word to its id. A special token UNK is mapped to 0.
    """
    
    dictionary_x = dict()  # WORD - INDEX
    dictionary_x[UNK] = 0

    cls = 1
    for t in corpus.texts:
        for s in t.sentences:
            for wd in s.words:
                if not (wd.lemma in dictionary_x.keys()):
                    dictionary_x[wd.lemma] = cls
                    cls += 1

    return dictionary_x


def dictionary_ambiguities(corpus):
    """
    This generates the vocabulary of the ambiguities of a word within the training set.
    It returns a dictionary mapping a word to all the senses that make it ambiguous.
    """
    dictionary_a = dict()  # SENSE - {SENSE: occ}
    
    for t in corpus.texts:
        for s in t.sentences:
            for wd in s.words: 
                if wd.type == 'instance' and not (wd.word_encoding == 0 or wd.meaning_encoding == 0): # If the word and the sense are not UNK 
                    if wd.word_encoding not in dictionary_a.keys():                                   
                        dictionary_a[wd.word_encoding] = Counter([wd.meaning_encoding])               # Add the new word in the vocabulary 
                    else:
                        dictionary_a[wd.word_encoding].update({wd.meaning_encoding: 1})               # The sense for this word was already met, increase its counter 
    return dictionary_a


def create_embeddings_dictionary(embedding_family, word_id_dict):
    """
    This function given a vocabulary mapping words to id's and an embedding matrix of choice (enumeration embedding_family), 
    returns an embeddings matrix and a vocabulary mapping words to ids in the embedding matrix. The embeddings are put from the whole orginal matrix 
    to the excerpt matrix if the words they represent are in the word vocabulary 'word_id_dict'.
    """
    
    # Read the pretrained word embeddings dictionaries
    if embedding_family == EmbeddingsFamily.GOOGLE:
        model = pret_embs.load_google_vectors(PRE_TRAINED_EMB_DIR_GOOGLE)

    elif embedding_family == EmbeddingsFamily.GLOVE:
        model = pret_embs.load_glove_vectors(PRE_TRAINED_EMB_DIR_GLOVE)

    elif embedding_family == EmbeddingsFamily.FASTTEXT:
        model = pret_embs.load_fasttext_vectors(PRE_TRAINED_EMB_DIR_FASTTEXT)

    # Generate word embeddings as a numpy array 
    embeddings = []
    # Generate word embeddings as a numpy array
    wemb_index = dict()  # WEMB - INDEX

    we_indx = 0
    for wd in word_id_dict.keys():

        if embedding_family == EmbeddingsFamily.GOOGLE and wd in model.vocab.keys():
            wemb_index[wd] = we_indx
            embeddings.append(model.wv[wd])
            we_indx += 1

        elif not embedding_family == EmbeddingsFamily.GOOGLE and wd in model.keys():
            wemb_index[wd] = we_indx
            embeddings.append(model[wd])
            we_indx += 1
    
    # the embedding matrix and the vocabulary mapping words to ids in the embedding matrix
    return np.array(embeddings), wemb_index

# --------------------------- CREAZIONE INPUT NN -------------------------------

def tag_corpus_MFS(file_annotations, corpus):
    """
    This function is intended to be used when you get an input file form the Java script accessing to the BabelNet API's.
    Given a file of annotated lemmas, with their most frequent sense, annotate the corpus by adding to each word of the input corpus 
    its most frequent sense. The input file should heve the form: word_id|MFS
    """
    
    idd_mfs = dict()
    with open(file_annotations) as f:
        for entry in f.readlines():
            entry_pt = entry.split('|')
            idd_mfs[entry_pt[0]] = entry_pt[1].strip()


    for text in corpus.texts:
        for sentence in text.sentences:
            for word in sentence.words:
                if word.type == 'instance':
                    word.most_frequent_sense = idd_mfs[word.idd]  # Associate to the word its most frequent sense
    return corpus


def encode_corpus(corpus, dictionary_senses, dictionary_embeddings):
    """
    This function, taking all the information in the vocabularies (words and senses), encode the corpus appropiately.
    If a word in the corpus (dev or test) is not in either the words voc. or the senses voc. the word is encoded as UNK, id 0.
    """
    for t in corpus.texts:
        for s in t.sentences:
            for wd in s.words:

                if wd.lemma in dictionary_embeddings.keys():  # La parola ha un embedding associato
                    wd.word_encoding = dictionary_embeddings[wd.lemma]
                else:
                    wd.word_encoding = dictionary_embeddings[UNK]

                if wd.meaning in dictionary_senses.keys():  # La parola ha un senso associato
                    wd.meaning_encoding = dictionary_senses[wd.meaning]
                else:
                    wd.meaning_encoding = dictionary_senses[UNK]

    # NOT USED
    # for t in corpus.texts:
    #     random.shuffle(t.sentences)

    print('\nLoaded corpus:', str('Encoded'))
    stats_dataset(corpus)
    print()
    return corpus


def flat_corpus(corpus):
    """
    Given a corpus, it generates a flat representation of it.
    It simply generates a list of lists, the inner lists are the sentences. Eache sentence is a list of Word objects.
    """
    sentences = list()

    for t in corpus.texts:
        for s in t.sentences:
            sent = list()
            for wd in s.words:
                sent.append(wd)
            sentences.append(sent)

    return sentences


def generate_batches(dataset, batch_size, seq):
    """
    It generates the batches.
    Given a dataset (the flat corpus, list of sentences in the corpus), the batch_size, and a sequence number (to remeber where we stop last time),
    the function generates a sub list from 'dataset' containing 'batch_size' sentences, 
    """
    
    subdataset = dataset[seq * batch_size:(seq + 1) * batch_size]

    # It will contain a list of padded sentences in which the words are encoded according to the words vocabulary encording
    batch_x = list()
    # It will contain a list of padded sentences in which the senses are encoded according to the senses vocabulary encording
    batch_y = list()
    # It will contain the true lengths of the words, one length for sentence in the batch
    sent_length = list()
    # It will contain a list of sentences cntaining Word objects
    batch_words_extended = list()

    for sent in subdataset:
        wds = []
        sen = []
        wds_ext = []

        for wd in sent:
            wds_ext.append(wd)
            wds.append(wd.word_encoding)
            sen.append(wd.meaning_encoding)

        batch_x.append(wds)
        batch_y.append(sen)
        batch_words_extended.append(wds_ext)
        sent_length.append(len(wds))

    maxlength = np.max(sent_length)
    
    # The padded versions of batch_x and batch_y
    padded_batch_x = []
    padded_batch_y = []
    for i in range(batch_size):
        padded_batch_x.append(np.pad(batch_x[i], (0, maxlength - len(batch_x[i])), 'constant', constant_values=(0, 0)))
        padded_batch_y.append(np.pad(batch_y[i], (0, maxlength - len(batch_y[i])), 'constant', constant_values=(0, 0)))

    # Returns the padded batches x and y, the real lengths of the padded sentences, and the batch with the true Word objects (not just the encoding)
    return padded_batch_x, padded_batch_y, sent_length, batch_words_extended


# ---------------------------------------------- PREPARE DATASETS -----------------------------------------------------


def get_datasets(limit_train=-1):
    """
    This funcion exploits all the above functions to generate, tag and encode: 
    the training set, the development set and the test set.
    """
    
    # ------------------------------------------ TRAINING -----------------------------------------------------------
    # TRAINING

    corpus_train = parse_train_dev_corpus(DatasetType.TRAIN, limit_train)

    sense_id = dictionary_senses(corpus_train)
    word_id = dictionary_words(corpus_train)

    embeddings_matrix, wemb_id = create_embeddings_dictionary(EmbeddingsFamily.FASTTEXT, word_id)

    encoded_corpus_train = encode_corpus(corpus_train, sense_id, wemb_id)
    flat_tr_corpus = flat_corpus(encoded_corpus_train)

    dictionary_ambiguitis = dictionary_ambiguities(encoded_corpus_train)

    # ------------------------------------------ DEVELOPMENT -----------------------------------------------------------
    # DEVELOPMENT

    corpus_dev = parse_train_dev_corpus(DatasetType.DEVELOPER)

    encoded_corpus_dev = encode_corpus(corpus_dev, sense_id, wemb_id)
    encoded_corpus_dev = tag_corpus_MFS(DEV_MFS, encoded_corpus_dev)

    flat_dev_corpus = flat_corpus(encoded_corpus_dev)

    # ------------------------------------------ TEST -----------------------------------------------------------
    # TEST

    corpus_test = parse_test_corpus()
    encoded_corpus_dev = encode_corpus(corpus_test, sense_id, wemb_id)
    encoded_corpus_test = tag_corpus_MFS(TEST_MFS, encoded_corpus_dev)
    flat_test_corpus = flat_corpus(encoded_corpus_test)
    
    # Returns a flat representation of the corpus (list of sentences) of training set, sev set, tst set. The senses voucbulary.
    # The dictionary of the ambiguities and the word embedding matrix.
    return flat_tr_corpus, flat_dev_corpus, flat_test_corpus, sense_id, dictionary_ambiguitis, embeddings_matrix

