
class Corpus:
    """
    This class represents the corpus of the input XML document. The object keeps informations like the language of the
    corpus, the source and a list of Text objects contained in it.
    """

    source = None
    language = None
    texts = None

    def __init__(self, source, language, texts):
        self.id = id
        self.source = source
        self.language = language
        self.texts = texts

    def __init__(self, source, language):
        self.source = source
        self.language = language
        self.texts = list()

    def add_text(self, text):
        self.texts.append(text)


class Text:
    """
    This class represents a text in the corpus. The object keeps informations like the id, the source of the text and a
    list of Sentence objects contained in it.
    """

    id = None
    source = None
    sentences = None

    def __init__(self, id, source, sentences):
        self.id = id
        self.source = source
        self.sentences = sentences

    def __init__(self, id, source):
        self.id = id
        self.source = source
        self.sentences = list()

    def add_sentence(self, sent):
        self.sentences.append(sent)


class Sentence:
    """
    This class represents a sentence of a text in the corpus. The object keeps informations like the id and a list of
    Word objects contained in it.
    """

    id = None
    words = None

    def __init__(self, id, words):
        self.id = id
        self.words = words

    def __init__(self, id):
        self.id = id
        self.words = list()

    def add_word(self, wd):
        self.words.append(wd)


class Word:
    """
    This class represents a word of a sentence in a text in the corpus. The object keeps informations like the lemma of
    the word, the part of speech tag, wheather it is an instance or a wf, the meaning (lemma for wf and BabelNetSynstID
    otherwise), how the lemma is encoded in a vocabulary, how the meaning is encoded with a vocabulary and the most
    frequent sense of the lemma.
    """

    idd = None
    word = None
    lemma = None
    pos = None
    type = None
    meaning = None
    corpus = None
    word_encoding = None
    meaning_encoding = None
    most_frequent_sense = None

    def __init__(self, word, lemma, pos, type, corpus, idd):
        self.idd = idd
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.type = type
        self.meaning = ""
        self.corpus = corpus

    def __repr__(self):
        """
        For printing the word explicitly.
        :return: String - e.g. (bank, instance, senseval2) [wdenc: 8, senc: 7000] -> bn:xxxxx
        """
        return "(" + self.lemma + ", " + self.type + ", " + self.corpus + ") " + \
               "[wdenc:" + str(self.word_encoding) + ", senenc:" + str(self.meaning_encoding) + "] -> " + self.meaning








