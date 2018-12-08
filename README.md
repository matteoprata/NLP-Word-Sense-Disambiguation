# Natural Language Processing Homework 2 by Matteo Prata

**I suggest to read the project files in this order:**

1. *XMLDocument.py* -
It is the structure of the datasets (either training, dev or test) represented in an object oriented fashion. It helps to understand how the datasets are accessed throughout the code. This structure turned out to be extremely convenient for managing all the datasets.

2. *DataProcessing.py* -
It is responsible for all the pre-processing of the datasets and the I/O operations, creation of the vocabularies, encoding of the sentences, batches generation...

3. *PretrainedWordEmbeddingsLoader.py* -
To understand how embeddings of the words in the vocabulary are retrieved from the official embeddings matrix.

4. *GatherBabelNetSynsets.java* -
A Java script to interact with the BabelNet API's and gather MFS's. 

5. *BiLSTMWSD.py* -
The neural model for the BiLSTM, it can be separately used for training, evaluating and testing. 

6. *EvaluationAndStats.py* -
Contains some subroutines and utilities of BiLSTMWSD.py for evaluation and printing statistics.

7. *DoubleBiLSTMWSD.py* -
The alternative model, the double layered BiLSTM.
