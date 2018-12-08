Hello, I suggest to read the project files in this order:

- XMLDocument.py 
It is the structure of the datasets (either training, dev or test) represented in an object oriented fashion. It helps to understand how the datasets are accessed throughout the code. This structure turned out to be extremely convenient for managing all the datasets.

- DataProcessing.py
It is responsible for all the pre-processing of the datasets and the I/O operations, creation of the vocabularies, encoding of the sentences, batches generation...

- PretrainedWordEmbeddingsLoader.py
To understand how embeddings of the words in the vocabulary are retrieved from the official embeddings matrix.

- GatherBabelNetSynsets.java
A Java script to interact with the BabelNet API's and gather MFS's. 

- BiLSTMWSD.py
The neural model for the BiLSTM, it can be separately used for training, evaluating and testing. 

- EvaluationAndStats.py
Contains some subroutines and utilities of BiLSTMWSD.py for evaluation and printing statistics.

- DoubleBiLSTMWSD.py
The alternative model, the double layered BiLSTM.

Thank you.