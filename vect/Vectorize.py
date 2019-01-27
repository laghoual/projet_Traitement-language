from typing import List

import numpy as np
from gensim.models import KeyedVectors

from Token.Tokenize import Document


class Vectorizer:
    """ Transformer a string into a vector representation"""

    def __init__(self, word_embedding_path: str):
        """
        :param word_embedding_path: path to gensim embedding file  Lier un vecteur a chaque mot
        """

        self._word_embeddings = KeyedVectors.load_word2vec_format(word_embedding_path)
        # Create shape to index dictionary
        self.shape_to_index = {'NL': 0, 'NUMBER': 1, 'SPECIAL': 2, 'ALL-CAPS': 3, '1ST-CAP': 4, 'LOWER': 5, 'MISC': 6}
        # Create labels to index
        self.pos_to_index = {"$":37,"''": 45,'"':38,"(":39,")":40,",":41,"--":42,".":43,":":44,"CC":0,"CD":1,"DT":2,"EX":3,"FW":4,
                          "IN":5,"JJ":6,"JJR":7,"JJS":8,"LS":9,"MD":10,"NN":11,"NNP":12,"NNPS":13,"NNS":14,"PDT":15,
                          "POS":16,"PRP":17,"PRP$":18,"RB":19,"RBR":20,"RBS":21,"RP":22,"SYM":23,"TO":24,"UH":25,
                          "VB":26,"VBD":27,"VBG":28,"VBN":29,"VBP":30,"VBZ":31,"WDT":32,"WP":33,"WP$":34,"WRB":35,
                          "``":36,"NN|SYM":46}
        #self.labels_to_index= {'o':0, 'PER':1, 'I-PER':1, 'B-PER':1, 'LOC':2, 'I-LOC':2, 'B-LOC':2, 'ORG':3, 'I-ORG':3,
        #                       'B-ORG':3, 'MISC':4, 'I-MISC':4, 'B-MISC':4}
        #self.labels = ['o','PER','LOC','ORG','MISC']


    def encode_features(self, documents: List[Document]):
        """
        Creer  a feature matrix POUR  tous les  documents in the sample list
        :param documents: list of all samples as document objects
        :return: lists of numpy arrays for word, pos and shape features.
                 chaque  item dans  la liste est  une   sentence,
        """
        words=[]
        shapes=[]
        # boucler over documents
        for doc in documents:
        #    boucler over sentences
            for sentence in doc.sentences:
                sentence_words=[]
                sentence_shapes=[]
        #        Loop over tokens
                for token in sentence.tokens:
        #           Convert features to indices
        #           Append to sentence
                    if token.text.lower() in self._word_embeddings.vocab:
                        sentence_words.append(self._word_embeddings.index2word.index(token.text.lower()))
                        sentence_shapes.append(self.shape_to_index[token.shape])
                words.append(sentence_words)
                shapes.append(sentence_shapes)
        return np.asarray(words), np.asarray(shapes)

    def encode_annotations(self, documents: List[Document]):
        """
        on cree  the Y matrix qui represente l annotations (or true positives) pour  une liste de documents
        :param documents: list of documents à etre  dans les annotations vector
        :return: numpy array. Each item in the list is a sentence, i.e. a list of labels (one per Token)
        """
        labels = []
        # boucler over documents
        for doc in documents:
            #    boucler over sentences
            for sentence in doc.sentences:
                sentence_labels = []
                #        boucler over tokens
                for token in sentence.tokens:
                    #           Convertir features to indices
                    #           ajouter  to sentence
                    if token.text.lower() in self._word_embeddings.vocab:
                        sentence_labels.append(self.pos_to_index[token.label])
                labels.append(sentence_labels)
        return np.asarray(labels)
