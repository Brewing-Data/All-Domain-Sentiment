import gensim
import itertools
import pandas as pd
import numpy as np
import warnings

class W2VFeaturizer:
    UNKNOWN_WORD = '--UNK--'
    PAD_WORD = '--PAD--'
    MAXL = None
    
    def __init__(self,model_path):
        self.model_path = model_path
        self.__trained = False
        # Load Google's pre-trained Word2Vec model
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)

    @staticmethod
    def tokenize_text(doc_ser):
        token_doc_ser = doc_ser.str.lower().apply(word_tokenize)
        return token_doc_ser
    
    @staticmethod
    def get_max_doc_len(ser_tokens):
        maxl = ser_tokens.apply(lambda x: len(x)).max()
        return maxl
    
    def get_padding(self, tok_list):
        pad_size = self.MAXL - len(tok_list)
        ret_list = list(tok_list)
        if pad_size > 0:
            for i in range(pad_size):
                ret_list.append(self.PAD_WORD)
        else:
            ret_list = list(tok_list)[:self.MAXL]
        return ret_list
    
    def get_vocab_from_texts(self, text_2dlist):
        print("--> Getting vocab from text")
        text_list = list(itertools.chain.from_iterable(text_2dlist))
        set_vocab = set(text_list)
        list_vocab = list(set_vocab)
        list_vocab.insert(0, self.UNKNOWN_WORD)  # vocab word at index 1 will be unknown  word
        list_vocab.insert(0, self.PAD_WORD)  # vocab word at index 0 will be pad word
        return list_vocab
    
    def vector_reprn_of_text(self, padded_sent_ser, loaded_w2v_dim):
        print("Converting text into numeric form ------->")
        sent_ser = []
        for word in padded_sent_ser:
            if word in self.subset_vocab and  word in self.model.wv.vocab:
                word_vec = self.model.wv.syn0[self.model.wv.vocab[word].index]
#                 print(len(word_vec))
                sent_ser.append(word_vec)
            else:                
                sent_ser.append(random_vector)
#                 self.UNKNOWN_WORD for word in sent]
        return sent_ser
    
    def apply(self, sent_ser): # will be used for a test set, when the vocab has already been fixed
        assert isinstance(sent_ser, pd.Series), "The input document is not a Series, pass a Series !"
        tok_test_doc = pd.Series(W2VFeaturizer.tokenize_text(sent_ser))
        padded_tok_ser = tok_test_doc.apply(self.get_padding)
        loaded_w2v_dim = self.model.wv.syn0.shape[1]
        vec_ser = padded_tok_ser.apply(self.vector_reprn_of_text, args=(loaded_w2v_dim,))
        return vec_ser
        
        
    def train(self, raw_doc): # will fix a vocabulary based on the training data
        if self.__trained == True:
            warnings.warn('Model already trained !!')
            return 
        else:
            self.__trained = True
            assert isinstance(raw_doc, pd.Series), "The input document is not a Series, pass a Series !"
            token_doc_ser = W2VFeaturizer.tokenize_text(raw_doc)
            self.MAXL = W2VFeaturizer.get_max_doc_len(token_doc_ser)
            self.subset_vocab = self.get_vocab_from_texts(pd.Series(token_doc_ser))
            
            
            