import os
import numpy as np
from codes.dictionary import Dictionary
from random import shuffle

class Corpus(object):
    def __init__(self, data_type, limit_count = None, limit_num_words = None, dict_only = False, use_preprocessed_data = True, data_only = False):

        if limit_count is None:
            limit_count = 0

        if limit_num_words is None or limit_num_words == 0:
            limit_num_words = float('inf')
    
        self.data_type = data_type
        self.dictionary = Dictionary()
        self.train_count = 0    
        self.valid_count = 0    
        self.test_count = 0    
        self.use_preprocessed_data = use_preprocessed_data
   
        if use_preprocessed_data:
            datadir = './data/index/'
            train = 'kowiki_'+data_type+'_'+str(limit_count)+'_train_preprocessed.txt'
            valid = 'kowiki_'+data_type+'_'+str(limit_count)+'_valid_preprocessed.txt'
            test = 'kowiki_'+data_type+'_'+str(limit_count)+'_test_preprocessed.txt'
        else:
            datadir = './data/text/'
            read_file_data_type = data_type

            if data_type == 'penn':
                prefix = ''
            else:
                prefix = 'kowiki_'
            
            train = prefix+read_file_data_type+'_train.txt'
            valid = prefix+read_file_data_type+'_valid.txt'
            test = prefix+read_file_data_type+'_test.txt'

            if data_type == 'test':
                train = 'test_data.txt'
                valid = train
                test = train

        dictionary_location = './data/dictionary/'
    
        if not data_only:
            self.dictionary.load_dict(dictionary_location+data_type+'_dictionary.npy')
            #limit_count : only use words that counted over N
            #limit_num_words : only use N words that most counted
            self.dictionary.count_to_dict(limit_count, limit_num_words)
        
        if not dict_only:
            print("reading data")
            self.train_data = open(os.path.join(datadir, train), 'r').readlines()
            self.valid_data = open(os.path.join(datadir, valid), 'r').readlines()
            self.test_data = open(os.path.join(datadir, test), 'r').readlines()



    def _split_into_words(self, data):
        data = [line.split() for line in data]
#for i in range(len(data)):
#data[i] = data[i].split()           
        return data

    def _split_into_words_without_space_token(self,data):
        data = [[word for word in line.split() if word != '<space>'] for line in data]
        return data

    def _remove_newline(self, data):
        for i in range(len(data)):
            data[i] = data[i][:-1]
        return data

    def get_data_size(self):
        return {'train': len(self.train_data), "valid" : len(self.valid_data), "test" : len(self.test_data)}

    def shuffle_train_data(self):
        shuffle(self.train_data)

    def preprocess_data(self, data, get_index = True):
        if self.data_type == 'word':
            data = self._split_into_words_without_space_token(data)
        elif self.data_type == 'penn' or self.data_type[:4] in ['pos_', 'bpe_']:
            data = self._split_into_words(data)
        else:
            data = self._remove_newline(data)
        
        if self.data_type == 'penn':
            data = self.tokenize_penn(data)
        else:
            data = self.tokenize_padding(data, padding = False)
        
        if get_index:
            data = self.dictionary.translate_data_to_idx(np.array(data))
        
        return data

    def get_valid_data(self, n_data = None):
        if n_data is None:
            valid_data = list(self.valid_data)
        else:
            valid_data = self.valid_data[self.valid_count:self.valid_count+n_data]
            self.valid_count += n_data
            if self.valid_count >= len(self.valid_data):
                self.valid_count = 0

        if self.use_preprocessed_data:
            valid_data = self._split_into_words(valid_data)
        else:
            valid_data = self.preprocess_data(valid_data)
        
        return valid_data

    def get_train_data(self, n_data, get_index = True):
        if n_data is None:
            train_data = list(self.train_data)
        else:
            train_data = self.train_data[self.train_count:self.train_count+n_data]
            self.train_count += n_data
            if self.train_count >= len(self.train_data):
                self.train_count = 0

        if self.use_preprocessed_data:
            train_data = self._split_into_words(train_data)
        else:
            train_data = self.preprocess_data(train_data, get_index)
            
        return train_data

    def get_test_data(self, n_data = None):
        if n_data is None:
            test_data = list(self.test_data)
        else:
            test_data = self.test_data[self.test_count:self.test_count+n_data]
            self.test_count += n_data
            if self.test_count >= len(self.train_data):
                self.test_count = 0

        if self.use_preprocessed_data:
            test_data = self._split_into_words(test_data)
        else:
            test_data = self.preprocess_data(test_data)

        return test_data

                
    def tokenize_penn(self, data, padding = False):
        newdata = []
        for idx in range(len(data)):
            newline = list(data[idx]) + ['<end>']
            newdata.append(newline)
        return newdata

    def tokenize_padding(self, data, padding = False):
        if padding:
            size = []
            for line in data:
                size.append(len(line)+2) # +2 : include start and end token
            maxsize = max(size)

        newdata = []
        pad_idx = ['<pad>']

        for idx in range(len(data)):
            newline = ['<start>'] + list(data[idx]) + ['<end>']
            if padding:
                pad_size = maxsize - len(data[idx])
                newline = newline + pad_idx*pad_size
            newdata.append(newline)
        return newdata
 
