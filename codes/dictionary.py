import numpy as np
import codes.data_process
import os
import collections

class Dictionary():
    def __init__(self):
        self._dict_count = {}
        self._word_to_num = {} 
        self._num_to_word = {}

    def get_num_of_all_vocabs(self):
        return len(self._dict_count)

    def __len__(self):
        return len(self._word_to_num)

    def load_dict(self, name):
        print('load', name)
        self._dict_count, self._word_to_num, self._num_to_word = np.load(name)
        print("dictionary loaded")

    def save_dict(self, name):
        print("saving dictionary")
        os.makedirs("".join(name.split('/')[:-1]), exist_ok = True)
        np.save(name, (self._dict_count, self._word_to_num, self._num_to_word))
        print("save", name)

    def count(self, word):
        if self._dict_count.get(word) == None:
            self._dict_count[word] = 1
        else:
            self._dict_count[word] += 1

    def count_to_dict(self, limit_count = -float('inf'), limit_num_words = float('inf')):
        special_token = [
            '<unknown>',
            '<start>',
            '<end>',
            '<pad>',
            ]

        word_list = sorted(self._dict_count.items(), key=lambda x: x[1], reverse=True)
        num_of_words = 0
        self._word_to_num = {}
        self._num_to_word = {}
        for i in range(len(word_list)):
            if int(word_list[i][1]) <= limit_count:
                break
            elif num_of_words == limit_num_words - len(special_token):
                break
            else:
                self._word_to_num[word_list[i][0]] = num_of_words
                self._num_to_word[num_of_words] = word_list[i][0]
                num_of_words += 1
        for token in special_token:
            self._word_to_num[token] = num_of_words
            self._num_to_word[num_of_words] = token 
            num_of_words += 1

    def word_to_num(self, word):
        try:
            return self._word_to_num[word]
        except:
            return self._word_to_num['<unknown>']


    def translate_data_to_idx(self, text):
        trans = []
        for i in range(len(text)):
            line = []
            for j in range(len(text[i])):
                line.append(self.word_to_num(text[i][j]))
            trans.append(line)

        try:
            trans = np.array(trans).astype(int)
        except:
            trans = np.array(trans)
        return trans

    def translate_idx_to_data(self, text):
        trans = []
        for i in range(len(text)):
            trans.append(self._num_to_word[text[i]])
        return trans


