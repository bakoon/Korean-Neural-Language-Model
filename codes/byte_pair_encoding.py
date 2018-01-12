import pickle
from math import ceil
from collections import OrderedDict
from codes import data

class BPE():
    def __init__(self, data_type = 'jaso'):
        self._translate_dict_text_to_bpe = OrderedDict()
        self._translate_dict_bpe_to_text = {}
        self.n_vocab = 0

    def load_init_text(self, data_type = 'jaso'):
        corpus = data.Corpus(data_type, use_preprocessed_data = False)
        self.text = corpus._remove_newline(corpus.train_data)
        self.valid = corpus._remove_newline(corpus.valid_data)
        self.test = corpus._remove_newline(corpus.test_data)
        
        self.text = self._divide_into_word(self.text)
        self.valid = self._divide_into_word(self.valid)
        self.test = self._divide_into_word(self.test)

        """
        self.text = self._make_space_token(self.text)
        self.valid = self._make_space_token(self.valid)
        self.test = self._make_space_token(self.test)
        """

    def encoding(self, count_limit = 1, loop_limit = 99999, make_space_token = True):
        epoch = 0
        for _ in range(loop_limit):
            epoch += 1
            count_dict = self._count_pair(self.text)
            max_count_pair = max(count_dict, key=count_dict.get)
            if count_dict[max_count_pair] <= count_limit:
                break
            print("epoch", epoch, " pair :", "".join(max_count_pair), "count :", count_dict[max_count_pair])
            self.n_vocab += 1 
            self._translate_dict_text_to_bpe[max_count_pair] = '<bpe'+str(self.n_vocab)+'>'
            bpe_to_text = self._translate_bpe_pair_into_text(max_count_pair)
            self._translate_dict_bpe_to_text['<bpe'+str(self.n_vocab)+'>'] = bpe_to_text 

            self.text = self._translate_pair_to_bpe(self.text, max_count_pair)
#        if make_space_token:
#self.text = self._make_space_token(self.text)

    def translate_text_to_bpe(self, text, make_space_token = True):
        print("spliting text into words")
        text = [line.split() for line in text]
        pairlist = list(self._translate_dict_text_to_bpe.keys())
        for n_pair, pair in enumerate(pairlist):
#print("{:7}/{:7}".format(n_pair+1, len(pairlist)), end='\r')
            text = self._translate_pair_to_bpe(text, pair)
#        if make_space_token:
#            text = self._make_space_token(text)
        return text

    def translate_bpe_to_text(self, text):
        newtext = []
        for n_line, line in enumerate(text):
            newline = ''
            for char in line.split():
                if char == '<space>':
                    newline += ' '
                elif char[0] == '<':
                    newline += "".join(self._translate_dict_bpe_to_text[char])
                else:
                    newline += char
            newtext.append(newline)
        return newtext


    def save_text(self, in_text, name):
        print("save")
        text = self._make_space_token(in_text)
        txtfile = open(name, 'w')
        for n_line, line in enumerate(text):
#print("saving {:7}/{:7}".format(n_line+1, len(text)), end='\r')
            writeline = " <space> ".join([" ".join(word) for word in line])
            txtfile.write(writeline+'\n')
        print()
        txtfile.close()

    def load_text(self, name):
        print("loading text")
        self.text = open(name, 'r').readlines()
        self.text = [line.split(' <space> ') for line in self.text] 
        self.text = [[word.split() for word in line] for line in self.text]

    def save_dict(self, name):
        with open(name, 'wb') as output:
            pickle.dump((self._translate_dict_bpe_to_text, self._translate_dict_text_to_bpe), output)

    def load_dict(self, name):
        with open(name, 'rb') as input:
            self._translate_dict_bpe_to_text, self._translate_dict_text_to_bpe = pickle.load(input)
        self.n_vocab = len(self._translate_dict_text_to_bpe)

    def _count_pair(self, text):
        print("count")
        count_dict = {}
        for n_line, line in enumerate(text):
            for word in line:
                for n_char in range(len(word)-1):
                    pair = ( word[n_char], word[n_char+1] )
                    try:
                        count_dict[pair] += 1
                    except:
                        count_dict[pair] = 1
        return count_dict

    def _translate_pair_to_bpe(self, text, pair_to_translate):
        print("translate")
        newtext = []
        for n_line, line in enumerate(text):
            newline = []
            for word in line:
                translated = False
                newword = []
                for n_char in range(len(word)-1):
                    if translated == True:
                        translated = False
                    else:
                        pair = ( word[n_char], word[n_char+1] )
                        if pair == pair_to_translate:
                            translated = True
                            newword.append(self._translate_dict_text_to_bpe[pair_to_translate])
                        else:
                            newword.append(word[n_char])
                if translated == False:
                    newword.append(word[-1])
                newline.append(newword)
            newtext.append(newline)
        return newtext

    def _divide_into_word(self, text):
        for i in range(len(text)):
            text[i] = text[i].split()
            #text[i] = [x for x in text[i]]
        return text

    def _make_space_token(self, text):
        newtext = []
        for line in text:
            newline = list(line)
            for n_char in range(len(newline)):
                if newline[n_char] == ' ':
                    newline[n_char] = '<space>'
            newtext.append(newline)
        return newtext

    def _translate_bpe_pair_into_text(self, pair):
        if pair[0] != '<space>' and pair[0][0] == "<":
            text1 = self._translate_bpe_pair_into_text(self._translate_dict_bpe_to_text[pair[0]])
        else:
            text1 = pair[0]
        if pair[1] != '<space>' and pair[1][0] == "<":
            text2 = self._translate_bpe_pair_into_text(self._translate_dict_bpe_to_text[pair[1]])
        else:
            text2 = pair[1]
        return ("".join(text1), "".join(text2))

