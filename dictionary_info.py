import sys
from codes.data import Corpus

data_type = []
limit_count = []
num_words = []

#data_type.append('pos_Twitter')
#limit_count.append(range(0,1,1))
#data_type.append('pos_Hannanum')
#limit_count.append(range(0,1,1))
#data_type.append('pos_Mecab')
#limit_count.append(range(0,1,1))
#data_type.append('pos_Komoran')
#limit_count.append(range(0,1,1))
#data_type.append('pos_Kkma')
#limit_count.append(range(0,1,1))

#data_type.append('bpe_1000')
#limit_count.append(range(0,1,1))
#data_type.append('bpe_5000')
#limit_count.append(range(0,1,1))
#data_type.append('bpe_10000')
#limit_count.append(range(0,1,1))
data_type.append('bpe_15000')
limit_count.append(range(0,1,1))
data_type.append('bpe_20000')
limit_count.append(range(0,1,1))
#data_type.append('penn')
#limit_count.append(range(0,1,1))
#data_type.append('jaso_unkdata')
#limit_count.append(range(0,1,1))
#num_words.append(range(0, 1, 1))
#data_type.append('word_unkdata')
#limit_count.append(range(1, 5+1, 1))
#data_type.append('word')
#limit_count.append(range(0, 0+1, 1))
#data_type.append('char_unkdata')
#limit_count.append(range(0, 1, 1))

"""
data_type.append('pos_Hannanum')
ninety = 200
#limit_count.append(range(0, ninety + 1, ninety // 5))
limit_count.append(range(int(ninety*0.8), int(ninety*0.8+1), 1))

data_type.append('pos_Kkma')
ninety = 700
#limit_count.append(range(0, ninety + 1, ninety // 5))
limit_count.append(range(int(ninety*0.8), int(ninety*0.8+1), 1))

data_type.append('pos_Komoran')
ninety = 500
#limit_count.append(range(0, ninety + 1, ninety // 5))
limit_count.append(range(int(ninety*0.8), int(ninety*0.8+1), 1))

data_type.append('pos_Mecab')
ninety = 400
#limit_count.append(range(0, ninety + 1, ninety // 5))
limit_count.append(range(int(ninety*0.8), int(ninety*0.8+1), 1))

data_type.append('pos_Twitter')
ninety = 300
#limit_count.append(range(0, ninety + 1, ninety // 5))
limit_count.append(range(int(ninety*0.8), int(ninety*0.8+1), 1))
"""



def main(data_type = 'word', limit_count = 5, num_words = None):

    if num_words is None:
        corpus = Corpus(data_type, limit_count = limit_count, use_preprocessed_data = False)
    else:
        corpus = Corpus(data_type, limit_num_words = num_words, use_preprocessed_data = False)
    
    text = corpus.get_train_data(n_data = 99999999999999999999999, get_index = False)

    text_count = 0
    for line in text:
        text_count += len(line)
    
    dict_count = corpus.dictionary._dict_count

    full_vocab = len(dict_count)
    used_vocab = len(corpus.dictionary)

    try:
        unk_count = dict_count['<unk>']
    except:
        unk_count = 0
    
    keys = list(corpus.dictionary._word_to_num.keys())

    count_used = 0
    for key in keys:
        if key == '<space>' or key[:4] == '<bpe' or key[0] != '<':
            count_used += dict_count[key]
    full_items = list(dict_count.items())

    count_full = 0
    for item in full_items:
        count_full += item[1]

    if num_words is None:
        print("{}\nlimit_count {}\n\tvocab {}/{}({:.5f})\n\ttext {}/{}({:.5f})\n\tnum of lines {}\n\tvocab per line {} {}".format(data_type, limit_count, used_vocab, full_vocab, (used_vocab/full_vocab), count_used, count_full, (count_used/count_full), len(text), text_count / len(text), count_full / len(text)))
    else:
        print("{}\nnum_words {}\n\tvocab {}/{}({:.5f})\n\ttext {}/{}({:.5f})\n\tnum of lines {}\n\tvocab per line {} {}".format(data_type, num_words, used_vocab, full_vocab, (used_vocab/full_vocab), count_used, count_full, (count_used/count_full), len(text), text_count / len(text), count_full / len(text)))

if __name__ == "__main__":
    if num_words != [] and limit_count != []:
        print("select one limit method")
    elif limit_count != []:
        for i in range(len(data_type)):
            for j in range(len(limit_count[i])):
                main(data_type[i], limit_count[i][j])
                print()
    elif num_words != []:
        for i in range(len(data_type)):
            for j in range(len(num_words[i])):
                main(data_type[i], num_words = num_words[i][j])
                print()
    else:
        print("select one limit method")

