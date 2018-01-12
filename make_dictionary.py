import sys
from codes import data
import os

def main():
#    make_dictionary('jaso_unkdata')
#    make_dictionary('word')
#    make_dictionary('pos_Twitter')
#    make_dictionary('pos_Kkma')
#    make_dictionary('pos_Hannanum')
#    make_dictionary('pos_Komoran')
#    make_dictionary('pos_Mecab')
#    make_dictionary('char_unkdata')
#make_dictionary('test')
#    make_dictionary('penn')



    for i in range(15000, 20001, 5000):
        make_dictionary('bpe_'+str(i))

#    for i in range(1800, 3001, 100):
#        make_dictionary('bpe_'+str(i))



def make_dictionary(data_type):
    print(data_type)

    corpus = data.Corpus(data_type, use_preprocessed_data = False, data_only = True)
    my_dict = corpus.dictionary

    filedata = corpus.train_data 
    for n_data, linedata in enumerate(filedata):
        print("{:3}/{:3}".format(n_data+1, len(filedata),), end='\r') 
        if data_type[:4] == 'word':
            linedata = [word for word in linedata.split() if word != '<space>']
        elif data_type[:4] in ['penn', 'bpe_'] or data_type[:4] in ['pos_', 'bpe_']:
            linedata = linedata.split()
        elif data_type[:4] in ['jaso', 'char', 'test',]:
            linedata = linedata[:-1] # remove newline char
        else:
            print("set preprocessing method for", data_type)

        for i, count_data in enumerate(linedata):
            my_dict.count(count_data)
    my_dict.save_dict('./data/dictionary/'+data_type+'_dictionary.npy')


if __name__ == '__main__':
    main()
