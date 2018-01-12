from codes.data import Corpus
import os


param_set = [

#    ('bpe_1000', 0),
    ('bpe_15000', 0),
    ('bpe_20000', 0),
#        ('pos_Hannanum', 0),
#        ('pos_Twitter', 0),
#        ('pos_Mecab', 0),
#        ('pos_Kkma', 0),
#        ('pos_unkdata_Kkma', 560),


#        ('word', 0),
#        ('char', 20000),
#    ('jaso_unkdata', 0),
#        ('char', 40000),
#        ('char', 50000),
#        ('word', 1),
#    ('word', 1),
#    ('word', 3),
#    ('word', 5),
#    ('char_unkdata', 0),
#('char', 20000),
#        ('char', 30000),
#        ('char', 40000),
#        ('char', 50000),
#        ('char', 60000),
#    ('char', 0), 
    ]


def make_file(data_type, count_over_N):
    corpus = Corpus(data_type, limit_count = count_over_N, use_preprocessed_data = False)

    save_folder = './data/index/'
    os.makedirs(save_folder, exist_ok = True)
    length = len(corpus.train_data)

    print('train 1/2')
    text = corpus.preprocess_data(corpus.train_data[:length//2])
    text = [" ".join(list(map(str, line)))+'\n' for line in text]

    writefile = open(save_folder+'kowiki_'+data_type+"_"+str(count_over_N)+"_train_preprocessed.txt", 'w')
    writefile.writelines(text)
    writefile.close()

    print('train 2/2')
    text = corpus.preprocess_data(corpus.train_data[length//2:])
    text = [" ".join(list(map(str, line)))+'\n' for line in text]

    writefile = open(save_folder+'kowiki_'+data_type+"_"+str(count_over_N)+"_train_preprocessed.txt", 'a')
    writefile.writelines(text)
    writefile.close()

    print('valid')
    text = corpus.preprocess_data(corpus.valid_data)
    text = [" ".join(list(map(str, line)))+'\n' for line in text]

    writefile = open(save_folder+'kowiki_'+data_type+"_"+str(count_over_N)+"_valid_preprocessed.txt", 'w')
    writefile.writelines(text)
    writefile.close()

    print('test')
    text = corpus.preprocess_data(corpus.test_data)
    text = [" ".join(list(map(str, line)))+'\n' for line in text]

    writefile = open(save_folder+'kowiki_'+data_type+"_"+str(count_over_N)+"_test_preprocessed.txt", 'w')
    writefile.writelines(text)
    writefile.close()

if __name__ == "__main__":
    for data_type, count_over_N in param_set:
        print(data_type, count_over_N)
        make_file(data_type, count_over_N)

