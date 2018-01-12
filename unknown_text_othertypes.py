
import os
from codes import data, rawdata_process

save_folder = './data/text/'
os.makedirs(save_folder, exist_ok = True)

#file_list = ['valid']

file_list = ['train', 'test']#, 'valid']
type_list = ['jaso', 'char']

for filename in file_list:
    print("test")
    with open(save_folder + 'kowiki_word_unkdata_'+filename+'.txt', 'r') as f:
        text = f.readlines()
    
    for typename in type_list:
        if typename == 'jaso':
            newtext = rawdata_process.collect_jaso_hangul(text)
        elif typename == 'char':
            newtext = rawdata_process.collect_char_hangul(text)
        with open(save_folder + 'kowiki_'+typename+'_unkdata_'+filename+'.txt', 'w') as f:
            print(save_folder + 'kowiki_'+typename+'_unkdata_'+filename+'.txt')
            for line in newtext:
                f.write("".join(line) + '\n')


