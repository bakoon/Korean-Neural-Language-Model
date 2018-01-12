import json
import bz2
import os
import numpy as np
from codes import data_process
import math

_dirname = 'data/namu_dump'

_line_change_chars = ['.',]
_not_count_line_change_chars = ['!', '?']
_special_chars = [' ', ',',]


def _load_json():
    _json_location = 'data/namuwiki_20170327.json'
    print("loading json file")
    json_file = open(_json_location, 'r', encoding='utf-8').read()
    print("file loaded")
    data = json.loads(json_file)
    text = []
    for i in range(len(data)):
        print ("collecting {0} / {1}".format(i+1, len(data)), end='\r')
        text.append(data[i]['text'])
    return text

def _save_json_to_npy():
    num_per_save = 1000
    json_file = open(_json_location, 'r', encoding='utf-8').read()
    data = json.loads(json_file)

    text = []
    for i in range(len(data)):
        print ("save ", i)
        text.append(data[i]['text'])
        if i % num_per_save == num_per_save - 1:
            np.save('data/namu_dump/namu_dump_' + str(int(i / num_per_save)) + ".npy", text)
            text = []


def _readFile(name = None):
    if name == None:
        filenames = os.listdir(_dirname)
        full_filenames = []
        for filename in filenames:
            full_filenames.append(os.path.join(_dirname, filename))
        num = np.random.rand()*len(full_filename)
        text = np.load(full_filenames[int(num)])
    else:
        text = np.load(_dirname+name+'.npy')
    return text

def readFiles(num = None):
    text = []
    filenames = os.listdir(_dirname)
    full_filenames = []
    for filename in filenames:
        full_filenames.append(os.path.join(_dirname, filename))
    if num == None:
        n_read = len(full_filenames)
    else:
        n_read = min( num, len(full_filenames))
    for i in range(n_read):
        print("reading files {0} / {1}".format(i+1, n_read), end='\r')
        new = np.random.rand()*len(full_filenames)
        text = np.concatenate((text, np.load(full_filenames[int(new)])), axis=0)
    print ("")

    return text




def collect_jaso_hangul(text, remove_newline_char = True, throw_not_kor_line = True):
    new_text = []
    if remove_newline_char:
        text = [line[:-1] for line in text]
    text = " ".join(text) + " "
    pass_until_next_line = False
    need_check_end_of_line = False
    line = []
    text = text.split()
    log_per = math.ceil(len(text) / 10)
    for i in range(len(text)*2):
        if i % log_per == 0:
            print("{}/{}".format(i+1, len(text)*2))
        if i%2 == 0:
            if text[int(i/2)] == '<unknown>':
                line.append('U')
                word = None
            else:
                word = text[int(i/2)]
        else:
            word = ' '
        if word is not None:
            for j in range(len(word)):
                if pass_until_next_line:
                    if word[j] in _line_change_chars + _not_count_line_change_chars:
                        line = []
                        pass_until_next_line = False
                        need_check_end_of_line = False
                elif line == [] and (word[j] in _line_change_chars or word[j] == ' '):
                    None
                else :
                    new_line, new_char = _check_jaso_hangul(word[j])
                    if throw_not_kor_line and (new_char == []):
                        pass_until_next_line = True
                    elif new_line:
                        need_check_end_of_line = True
                        line += new_char
                    elif need_check_end_of_line:
                        need_check_end_of_line = False
                        if new_char == [' ']:
                            new_text.append("".join(line))
                            line = []
                        else:
                            line += new_char
                    else:
                        line += new_char
    return new_text


def collect_char_hangul(text, remove_newline_char = True, throw_not_kor_line = True):
    new_text = []
    if remove_newline_char:
        text = [line[:-1] for line in text]
    text = " ".join(text) + " "
    pass_until_next_line = False
    need_check_end_of_line = False
    line = []
    log_per = math.ceil(len(text) / 10)
    text = text.split()
    for i in range(len(text)*2):
        if i % log_per == 0:
            print("{}/{}".format(i+1, len(text)*2))
        if i%2 == 0:
            if text[int(i/2)] == '<unknown>':
                line.append('U')
                word = None
            else:
                word = text[int(i/2)]
        else:
            word = ' '
        if word is not None:
            for j in range(len(word)):
                if pass_until_next_line:
                    if word[j] in _line_change_chars + _not_count_line_change_chars:
                        line = []
                        pass_until_next_line = False
                        need_check_end_of_line = False
                elif line == [] and (word[j] in _line_change_chars or word[j] == ' '):
                    None
                else :
                    new_line, new_char = _check_char_hangul(word[j])
                    if throw_not_kor_line and (new_char == ''):
                        pass_until_next_line = True
                    elif new_line:
                        need_check_end_of_line = True
                        line += new_char
                    elif need_check_end_of_line:
                        need_check_end_of_line = False
                        if new_char == ' ':
                            new_text.append("".join(line))
                            line = []
                        else:
                            line += new_char
                    else:
                        line += new_char
    return new_text


"""
def collect_char_hangul(text):
    new_text = []
    for i in range(len(text)):
        page = []
        line = []
        for j in range(len(text[i])):
            print ("{0} / {1}    {2} / {3}          ".format(i+1, len(text), j+1, len(text[i])), end='\r')
            if text[i][j] == '\n':
                None
            elif line == [] and (text[i][j] in _line_change_chars or text[i][j] == ' '):
                None
            else :
                new_line, new_char = _check_char_hangul(text[i][j])
                line += new_char
                if new_line:
                    page.append("".join(line))
                    line = []
        if page != []:
            new_text.append(page)
            page = []
    return new_text
"""

def collect_word_hangul(text, throw_not_kor_line = True, remove_newline_char = True,  use_print = True):
    new_text = []
    if remove_newline_char:
        text = [line[:-1] for line in text]
    text = " ".join(text) + " "
    pass_until_next_line = False
    need_check_end_of_line = False
    line = []
    word = []
    log_per = math.ceil(len(text) / 10)
    for j in range(len(text)):
        if use_print and j % log_per == 0:
            print ("{0} / {1}".format(j+1, len(text), ), end='\r')
        if pass_until_next_line:
            if text[j] in _line_change_chars + _not_count_line_change_chars:
                line = []
                word = []
                pass_until_next_line = False
                need_check_end_of_line = False
        elif word == [] and line == [] and text[j] in _line_change_chars:
            None
        else :
            new_line, new_word, new_char = _check_word_hangul(text[j])
            if throw_not_kor_line and (new_char == ''):
                pass_until_next_line = True
            elif new_line:
                if line == [] and word == []:
                    None
                else:
                    need_check_end_of_line = True
                    line.append("".join(word))
                    word = []
                    word.append(new_char)

            elif need_check_end_of_line:
                need_check_end_of_line = False
                if new_char == " ":
                    line.append("".join(word))
                    word = []
                    new_text.append(line)
                    line = []
                else:
                    word.append(new_char)

            elif new_word:
                if line == [] and word == []:
                    None
                else:
                    if not word == []:
                        line.append("".join(word))
                    if new_char == ' ':
                        line.append('<space>')
                    else:
                        line.append(new_char)
                    word = []
            else:
                word.append(new_char)
    return new_text

def _check_jaso_hangul(text):
    new_text = []
    text = data_process.text_to_jaso(text)
    for i in range(len(text)):
        if (12593 <= ord(text[i]) <= 12643) or (48 <= ord(text[i]) <= 57)  or (text[i] in _line_change_chars) or (text[i] in _special_chars):
            new_text.append(text[i])
        else:
            None
    if len(text) != 0 and text[-1] in _line_change_chars:
        new_line = True
    else:
        new_line = False
    return new_line, new_text



def _check_word_hangul(text):
    if 44032 <= ord(text) <= 55203 or 48 <= ord(text) <= 57:
        new_text = text
        new_line = False
        new_word = False
    elif text in _special_chars:
        new_text = text
        new_line = False
        new_word = True
    elif text in _line_change_chars:
        new_text = text
        new_line = True
        new_word = False
    else:
        new_text = ''
        new_line = False
        new_word = False
    return new_line, new_word, new_text


def _check_char_hangul(text):
    if 44032 <= ord(text) <= 55203 or 48 <= ord(text) <= 57 or text in _special_chars:
        new_text = text
        new_line = False
    elif text in _line_change_chars:
        new_text = text
        new_line = True
    else:
        new_text = ''
        new_line = False
    return new_line, new_text







