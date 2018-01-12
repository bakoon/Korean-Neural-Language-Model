import os
import numpy as np

import unicodedata


def read_data(encoding = 'CP949', output_type = 'word'):
    try:
        print ("loading data")
        text = np.load('data/data_'+output_type+'.npy')
        print ("data loaded")
    except:
        text = _read_all_file(encoding, output_type)
    return text

def _read_all_file(encoding = 'CP949', output_type = 'word'):

    directory1 = 'data/high_quality/'
#    directory2 = 'data/Corpus3/'
    
    text = []
    filenames = os.listdir(directory1)
    for filename in filenames:
        full_filename = os.path.join(directory1, filename)
        extension = os.path.splitext(full_filename)[-1]
        if extension == '.vrt':
            text.append(_readfile(full_filename, encoding, output_type))
    return np.array(text)

def _readfile(filename, encoding = 'CP949', output_type = 'word'):
    f = open(filename, encoding = encoding)
    print(filename)
    text = []
    temp = []
    while True:
        line = f.readline()
        #print(line)
        if not line :
            break
        word = line.split('\t')[0]
        #word = text_to_ascii(word)
        if (len(word) == 0) or (word == '*') or (word[0] == '<') or (word == '\n'):
            None
        elif word[0] == '.':
            try :
                temp[-1] = word[0]
                #temp[-2] = temp[-2] + '.'
                #temp.pop()
                text.append(np.array(temp))
                temp = []
            except :
                None
        else :
            temp.append(word)
            temp.append(' ')
    return np.array(text)


def save(encoding = 'CP949', output_type = 'word'):
    text = _read_all_file(encoding, output_type)
    np.save('data/data_'+output_type+'.npy', text)






def _conv_compatibility_jamo(ch):
    unicode_names = unicodedata.name(ch)
    # print ch, unicode_names
    if unicode_names.find('CHOSEONG') >= 0:
        unicode_names = unicode_names.replace('CHOSEONG', 'LETTER')
    elif unicode_names.find('JUNGSEONG') >= 0:
        unicode_names = unicode_names.replace('JUNGSEONG', 'LETTER')
    elif unicode_names.find('JONGSEONG') >= 0:
        unicode_names = unicode_names.replace('JONGSEONG', 'LETTER')
    return unicodedata.lookup(unicode_names)


def _char_to_jaso(ch):
    if unicodedata.name(ch)[7:13] == 'LETTER' :
        return ch
    else:
        jaso = []
        ch = int( ord(ch) - 0xAC00 )
        jong = int( ch % 28 )
        jung = int( ((ch - jong) / 28) % 21 )
        cho = int( (((ch - jong) / 28) - jung) / 21 )
        if cho >= 0:
            jaso.append(_conv_compatibility_jamo(chr(cho + 0x1100)))
        if jung >= 0:
            jaso.append(_conv_compatibility_jamo(chr(jung + 0x1161)))
        if jong > 0:
            jaso.append(_conv_compatibility_jamo(chr(jong + 0x11A7)))
        return ''.join(jaso)

def text_to_jaso(text):
    jaso = []
    for i in range(len(text)):
        try:
            if unicodedata.name(text[i])[0:6] == 'HANGUL' :
                jaso.append(_char_to_jaso(text[i]))
            else:
                jaso.append(text[i])
        except:
            None
    return ''.join(jaso)


def jaso_to_text(ch):
    text = ''
    temp = ''
    for i in range(len(ch)):
        if not unicodedata.name(ch[i])[0:6] == 'HANGUL':
            text = text + _jaso_to_char(temp) + ch[i]
            temp = ''
        elif len(temp) == 3:
            if ord(ch[i]) < 12623 : ## ch[i] : 자음
                text = text + _jaso_to_char(temp)
                temp = ch[i]
            else : ## ch[i] : 모음
                text = text + _jaso_to_char(temp[0:2])
                temp = temp[2] + ch[i]
        else:
            temp = temp + ch[i]
    text = text + _jaso_to_char(temp)
    return text

def _jaso_to_char(ch):
    if len(ch) == 3:
        cho = _chosung_num(ch[0])
        jung = ord(ch[1]) - 12623
        jong = _jongsung_num(ch[2])
        char_code = 0xac00
        char_code += 28 * 21 * cho
        char_code += 28 * jung
        char_code += jong + 1
        return chr(char_code)
    elif len(ch) == 2:
        cho = _chosung_num(ch[0])#ord(ch[0]) - 12593
        jung = ord(ch[1]) - 12623
        char_code = 0xac00
        char_code += 28 * 21 * cho
        char_code += 28 * jung
        return chr(char_code)
    else :
        return ''

def _chosung_num(ch):
    x = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
            'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    return x.index(ch)

def _jongsung_num(ch):
    x = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ',
            'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    return x.index(ch)



def jaso_to_code(text):
    letters = []
    for i in range(len(text)):
        n_ascii = ord(text[i])
        ##숫자, 영어, 특수문자 및 한글 ascii character
        if n_ascii < 128 :
            letters.append(n_ascii)# = letters + text[i]
        elif (n_ascii > 12592) and (n_ascii < 12644) :
            letters.append( n_ascii - 12593 + 128 )# = letters + text[i]
    return letters



def code_to_jaso(text):
    jaso = []
    for i in range(len(text)):
        if text[i] < 128:
            jaso.append(chr(text[i]))
        else:
            jaso.append(chr(text[i] - 128 + 12593))
    return ''.join(jaso)



def num_of_char():
    return 128 + 52




def text_to_code(text):
    return jaso_to_code(text_to_jaso(text))

def code_to_text(code):
    return jaso_to_text(code_to_jaso(code))





