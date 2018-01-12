###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

from codes import data, data_process
from codes import model as m
from codes.byte_pair_encoding import BPE


def main(setting):
     
    data_type = setting.data_type
    n_vocab = setting.use_N_vocab
    n_limit_count = setting.use_vocab_count_over_N
    model_type = setting.model_type
    num_layers = setting.num_layers
    hidden_size = setting.hidden_size

    save_name = './model/{}_{}_{}_{}_{}_{}'.format(data_type, n_vocab, n_limit_count, model_type, num_layers, hidden_size)


    save_name = save_name + '/' + save_name.split('/')[-1]
    checkpoint = torch.load(save_name+'.pt')
    model_info = checkpoint['model_info']
    model_state_dict = checkpoint['state_dict']
    model_type, n_tokens, hidden_size, num_layers, dropout = model_info

    corpus = data.Corpus(data_type, limit_num_words = n_vocab, limit_count = n_limit_count, dict_only = True)
    ntokens = len(corpus.dictionary)

    model = m.RNNModel(model_type, n_tokens, hidden_size, num_layers, dropout)
    model.load_state_dict(model_state_dict)



    start_token = '<start>'
    n_sentences = 30
    temperature = 1#00000
    cuda = True

    model.eval()

    if cuda:
        model.cuda()
    else:
        model.cpu()

    hidden = model.init_hidden(1)

    if start_token is None:
        input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    else:
        start_idx = corpus.dictionary.translate_data_to_idx([start_token])[0]
        input = Variable(torch.LongTensor(start_idx).resize_(1,1), volatile=True)

    if cuda:
        input.data = input.data.cuda()

    if data_type[:3] == 'bpe':
        bpe = BPE()
        bpe.load_dict('./data/dictionary/'+data_type+'_dict.pkl')

    word = [start_token]
    count = 0
    while(True):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        new_word = corpus.dictionary._num_to_word[word_idx]
        word.append(new_word)
        if new_word == '<end>':
            if data_type[:4] == 'jaso':
                print_text = data_process.jaso_to_text(word[1:-1])
            elif data_type[:4] == 'word' or data_type[:4] == 'penn' or data_type[:3] == 'pos':
                print_text = " ".join(word)
            elif data_type[:3] == 'bpe':
                print_text = bpe.translate_bpe_to_text(word[1:-1])
                print_text = data_process.jaso_to_text("".join(print_text))
            else:
                print_text = "".join(word)
            print(print_text)
            word = []
            count += 1
        if count == n_sentences:
            break

