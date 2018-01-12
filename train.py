import time
from time import localtime, strftime
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

from codes import data
from codes import model as Model

#import setting as s


def main(setting = None):

    ###############################################################################
    # Setting 
    ###############################################################################
    #setting = s.SETTING()
    
    state = setting.state
    load_epoch_num = setting.num_load_epoch
    model_type = setting.model_type
    
    optim = setting.optimizer
    opt = getattr(torch.optim, optim)
    
    hidden_size = setting.hidden_size
    num_layers = setting.num_layers
    
    data_type = setting.data_type
    n_limit_count = setting.use_vocab_count_over_N
    n_vocab = setting.use_N_vocab
    
    n_read_data = setting.num_data_per_minibatch
    n_valid_read_data = n_read_data
    batch_size = setting.batch_size   
    sequence_length = setting.sequence_length
    train_sequence_length = sequence_length
    eval_sequence_length = sequence_length
    
    lr = setting.initial_learning_rate
    lr_decay_rate = setting.lr_decay_rate
    dropout = setting.dropout_rate_tobe_zeroed
    grad_clipping = setting.grad_clipping

    max_patience = setting.max_patience 
    max_lr_change = setting.max_lr_change 
    num_epochs = setting.max_epoch

    save_name = './model/{}_{}_{}_{}_{}_{}'.format(data_type, n_vocab, n_limit_count, model_type, num_layers, hidden_size)
    os.makedirs(save_name, exist_ok = True)
    txt_save_name = save_name
    save_name = save_name + '/' + save_name.split('/')[-1]

    log_per = setting.log_per_N_batch


    try:
        layerNorm = setting.layerNorm
    except:
        layerNorm = False

    try:
        make_unknown_false = setting.make_unknown_false
    except:
        make_unknown_false = False

    ###############################################################################
    # Load data
    ###############################################################################

    best_val_loss = float('inf')
    patience = 0
    start_epoch = 1
    lr_change = 0
    
    if state == 'load_model':
        print("load model")
        if load_epoch_num is not None:
            load_name = save_name + '_' + str(load_epoch_num)
        else:
            load_name = save_name
        start_epoch, model_state_dict, opt_state_dict, lr, best_val_loss, model_info = restore_model(load_name+'.pt')
        model_type, n_tokens, hidden_size, num_layers, dropout = model_info

    corpus = data.Corpus(data_type, limit_num_words = n_vocab, limit_count = n_limit_count)
    n_tokens = len(corpus.dictionary)
    unknown_token = corpus.dictionary.word_to_num("<unknown>") 
    data_size = corpus.get_data_size()
    train_data_size = data_size['train']
    valid_data_size = data_size['valid']
    test_data_size = data_size['test']
    print('{} train data, {} valid data, {} test data'.format(train_data_size, valid_data_size, test_data_size))

    ###############################################################################
    # Build the model
    ###############################################################################

    if make_unknown_false:
        ignore_index = corpus.dictionary.word_to_num('<unknown>')
    else:
        ignore_index = corpus.dictionary.word_to_num('<pad>')

    criterion = nn.CrossEntropyLoss(ignore_index = ignore_index)

    model = Model.RNNModel(model_type, n_tokens, hidden_size, num_layers, dropout)
    optimizer = opt(model.parameters(), lr = lr)

    if torch.cuda.is_available():
        model.cuda()

    if state == 'load_model':
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(opt_state_dict) 
    

    try:
        log_file = open(txt_save_name+'.txt', 'r+')
        log_file.readlines()
    except:
        log_file = open(txt_save_name+'.txt', 'w')

    log_file.write("\nvocab size : {}\n".format(len(corpus.dictionary)))
    print("vocab size : {}".format(len(corpus.dictionary)))
    log_file.close()


    ###############################################################################
    # Train the model
    ###############################################################################

    for epoch in range(start_epoch, num_epochs+1):
        
        log_file = open(txt_save_name+'.txt', 'r+')
        log_file.readlines()
        
        print(state+ '\n' + '-' * 40)
        log_file.write("{} epoch {} ".format(strftime("%m%d%H%M", localtime()), epoch))
        if state == 'test':
            _, model_state_dict, opt_state_dict, lr, best_val_loss, _ = restore_model(save_name+'.pt')
            model.load_state_dict(model_state_dict)
            break
        if state == 'end_train':
            _, model_state_dict, opt_state_dict, lr, best_val_loss, _ = restore_model(save_name+'.pt')
            model.load_state_dict(model_state_dict)

            break
        elif state == 'save_model':
            check_point = {
                    'epoch' : epoch, 
                    'best_valid_loss' : best_val_loss, 
                    'learning_rate' : lr,
                    'state_dict' : model.state_dict(),
                    'opt_state_dict' : optimizer.state_dict(),
                    'model_info' : (model_type, n_tokens, hidden_size, num_layers, dropout),        
                    'lr_change' : lr_change,
                    'setting' : setting,
            }
            torch.save(check_point, save_name+'.pt')
#torch.save(check_point, save_name+'_'+str(epoch)+'.pt')
        elif state == 'change_learning_rate':
            _, model_state_dict, opt_state_dict, _, best_val_loss, _ = restore_model(save_name+'.pt')
            model.load_state_dict(model_state_dict)
            lr *= lr_decay_rate
            optimizer = opt(model.parameters(), lr = lr)
            """
            new_opt = optimizer.state_dict()
            new_opt['param_groups'][0]['lr'] = lr
            optimizer.load_state_dict(new_opt)
            """
        print('\n\n****Epoch', epoch, data_type, n_limit_count)
        if data_type != 'penn':
            print("shuffling data")
            corpus.shuffle_train_data()

        n_read_data = min(n_read_data, train_data_size)
        n_batch = math.ceil(train_data_size / n_read_data)
        epoch_start_time = time.time()
        print("training")
        train_loss = 0
        start_time = time.time()
        for minibatch in range(n_batch):
            batch_time = time.time()
            train_data = corpus.get_train_data(n_read_data)
            batch_loss = train(model, train_data, criterion, optimizer, grad_clipping, lr, batch_size, train_sequence_length, minibatch, n_batch)
            train_loss += batch_loss
            if (minibatch+1) % log_per == 0:
                end_time = time.time()
                train_time = end_time - start_time 
                start_time = end_time
                print(".......  {:3}/{:3}  time {:2.1f}m   loss : {:2.3f}    ppl : {:2.3f}   ".format(minibatch+1, n_batch, train_time/60, train_loss/log_per, math.exp(train_loss/log_per)))
                train_loss = 0

        print()    
        n_valid_read_data = min(n_valid_read_data, valid_data_size)
        n_batch = math.ceil(valid_data_size / n_valid_read_data)
        val_loss = 0
        print("validating") 
        for minibatch in range(n_batch): 
            valid_data = corpus.get_valid_data(n_valid_read_data)
            batch_loss = evaluate(model, valid_data, criterion, batch_size, eval_sequence_length)
            val_loss += batch_loss
            print("    {:3}/{:3}   loss : {:2.3f}    ppl : {:2.3f}".format(minibatch+1, n_batch, val_loss/(minibatch+1), math.exp(val_loss/(minibatch+1))), end='\r')
        epoch_time = time.time() - epoch_start_time
        val_loss /= n_batch
        print()
        print('epoch time: {:2.1f}h    valid loss {:2.3f}    valid ppl {:2.3f}    best ppl {:.3f}   '.format(epoch_time/3600, val_loss, math.exp(val_loss), math.exp(best_val_loss)))
        

        if val_loss < best_val_loss:
            state = 'save_model'
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience < max_patience:
                state = "keep_train"
            else :
                patience = 0
                lr_change += 1
                if lr_change == max_lr_change:
                    state = 'end_train'
                else:
                    state = 'change_learning_rate'
        log_file.write("loss: {:.3f} ppl: {:.3f} time: {:.3f}s {}\n".format(val_loss, math.exp(val_loss), epoch_time, state))
        print('lr :', lr, '\tlr_change :', lr_change, '\tpatience :', patience,)
        
    # Run on test data.
    n_test_read_data = min(n_valid_read_data, test_data_size)
    n_batch = math.ceil(test_data_size / n_test_read_data)
    test_loss = 0
    print("testing")
    for minibatch in range(n_batch): 
        test_data = corpus.get_test_data(n_test_read_data)
        batch_loss = evaluate(model, test_data, criterion, batch_size, eval_sequence_length)
        test_loss += batch_loss
        print("    {:3}/{:3}   loss : {:2.3f}    ppl : {:2.3f}".format(minibatch+1, n_batch, test_loss/(minibatch+1), math.exp(test_loss/(minibatch+1))), end='\r')
    test_loss /= n_batch
    print(save_name)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    log_file.write("loss: {:.3f} ppl: {:.3f} ".format(test_loss, math.exp(test_loss)))



###############################################################################
# Training code
###############################################################################


def restore_model(save_name):
    checkpoint = torch.load(save_name)
    start_epoch = checkpoint['epoch']
    model_state_dict = checkpoint['state_dict']
    try:
        optimizer_state_dict = checkpoint['opt_state_Dict']
    except:
        optimizer_state_dict = checkpoint['opt_state_dict']

    learning_rate = checkpoint['learning_rate']
    loss = checkpoint['best_valid_loss']
    model_info = checkpoint['model_info']
    return start_epoch, model_state_dict, optimizer_state_dict, learning_rate, loss, model_info

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)



def evaluate(model, evaluate, criterion, batch_size, eval_sequence_length):
    model.eval()
    evaluate = make_batch(evaluate, batch_size)
    hidden = model.init_hidden(evaluate.size(1))
    total_loss = 0
    for batch, start in enumerate(range(0, evaluate.size(0)-1, eval_sequence_length)):
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        evaluate_data, evaluate_targets = get_batch(evaluate, start, eval_sequence_length, evaluate = True)
        output, hidden = model(evaluate_data, hidden)
        output = output.view(-1, output.size(-1))
        evaluate_targets = evaluate_targets.contiguous().view(-1)
        
        loss = criterion(output, evaluate_targets)
        total_loss += loss.data[0]
    return total_loss / len(range(0, evaluate.size(0)-1, eval_sequence_length))
    

def train(model, train, criterion, optimizer, grad_clipping, lr, batch_size, train_sequence_length, minibatch, n_minibatch):
    model.train()
    train = make_batch(train, batch_size)
    hidden = model.init_hidden(batch_size)
    total_loss = 0
    batch_range = range(0, train.size(0)-1, train_sequence_length)
    n_batch = len(batch_range)
    for batch, start in enumerate(batch_range):
        batch_time = time.time()
        train_data, train_targets = get_batch(train, start, train_sequence_length, evaluate = False)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(train_data, hidden)
        output = output.view(-1, output.size(-1))
        train_targets = train_targets.contiguous().view(-1)
        
        loss = criterion(output, train_targets)
        loss.backward()
        total_loss += loss.data[0]# * len(train_data)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), grad_clipping)
        optimizer.step()
        print("{:3}/{:3}  {:3}/{:3}  time {:3.1f}ms   loss : {:.3f}    ppl : {:.3f}".format(batch+1, n_batch, minibatch+1, n_minibatch, 1000*(time.time()-batch_time), loss.data[0], math.exp(loss.data[0])), end='\r')

    return total_loss / n_batch

def make_batch(input, batch_size):
    data = np.concatenate(input, axis = 0).astype(int)
    sequence_length = len(data) // batch_size
    data = data[:batch_size*sequence_length]
    data = data.reshape([batch_size, sequence_length])
    data = torch.LongTensor(data).t()
    if torch.cuda.is_available():
        data = data.cuda()
    return data


def get_batch(input, start, sequence_length, evaluate):
    sequence_length = min(sequence_length, len(input)-start-1)
    data = input[start:start+sequence_length]
    targets = input[start+1:start+1+sequence_length]
    data = Variable(data, volatile = evaluate)
    targets = Variable(targets, volatile = evaluate).view(-1)

    return data, targets


if __name__ == "__main__":
    main()
