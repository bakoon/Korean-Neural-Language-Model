class SETTING(object):
    def __init__(self):

        model_type = ['LSTM', 'RNN_TANH', 'RNN_RELU', 'GRU']
        self.model_type = "LSTM"
        
        optimizer = ['SGD', 'Adam']
        self.optimizer = 'SGD'

        self.hidden_size = 650#1500
        self.num_layers = 1
        
        state = ["new_model", "load_model", "test"]
        self.state = state[1]
        self.num_load_epoch = None

        #data_type = ['jaso', 'char', 'word', 'test', 'penn'] and more
        self.data_type = 'pos_Mecab' 
        self.use_N_vocab = 0
        self.use_vocab_count_over_N = 0#320#240#60000#1#100 

        self.num_data_per_minibatch = 20000#0000000
        self.batch_size = 20#256#512

        ### average sequence length
        # word      : 12 (not include space)
        # char      : 53 (include space)
        # jaso      : 107 (include space)
        # penn      : 21 (not include space)

        ## pos-tagger
        # Twitter   : 26 (not include space)
        # Mecab     : 28 (not
        # Komoran   : 30 (not
        # Kkma      : 31 (not
        # Hannanum  : 28 (not

        ## byte pair encoding
        # bpe100    : 73 (include space)
        # bpe1000   : 50 (inblude space)
        # bpe2000   : 46 (include space)
        # bpe3000   : 44 (include space)

        # jaso_unkdata : 97
        # char_unkdata : 51
        # bpe_unk_1000 : 45
        # bpe_unk 5000 : 38
        # bpe_unk 10000 : 36

        # pos_twitter_unkdata : 24
        # pos_hannanum_unkdata : 26
        # pos_Mecab_unkdata : 26
        # pos_Momoran_unkdata : 27

        self.sequence_length = {
            'jaso' : 128,
            'char' : 64,
            'word' : 24,
            'test' : 64, 
            'penn' : 35,
            'pos_Twitter'   : 48, 
            'pos_Hannanum'  : 52,
            'pos_Mecab'     : 52,
            'pos_Kkma'      : 0,
            'pos_Komoran'   : 54,
            'bpe_100'       : 96,
            'bpe_1000'      : 64,
            'bpe_5000'      : 52,
            'bpe_10000'     : 48,
            'jaso_unkdata'  : 128,
            'char_unkdata'  : 80,
            }[self.data_type]
        
        self.initial_learning_rate = 20
        self.lr_decay_rate = 0.25
        self.dropout_rate_tobe_zeroed = 0.65
        self.grad_clipping = 0.25

        self.max_patience = 2 
        self.max_lr_change = 5#10
        self.max_epoch = 1000000

        self.log_per_N_batch = 10
        if self.data_type == 'test' or self.data_type == 'penn':
            self.log_per_N_batch = 0
        """
        {
            'jaso' : 10,
            'char' : 10,
            'word' : 10,
            'pos_Twitter' : 10,
            'pos_Mecab' : 10,
            'pos_Hannanum' : 10,
            'test' : 1,
            'penn' : 1,
        }[self.data_type]
        """


if __name__ == "__main__":
    import sys
    import train
    import generate

    setting = SETTING()

    if len(sys.argv) == 1:
        print("***** train or generate *****")
    elif sys.argv[1] == 'train':
        train.main(setting)
    elif sys.argv[1] == 'generate':
        generate.main(setting)
    else:        
        print("***** train or generate *****")

