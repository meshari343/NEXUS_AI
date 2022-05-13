import torch.nn as nn   
    
class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu= False, drop_prob=None, bidirectional=False):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional
        self.train_on_gpu = train_on_gpu
        # define all layers
        self.embd = nn.Embedding(vocab_size, embedding_dim)
        
        if drop_prob:
            if not (bidirectional):
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional= True, dropout=drop_prob)  
        else:
            if not (bidirectional):
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional= True)            
          
        self.dropout = nn.Dropout(0.3)
        
        if not (bidirectional):
            self.fc1 = nn.Linear(hidden_dim, output_size)
        else:
            self.fc1 = nn.Linear(hidden_dim * 2, output_size)
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        x = x.long()
        embeds = self.embd(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out[:, -1]
        
        output = self.dropout(lstm_out)
        output = self.fc1(lstm_out)
        sig_out= self.sigmoid(output)
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if not (self.bidirectional):
            if (self.train_on_gpu): 
                hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())            
        else:
            if (self.train_on_gpu): 
                hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().cuda(),
                            weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(),
                            weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())

        
        return hidden
  




    