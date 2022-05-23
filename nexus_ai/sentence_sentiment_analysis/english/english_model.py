import logging
import torch
from nexus_ai.sentence_sentiment_analysis.preprocessing import tokenize_data, pad_features
from nexus_ai.sentence_sentiment_analysis import model
from torch.utils.data import DataLoader  



def pred(reviews, gpu=False, seq_length=20, batch_size=1000,
 model_path='nexus_ai/sentence_sentiment_analysis/english/models/yelp_test20.pth'):
    ''' 
    
    predict the sentiment of a list of reviews.

    params:
    reviews : a dataframe containg reviews 
    gpu : whethever the model would use the gpu or not
    seq_length : the maximum length of a review
    batch_size : how many of the data are going to be predicted at a time
    model_path : the location of the saved model to be loaded
    
    returns:
    a list of the cleaned and preprocessed reviews, and a list of each review prediction


    '''

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if model_path:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        vocab_to_int = checkpoint['vocab_to_int']
        
        vocab_size = len(vocab_to_int)+1
        
        if checkpoint['seq_length']: 
            seq_length = checkpoint['seq_length']
        else:
            seq_length = 40
 
        net = model.SentimentRNN(
            vocab_size = vocab_size, output_size = checkpoint['output_size'],
            embedding_dim = checkpoint['embedding_dim'], hidden_dim = checkpoint['hidden_dim'],
            n_layers = checkpoint['n_layers'], train_on_gpu = gpu,
            drop_prob = 0, bidirectional = checkpoint['bidirectional']
            )

        net.load_state_dict(checkpoint['state_dict'])
        
    if(gpu):
        net.cuda()
        
    net.eval()  

    # tokenize reviews
    reviews_ints = tokenize_data(reviews, vocab_to_int)  
        
    # pad tokenized sequence
    features = pad_features(reviews_ints, seq_length)
    if len(features) == 0:
        raise Exception('the reviews list must contain at least one review!')
    # convert to tensor to pass into model
    feature_tensor = torch.from_numpy(features)

    # if(len(features)) < batch_size:
    #     for i in range(1000):
    #         if(len(features)) < batch_size:
    #             batch_size = int(batch_size*10)
    #         else:
    #             break
    #     pred_loader = DataLoader(feature_tensor, shuffle=False, batch_size=batch_size)
    # else:

    pred_loader = DataLoader(feature_tensor, shuffle=False, batch_size=batch_size)
    
    pred_list = []
    
    for i, data in enumerate(pred_loader):
        # initialize hidden state
        h = net.init_hidden(data.shape[0])

        if(gpu):
            data = data.cuda()

        # get the output from the model        
        output, h = net(data, h)

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze()) 
        pred = pred.cpu()
        pred = pred.data.detach().numpy().tolist()
        
        if isinstance(pred, float):
            pred_list.append(pred)
        elif isinstance(pred, list):
            for i in pred:
                pred_list.append(i)
        else:
            logging.warning('irregular output in sentince sentiment analysis')

    pred_list = ['Positive' if prediction == 1 else('Negative' if prediction == 0 else None) for prediction in pred_list]
    # test logging 
    # pred_list.remove('positive')
    # pred_list.append(None)
    if None in pred_list:     
        logging.warning('irregular output in english sentince sentiment analysis')

    return pred_list  