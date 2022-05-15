import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nexus_ai.sentence_sentiment_analysis.preprocessing import tokenize_data, pad_features, clean_reviews_list, google_clean_reviews
from nexus_ai.sentence_sentiment_analysis.model import SentimentRNN
from torch.utils.data import TensorDataset, DataLoader


def pred(reviews, gpu=False, seq_length=20, batch_size=1000,
 model_path='nexus_ai/sentence_sentiment_analysis/models/yelp_test20.pth'):
    ''' 
    
    predict the sentiment of a list of reviews.

    params:
    reviews : a list containg reviews 
    gpu : whethever the model would use the gpu or not
    seq_length : the maximum length of a review
    batch_size : how many of the data are going to be predicted at a time
    model_path : the location of the saved model to be loaded
    
    returns:
    a list of the cleaned and preprocessed reviews, and a list of each review prediction


    '''

    if not reviews:
        raise Exception('please specify a reviews list')

    if not model_path:
        raise Exception('please specify a file path holding a net')

    if model_path:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        vocab_to_int = checkpoint['vocab_to_int']
        
        vocab_size = len(vocab_to_int)+1
        
        if checkpoint['seq_length']: 
            seq_length = checkpoint['seq_length']
        else:
            seq_length = 40
 
        net = SentimentRNN(vocab_size = vocab_size, output_size = checkpoint['output_size'],
                           embedding_dim = checkpoint['embedding_dim'], hidden_dim = checkpoint['hidden_dim'],
                           n_layers = checkpoint['n_layers'], train_on_gpu = gpu,
                           drop_prob = checkpoint['drop_prob'], bidirectional = checkpoint['bidirectional'])

        net.load_state_dict(checkpoint['state_dict'])
        
    if(gpu):
        net.cuda()
        
    net.eval()  

    # process the list of reviews
    deleted_idx, processed_reviews = clean_reviews_list(reviews)

    # tokenize review
    reviews_ints = tokenize_data(processed_reviews, vocab_to_int)  
        
    # pad tokenized sequence
    features = pad_features(reviews_ints, seq_length)
    
    # convert to tensor to pass into model
    feature_tensor = torch.from_numpy(features)
    # pred_data = TensorDataset(feature_tensor)
    if len(features) == 0:
        raise Exception('the reviews list must contain at least one review!')
    elif(len(features)) < batch_size:
        for i in range(1000):
            if(len(features)) < batch_size:
                batch_size = batch_size//10
            else:
                break
        pred_loader = DataLoader(feature_tensor, shuffle=False, batch_size=batch_size)
    else:
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
        pred = pred.detach().numpy().tolist()

        for i in pred:
            pred_list.append(i)

    pred_list = ['Positive' if prediction == 1 else('Negative' if prediction == 1 else None) for prediction in pred_list]
    # test logging 
    # pred_list.remove('positive')
    # pred_list.append(None)
    if None in pred_list:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.warning('irregular output in sentince sentiment analysis')

    return deleted_idx, pred_list    

# an old method used for testing
def predict(data_path=None, dataframe=None, reviews=None, labels=None, column=None, gpu=False, seq_length=20, batch_size=1000,
            vocab_to_int=None, net=None, model_path='/content/drive/MyDrive/sentiment_analysis/yelp_test20.pth',
            stats=False, use_labels=False):
    ''' 
    
    predict the sentiment of a list of reviews.

    params:
    df : a dataframe in which the reviews is the first column and the labels if exist the second column
    reviews : a list containg reviews if the dataframe is not provided
                or a string with one review to be anaylized
    labels : a list containg labels if the dataframe is not provided
    column: the columns names of they are not the defualt text, label
    gpu : whethever the model would work in the gpu or not
    seq_length : the padded length of a review
    batch_size : how many of the data are going to be predicted at a time
    vocab_to_int : the vocab used to train the model (only if the prediction is based on a provided net)
    net : A trained RNN to be used to predict 
    model_path : the location of the saved model to be loaded
    stats : whethever to print out the statics of the prediction or not
    
    '''

    print('note that this method is only for testing purposes')

    if reviews is None and dataframe is None and data_path is None:
        raise Exception('please specify either a 1) data_path 2) dataframe holding with the data 3) a review list or stirng')

    if (not net) and (not model_path):
        raise Exception('please specify either a net or a file path holding a net')

    if (net) and (not vocab_to_int) :
        raise Exception('if you are using a net please specify the vocab') 

    if not net:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        vocab_to_int = checkpoint['vocab_to_int']
        
        vocab_size = len(vocab_to_int)+1
        
        if checkpoint['seq_length']: 
            seq_length = checkpoint['seq_length']
        else:
            seq_length = 40
 
        net = SentimentRNN(vocab_size = vocab_size, output_size = checkpoint['output_size'],
                           embedding_dim = checkpoint['embedding_dim'], hidden_dim = checkpoint['hidden_dim'],
                           n_layers = checkpoint['n_layers'], train_on_gpu = gpu,
                           drop_prob = checkpoint['drop_prob'], bidirectional = checkpoint['bidirectional'])

        net.load_state_dict(checkpoint['state_dict'])
        
    if(gpu):
        net.cuda()
        
    net.eval()  

    # if the reviews (str) and only a review is given process that review and return the prediction
    if reviews is not None and dataframe is None and data_path is None:
        # proccess a string containing a review
        if isinstance(reviews, str):
            reviews = [reviews]
            labels = [0]
            reviews_ints,lebals = tokenize_data(reviews, labels, vocab_to_int, lebals_encoded=True)
            h = net.init_hidden(1)
            reviews_ints = torch.from_numpy(np.array(reviews_ints))
            
            if(gpu):
                net.cuda()
            if(gpu):
                reviews_ints = reviews_ints.cuda()
            output, h = net(reviews_ints, h)
            
            pred = torch.round(output.squeeze())
            pred = pred.cpu()
            pred = pred.detach().numpy().tolist()
            if pred == 1:
                return 'positive'
            elif pred == 0:
                return 'negative'

    # if the datapath is given and only reviws are given process the reviews in that file     
    if data_path is not None:
        dataframe = pd.read_json(data_path)
        original_col = dataframe.columns
        # if a str value for column is given process it as the text coulmn
        if column and isinstance(column, str):
            if not use_labels:
                text_original = 0
                for i in range(len(dataframe.columns)):
                    if i == column:
                        text_original = i
                # storing the column
                target = dataframe[column]
                # dropping it from the dataframe
                dataframe.drop(column, axis=1, inplace=True)
                # reinserting it at the start
                dataframe.insert(0, 'text', target)
            else:
                raise Exception('if you want to use labels please provide the column as a list with'
                                +"1) column name for texts"
                                +"2) column name for lables")
        # if a list value for column is given process it as the text coulmn, label columns
        elif column and isinstance(column, list):
            if use_labels:
                text_original = 0
                label_original = 0
                for i in range(len(dataframe.columns)):
                    if i == column[0]:
                        text_original = i
                    if i == column[1]:
                        label_original = i                    
                # storing the column
                target = dataframe[column[0]]
                # dropping it from the dataframe
                dataframe.drop(column[0], axis=1, inplace=True)
                # reinserting it at the start
                dataframe.insert(0, 'text', target)
                # storing the column
                target = dataframe[column[1]]
                # dropping it from the dataframe
                dataframe.drop(column[1], axis=1, inplace=True)
                # reinserting it at after the text column
                dataframe.insert(1, 'labels', target)
        # if there's no labels give a dummy value of 0 for the length of the reviews    
        if not use_labels:
            data = [0 for i in range(len(dataframe))]
            dataframe.insert(1, 'labels', data)
        dataframe = google_clean_reviews(dataframe)
        #convert cloumns to numbers to be used more easly on any dataframe
        dataframe.columns = [str(i) for i in range(len(dataframe.columns))]    
        df = pd.DataFrame(dataframe['0'])
        df['labels'] = dataframe['1']
        df.columns = ['reviews', 'labels']

    # if the dataframe containing the reviews is given proceess that dataframe  
    elif dataframe is not None and data_path is None:
        original_col = dataframe.columns
        # if a str value for column is given process it as the text coulmn
        if column and isinstance(column, str):
            if not use_labels:
                text_original = 0
                for i, col in enumerate(dataframe.columns):
                    if col == column:
                        text_original = i
                # storing the column
                target = dataframe[column]
                # dropping it from the dataframe
                dataframe.drop(column, axis=1, inplace=True)
                # reinserting it at the start
                dataframe.insert(0, 'text', target)
            else:
                raise Exception('if you want to use labels please provide the column as a list with'
                                +"1) column name for texts"
                                +"2) column name for lables")
        # if a list value for column is given process it as the text coulmn, label columns
        elif column and isinstance(column, list):
            if use_labels:
                text_original = 0
                label_original = 0
                for i, col in enumerate(dataframe.columns):
                    if col == column[0]:
                        text_original = i
                    if col == column[1]:
                        label_original = i                    
                # storing the column
                target = dataframe[column[0]]
                # dropping it from the dataframe
                dataframe.drop(column[0], axis=1, inplace=True)
                # reinserting it at the start
                dataframe.insert(0, 'text', target)
                # storing the column
                target = dataframe[column[1]]
                # dropping it from the dataframe
                dataframe.drop(column[1], axis=1, inplace=True)
                # reinserting it at after the text column
                dataframe.insert(1, 'labels', target)                
            else:
                raise Exception('if you do not want to use labels please provide the column as a string with the column name for texts')
        # if there's no labels give a dummy value of 0 for the length of the reviews    
        if not use_labels:
            data = [0 for i in range(len(dataframe))]
            dataframe.insert(1, 'labels', data)
        dataframe = google_clean_reviews(dataframe)
        # convert cloumns to numbers to be used more easly on any dataframe
        dataframe.columns = [str(i) for i in range(len(dataframe.columns))]
        df = pd.DataFrame(dataframe['0'])
        df['labels'] = dataframe['1']
        df.columns = ['reviews', 'labels']

    # if the reviews (list) and only a reviewd is given process that list of reviews
    if reviews is not None and dataframe is None and data_path is None:
        df = pd.DataFrame(reviews, columns=['reviews'])
        # of no labels given assign a dummy variable of one
        if labels is None:
            data = [0 for i in range(len(df))]
            df.insert(1, 'labels', data)
            df = google_clean_reviews(df)
        else:
            df['labels'] = labels
            df = google_clean_reviews(df)
    
    # tranform the reviews and labels to a list to tokenize them
    reviews = list(df['reviews'])
    labels = list(df['labels'])

    # tokenize review
    reviews_ints,labels = tokenize_data(reviews, labels, vocab_to_int, lebals_encoded=True)  
        
    # pad tokenized sequence
    features = pad_features(reviews_ints, seq_length)
    
    # convert to tensor to pass into model
    feature_tensor = torch.from_numpy(features)
    labels = torch.from_numpy(np.array(labels))
    pred_data = TensorDataset(feature_tensor, labels)
    
    if(len(features)) < batch_size:
        for i in range(1000):
            if(len(features)) < batch_size:
                batch_size = batch_size//10
            else:
                break
        pred_loader = DataLoader(pred_data, shuffle=False, batch_size=batch_size)
    else:
        pred_loader = DataLoader(pred_data, shuffle=False, batch_size=batch_size)
    
    pred_list = []
    
    for i, (data, lebals) in enumerate(pred_loader):
        # initialize hidden state
        h = net.init_hidden(data.shape[0])

        if(gpu):
            data = data.cuda()

        # get the output from the model        
        output, h = net(data, h)
        
        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze()) 
 
        pred = pred.cpu()
        pred = pred.detach().numpy().tolist()

        for i in pred:
            pred_list.append(i)
 
    
    df['pred'] = pred_list
    
    if stats and use_labels:
        positive_index = df[df['pred'] == 1].index
  
        negative_index = df[df['pred'] == 0].index
        
        postive_accuracy  = df.iloc[positive_index, 1] == df.iloc[positive_index, 2]

        if postive_accuracy.all() == True:
            TP = len(postive_accuracy)
            FP = 0
        elif postive_accuracy.any() == False:
            TP = 0
            FP = len(postive_accuracy)
        else:
            postive_accuracy = postive_accuracy.value_counts()
            TP = postive_accuracy[True]
            FP = postive_accuracy[False]

        negative_accuracy  = df.iloc[negative_index, 1] == df.iloc[negative_index, 2]

        if negative_accuracy.all() == True:
            TN = len(negative_accuracy)
            FN = 0
        elif negative_accuracy.any() == False:
            TN = 0
            FN = len(negative_accuracy)     
        else:
            negative_accuracy = negative_accuracy.value_counts()
            TN = negative_accuracy[True]
            FN = negative_accuracy[False]        

        accuracy  = (TP+TN)/(TP+FP+FN+TN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        try:
            print('accuracy = {:.2f}%'.format(accuracy*100))
            print('recall = {:.2f}%'.format(recall*100))
            print('Precision  = {:.2f}%'.format(precision*100))
            print('F1 score  = {:.2f}%'.format((2*(recall * precision) / (recall + precision))*100))
            print('TP:{}, TN:{}, FN:{}, FP:{}'.format(TP, TN, FN, FP))
        except ZeroDivisionError:
            raise Exception ('division by zero in the stats explore the sent data and make sure the data is large enough')
    
    
    if not use_labels:
        if dataframe is not None:
            dataframe.drop('1', axis=1, inplace=True)
        df.drop('labels', axis=1, inplace=True)
    # if specific column is provided reinsert it to it's original place    
    if column and isinstance(column, str):
        if not use_labels:
            #storing the column
            target = dataframe['0']
            #dropping it from the dataframe
            dataframe.drop('0', axis=1, inplace=True)
            #reinserting text column to it's orginal location
            dataframe.insert(text_original, 'text', target)
    # if specific columns (text,labels) is provided reinsert them to thier original place    
    elif column and isinstance(column, list):
        if use_labels:
            #storing the column
            target = dataframe['0']
            #dropping it from the dataframe
            dataframe.drop('0', axis=1, inplace=True)
            #reinserting text column to it's orginal location
            dataframe.insert(text_original, 'text', target)
            #storing the column
            target = dataframe['1']
            #dropping it from the dataframe
            dataframe.drop('1', axis=1, inplace=True)
            #reinserting label column to it's orginal location
            dataframe.insert(label_original, 'labels', target)            
    if dataframe is not None or data_path is not None:
        dataframe.columns = original_col
        dataframe['pred'] = pred_list
        return dataframe
        
    return df    


def l1_penalty(params, l1_lambda=0.0001):
    """Returns the L1 penalty of the params."""
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_lambda*l1_norm

def train(net, train_loader, valid_loader, optimizer, vocab_to_int, batch_size, seq_length, epochs, print_every,
          train_on_gpu, save_dic='test_01.pth', l1_lambda=None):
    
    print_every = int((len(train_loader.dataset)/batch_size)/print_every)
    
    criterion = nn.BCELoss()
    
    counter = 0
    clip=5 # gradient clipping

    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            if l1_lambda:
                loss = criterion(output.squeeze(), labels.float()) + l1_penalty(net.parameters(), l1_lambda)
            else:
                loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                    valid_loss = np.mean(val_losses)
                    
                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(valid_loss))
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                    checkpoint = {'state_dict': net.state_dict(),
                                  'vocab_to_int': vocab_to_int,
                                  'output_size' : net.output_size,
                                  'embedding_dim' : net.embedding_dim,
                                  'hidden_dim' : net.hidden_dim,
                                  'n_layers' : net.n_layers,
                                  'drop_prob' : net.drop_prob,
                                  'bidirectional' : net.bidirectional,
                                  'seq_length' : seq_length
                                 }
                    torch.save(checkpoint, save_dic)
                    valid_loss_min = valid_loss

                    
                    
def test(test_loader, train_on_gpu, batch_size, net=None, model_path=None):
    
    criterion = nn.BCELoss()
    
    if (not net) and (not model_path):
        raise Exception('please specify either a net or a file path holding a net')
        
    if not net:
        checkpoint = torch.load(model_path)
        
        vocab_to_int = checkpoint['vocab_to_int']
        
        vocab_size = len(vocab_to_int)+1
        if checkpoint['seq_length']: 
            seq_length = checkpoint['seq_length']
        else:
            seq_length = 40
        net = SentimentRNN(vocab_size = vocab_size, output_size = checkpoint['output_size'],
                           embedding_dim = checkpoint['embedding_dim'], hidden_dim = checkpoint['hidden_dim'],
                           n_layers = checkpoint['n_layers'], train_on_gpu=train_on_gpu,
                           drop_prob = checkpoint['drop_prob'], bidirectional = checkpoint['bidirectional'])

        net.load_state_dict(checkpoint['state_dict'])  
    # Get test data loss and accuracy
    if(train_on_gpu):
            net.cuda()

    test_losses = [] # track loss
    num_correct = 0

    
    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for batch_i, (inputs, labels) in enumerate(test_loader, 1):

        n_batches = len(test_loader.dataset)//batch_size
        if(batch_i > n_batches):
            break

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


