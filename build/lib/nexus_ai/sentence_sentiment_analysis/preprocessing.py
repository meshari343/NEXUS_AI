from collections import Counter
import numpy as np
import pandas as pd
from string import punctuation
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from torch.utils.data import TensorDataset, DataLoader
import torch

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u23F0"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# old method used in testing
def google_clean_reviews(data, punct=True, stopwords_=False, stemm=False):
    original_col = data.columns
    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    token_dict = {'.': '||*Period*||', ',': '||*Comma*||', '"': '||*Quotation_Mark*||', 
                    ';': '||*Semicolon*||', '!': '||*Exclamation_mark*||', '?': '||*Question_mark*||',
                    '(': '||*Left_Parentheses*||', ')':'||*Right_Parentheses*||', '-': '||*Dash*||', '\n': '||*Return*||'}

    data.columns = [str(i) for i in range(len(data.columns))]
    
    #proccess lables into 1/0 which is pos/neg instead of scores
    data['1'] = data['1'].apply(lambda x:
                            1 if isinstance(x, str) and float(x[6:9]) >= 2.5 else 
                            (0 if isinstance(x, str) and float(x[6:9]) <= 2 else 
                            (1 if isinstance(x, int) and x >= 2.5 else
                            (0 if isinstance(x, int) and x <= 2 else None))))
    #dropping rows with no condition met
    data.dropna(inplace=True)
    data = data.reset_index(drop=True)
    #remove emojis
    data['0'] = data['0'].apply(lambda x: remove_emoji(x))
    #remove outliers (zreo length reviews)
    lengths = data['0'].apply(lambda x: len(x))
    zero_idx = data[lengths == 0].index
    data.drop(zero_idx, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    #lower case all charcters to normalize them 
    data['0'] = data['0'].str.lower()
    #clearing stop words for example (a, we, on)
    #steemming the words(returning them to their root for example (stopped --> stop)
    if stopwords_ and stemm:
        data['0'] = data['0'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() 
                                              if not word in stop_words]))
    elif stopwords_:
        data['0'] = data['0'].apply(lambda x: ' '.join([word for word in x.split() 
                                            if not word in stop_words]))            
    elif stemm:
        data['0'] = data['0'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))         
    #removing google transliation
    data['0'] = data['0'].str.replace("(translated by google)", '', regex=False)
    data['0'] = data['0'].str.replace("(original)", '%*?<>!', regex=False)
    data['0'] = data['0'].str.partition('%*?<>!', expand=False)
    data['0'] = data['0'].apply(lambda x: x[0])
    #remove outliers (zreo length reviews)
    lengths = data['0'].apply(lambda x: len(x))
    zero_idx = data[lengths == 0].index
    data.drop(zero_idx, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    #remove non english reviews
    l = lambda x: x if detect(x) == 'en' else None
    for i in range(len(data)):
        if(len(data.iloc[i,0]) > 40):
            try:
                data.iloc[i,0] = l(data.iloc[i,0])
            # if the number of words in the text is shorter than 10 LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                pass
    data.dropna(axis=0, inplace=True)
    data = data.reset_index(drop=True)
    #replace punctuation with token_dict values speceifed above 
    #while putting space as to not create multiple instance of the same word 
    #for example eating. would be treated as a new word diffrent from eating
    if punct:
        for key, token in token_dict.items():
            data['0'] = data['0'].str.replace(key, ' {} '.format(token), regex=False)
    else:
        data['0'] = data['0'].apply(lambda x: [c for c in x if c not in punctuation])
        data['0'] = data['0'].str.join('')
    #remove outliers (zreo length reviews)
    lengths = data['0'].apply(lambda x: len(x))
    zero_idx = data[lengths == 0].index
    data.drop(zero_idx, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    #reseting the column names
    data.columns = original_col
    return data

def clean_reviews(reviews, punct=True, stopwords_=False, stemm=False):
    
    stop_words = stopwords.words('english')
    
    ps = PorterStemmer()
    
    token_dict = {'.': '||*Period*||', ',': '||*Comma*||', '"': '||*Quotation_Mark*||', 
                ';': '||*Semicolon*||', '!': '||*Exclamation_mark*||', '?': '||*Question_mark*||',
                '(': '||*Left_Parentheses*||', ')':'||*Right_Parentheses*||', '-': '||*Dash*||', '\n': '||*Return*||'}
    
    #remove non english reviews
    for i in range(len(reviews)):
         if(len(reviews[i]) > 40):
            try:
                if not (detect(reviews[i]) == 'en'):
                    reviews[i] = ''
            # if the number of words in the text is shorter than 10 LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                pass
    
    for i in range(len(reviews)):
        #lower case all charcters to normalize them 
        reviews[i] = reviews[i].lower()
        
        #clearing stop words for example (a, we, on)
        #steemming the words(returning them to their root for example (stopped --> stop)
        if stopwords_ and stemm:
            words = reviews[i].split()
            reviews[i] = ' '.join([ps.stem(word) for word in words if not word in stop_words])
        elif stopwords_:
            words = reviews[i].split()
            reviews[i] = ' '.join([word for word in words if not word in stop_words])            
        elif stemm:
            words = reviews[i].split()
            reviews[i] = ' '.join([ps.stem(word) for word in words])
            
        #replace punctuation with token_dict values speceifed above 
        #while putting space as to not create multiple instance of the same word 
        #for example eating. would be treated as a new word diffrent from eating
        if punct:
            for key, token in token_dict.items():
                reviews[i] = reviews[i].replace(key, ' {} '.format(token))
        else:
            reviews[i] = ''.join([c for c in reviews[i] if c not in punctuation])#remove punctuation
        
    #adding the words into a list to create a vocabulary later 
    words = []
    step = 100
    for index in range(0, len(reviews), step):
        words.append(' '.join(reviews[index:index+step]).split())
        
    return  reviews, words


def creat_vocab(words, size=None, greater_than=None):
    words_counter = Counter(words)
    words_set = sorted(Counter(words),key=Counter(words).get,reverse=True)
    vocab_to_int = []
    
    if size:
        vocab_to_int = {word: index for (index,word),stop in zip(enumerate(words_set, 1), range(size))}
    elif greater_than:
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)  if words_counter[word] > greater_than}
    else:
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)}
    
    
    return vocab_to_int


def clean_reviews_and_creat_vocab(reviews, size=None, greater_than=None, punct=True, stopwords_=False, stemm=False):

    stop_words = stopwords.words('english')
    
    ps = PorterStemmer()
    
    #remove non english reviews
    for i in range(len(reviews)):
        if(len(reviews[i]) > 40):
            try:
                if not (detect(reviews[i]) == 'en'):
                    reviews[i] = ''
            # if the number of words in the text is shorter than 10 LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                pass
    
    token_dict = {'.': '||*Period*||', ',': '||*Comma*||', '"': '||*Quotation_Mark*||', 
                ';': '||*Semicolon*||', '!': '||*Exclamation_mark*||', '?': '||*Question_mark*||',
                '(': '||*Left_Parentheses*||', ')':'||*Right_Parentheses*||', '-': '||*Dash*||', '\n': '||*Return*||'}

    for i in range(len(reviews)):
        #lower case all charcters to normalize them 
        reviews[i] = reviews[i].lower()
                    
        #replace punctuation with token_dict values speceifed above 
        #while putting space as to not create multiple instance of the same word 
        #for example eating. would be treated as a new word diffrent from eating
        if punct:
            for key, token in token_dict.items():
                reviews[i] = reviews[i].replace(key, ' {} '.format(token))
        else:
            reviews[i] = ''.join([c for c in reviews[i] if c not in punctuation])#remove punctuation
            
        #clearing stop words for example (a, we, on)
        #steemming the words(returning them to their root for example (stopped --> stop)    
        if stopwords_ and stemm:
            words = reviews[i].split()
            reviews[i] = ' '.join([ps.stem(word) for word in words if not word in stop_words])
        elif stopwords_:
            words = reviews[i].split()
            reviews[i] = ' '.join([word for word in words if not word in stop_words])            
        elif stemm:
            words = reviews[i].split()
            reviews[i] = ' '.join([ps.stem(word) for word in words])        
        
    
    #adding the words into a counter to create a vocabulary later 
    words = []
    words_counter = Counter()
    step = 100
    for i in range(0, len(reviews), step):
        words = ' '.join(reviews[i:i+step]).split() 
        for word in words:
            if words_counter[word]:
                words_counter[word] += 1
            else:
                words_counter[word] = 1
    #sorting the counter from highest words counted to smallest       
    words_set = sorted(words_counter, key=words_counter.get, reverse=True) 
    #creating the vocab
    if size:
        vocab_to_int = {word: index for (index,word),stop in zip(enumerate(words_set, 1), range(size))}
    elif greater_than:
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)  if words_counter[word] > greater_than}
    else:
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)}
    
    return  reviews, vocab_to_int



def tokenize_data(reviews, vocab_to_int, labels=None, labels_encoded=False):
    # encoding reviews
    reviews_ints = []
    for review_words in reviews:
        reviews_ints.append([vocab_to_int.get(word, 0) for word in review_words.split()])
        
    #encoding lebals
    if labels:
        if not labels_encoded:
            encoded_labels = []
            for label in labels:
                if label == 'positive':
                    encoded_labels.append(1)
                elif label == 'negative':
                    encoded_labels.append(0)
        else:
            encoded_labels = labels
        
    # get indices of any reviews with length above 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    # remove 0-length reviews and their labels if there's labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    if labels:
        encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
        return reviews_ints,encoded_labels
    else:
        return reviews_ints
    


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    features = np.empty((len(reviews_ints), seq_length))
    for idx, review in enumerate(reviews_ints):
        # print(review)
        if len(review) <= seq_length:
            features[idx, :len(review)] = review
            features[idx, len(review):seq_length] = 0.0
        else:
            features[idx, :] = review[:seq_length]
    
    return features


def train_test_split(features, encoded_labels, numclasses=2, train_frac=0.8, balanced=False):
    '''
        train test split that deals with balanced data and, return three sets train, validate, test.
    '''

    if numclasses < 2:
        print('wrong value for numclasses must be above 2')
        return
    if balanced:
        print('make sure the classes (labels) are sorted to have a balanced splits')
    split_frac_remain = 1 - train_frac
    
    ## split data into training, validation, and test data (features and labels, x and y)
    if not balanced:
        batches = features.shape[0]
        train_end = int(batches*train_frac)
        validate_end = int(train_frac + batches*(split_frac_remain/2))

        train_x = features[:train_end]
        train_y = encoded_labels[:train_end]

        validate_x = features[train_end:validate_end]
        validate_y = encoded_labels[train_end:validate_end]

        test_x = features[validate_end:]
        test_y = encoded_labels[validate_end:]
    else:
        # calculate the size for the train, validate, test
        batches = features.shape[0]
        train_size = int((batches*train_frac))
        validate_size = int((batches*(split_frac_remain/2)))
        test_size = int((batches*(split_frac_remain/2)))

        train_class_size= int((train_size)/numclasses)
        validate__class_size = int((validate_size)/numclasses)
        test_class_size = int((test_size)/numclasses)

        # concat each part of the data to the split e.g. 2 classes after taking the first half from the start
        # would take the second half from the start of half the batches
        for class_ in range(0, numclasses):
            # calculate class start to to use in getting the train end
            class_start = int(batches/numclasses) * class_ 
            # if first class initialize the arrays
            if class_ == 0:
                train_end = (class_start + train_class_size)
                validate_end = train_end + (validate__class_size)
                test_end = validate_end + (test_class_size)

                train_x = np.array(features[:train_end])
                train_y = np.array(encoded_labels[:train_end])

                validate_x = np.array(features[train_end:validate_end])
                validate_y = np.array(encoded_labels[train_end:validate_end])   

                test_x = np.array(features[validate_end:test_end])
                test_y = np.array(encoded_labels[validate_end:test_end])
            else:
                train_end = (class_start + train_class_size)
                train_x = np.concatenate( (train_x, features[test_end:train_end]))
                train_y = np.concatenate( (train_y, encoded_labels[test_end:train_end]))

                validate_end = train_end + (validate__class_size)
                validate_x = np.concatenate( (validate_x, features[train_end:validate_end]))
                validate_y = np.concatenate( (validate_y, encoded_labels[train_end:validate_end]))   

                test_end = validate_end + (test_class_size)
                test_x = np.concatenate( (test_x, features[validate_end:test_end]))
                test_y = np.concatenate( (test_y, encoded_labels[validate_end:test_end]))

    return train_x, validate_x, test_x, train_y, validate_y, test_y


def creat_dataloader(train_x, validate_x, test_x, train_y, validate_y, test_y, batch_size= 128):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(validate_x), torch.from_numpy(validate_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader