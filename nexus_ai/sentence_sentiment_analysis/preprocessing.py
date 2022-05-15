from collections import Counter
import numpy as np
import pandas as pd
from string import punctuation
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
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


def clean_reviews_list(reviews, source='google', labels=None, transform_punct=True, remove_punct=False, stopwords_=False, stemm=False):

    stop_words = stopwords.words('english')
    ps = PorterStemmer()
    token_dict = {'.': '||*Period*||', ',': '||*Comma*||', '"': '||*Quotation_Mark*||', 
                    ';': '||*Semicolon*||', '!': '||*Exclamation_mark*||', '?': '||*Question_mark*||',
                    '(': '||*Left_Parentheses*||', ')':'||*Right_Parentheses*||', '-': '||*Dash*||', '\n': '||*Return*||'}

    # transform to dataframe to make use of linear operations
    if labels:
        data = {'text':reviews,'label':labels}
        df = pd.DataFrame(data, columns=['text','label'])
    else:
        df = pd.DataFrame(reviews, columns=['text'])

    # To make sure to not return the preprocced text, 
    # while at the same time deleting reviews that do not meet the requirements.
    deleted_idx = []

    # if labels are included proccess lables into 1/0 which is pos/neg instead of scores
    if labels:
        df['label'] = df['label'].apply(lambda x:
                                1 if isinstance(x, str) and float(x[6:9]) >= 2.5 else 
                                (0 if isinstance(x, str) and float(x[6:9]) <= 2 else 
                                (1 if isinstance(x, int) and x >= 2.5 else
                                (0 if isinstance(x, int) and x <= 2 else None))))
        none_value = df[df['label'].isnull()].index
        df.drop(none_value, axis=0, inplace=True)
        deleted_idx.extend(list(none_value))
  
    # remove emojis
    df['text'] = df['text'].apply(lambda x: remove_emoji(x))
    # remove outliers (zreo length reviews)
    lengths = df['text'].apply(lambda x: len(x))
    zero_idx = df[lengths == 0].index
    deleted_idx.extend(list(zero_idx))
    df.drop(zero_idx, axis=0, inplace=True)
    # df = df.reset_index(drop=True)
    # drop in not processed
    # df_non_processed.drop(zero_idx, axis=0, inplace=True)
    # df_non_processed = df.reset_index(drop=True)

    # lower case all charcters to normalize them 
    df['text'] = df['text'].str.lower()
    # clearing stop words for example (a, we, on)
    # steemming the words(returning them to their root for example (stopped --> stop)
    if stopwords_ and stemm:
        df['text'] = df['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() 
                                              if not word in stop_words]))
    elif stopwords_:
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() 
                                            if not word in stop_words]))            
    elif stemm:
        df['text'] = df['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))     

    # if the reviews source is google remove google translataion
    if source == 'google': 
        df['text'] = df['text'].str.replace("(translated by google)", '', regex=False)
        df['text'] = df['text'].str.replace("(original)", '%*?<>!', regex=False)
        # split each review into before and after the original
        df['text'] = df['text'].str.partition('%*?<>!', expand=False)
        # take the text before the original (the translated review)
        df['text'] = df['text'].apply(lambda x: x[0])
        # after removing the google translation remove outliers (zreo length reviews)
        lengths = df['text'].apply(lambda x: len(x))
        zero_idx = df[lengths == 0].index
        deleted_idx.extend(list(zero_idx))
        df.drop(zero_idx, axis=0, inplace=True)
        # df = df.reset_index(drop=True)
        # drop in not processed
        # df_non_processed.drop(zero_idx, axis=0, inplace=True)
        # df_non_processed = df.reset_index(drop=True)

    # remove non english reviews
    lambda_ = lambda x: x if detect(x) == 'en' else None
    for i in range(len(df)):
        if(len(df.iloc[i, 0]) > 40):
            try:
                df.iloc[i, 0] = lambda_(df.iloc[i, 0])
            # if the number of words in the text is shorter than 10 LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                pass
    none_value = df[df['text'].isnull()].index
    df.drop(none_value, axis=0, inplace=True)
    deleted_idx.extend(list(none_value))
    # df = df.reset_index(drop=True)
    # drop in not processed
    # df_non_processed.drop(none_value, axis=0, inplace=True)
    # df_non_processed = df.reset_index(drop=True)

    # replace punctuation with token_dict values speceifed above 
    # while putting space as to not create multiple instance of the same word 
    # for example eating. would be treated as a new word diffrent from eating
    if transform_punct:
        for key, token in token_dict.items():
            df['text'] = df['text'].str.replace(key, ' {} '.format(token), regex=False)
    elif remove_punct:
        df['text'] = df['text'].apply(lambda x: [c for c in x if c not in punctuation])
        df['text'] = df['text'].str.join('')

    #remove outliers (zreo length reviews)
    lengths = df['text'].apply(lambda x: len(x))
    zero_idx = df[lengths == 0].index
    df.drop(zero_idx, axis=0, inplace=True)
    deleted_idx.extend(list(zero_idx))
    # df = df.reset_index(drop=True)
    
    # drop in not processed
    # df_non_processed.drop(zero_idx, axis=0, inplace=True)
    # df_non_processed = df.reset_index(drop=True)
    
    if labels:
        return deleted_idx, list(df['text']), df['label']
    else:
        return deleted_idx, list(df['text'])

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
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)  if words_counter[word] > 100}
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
        vocab_to_int = {word: index for index,word in enumerate(words_set, 1)  if words_counter[word] > 100}
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
    features = []
    for review in reviews_ints:
        if len(review) < seq_length:
            future = []
            review_index = 0
            for index in range(seq_length):
                if index < (seq_length - len(review)):
                    future.append(0)
                else:
                    future.append(review[review_index])
                    review_index += 1
        else:
            future = [review_int for index, review_int in enumerate(review) if index < seq_length]
        features.append(future)  
    
    features= np.array(features)
    
    return features


def train_test_split(features, encoded_labels, split_frac=0.8, ordered_by_labels=False):

    split_frac_remain = 1 - split_frac
    
    ## split data into training, validation, and test data (features and labels, x and y)
    if not ordered_by_labels:
        batches = features.shape[0]
        train_size = int(batches*split_frac)
        validate_size = int(train_size + batches*(split_frac_remain/2))

        train_x = features[:train_size]
        train_y = encoded_labels[:train_size]

        validate_x = features[train_size:validate_size]
        validate_y = encoded_labels[train_size:validate_size]

        test_x = features[validate_size:]
        test_y = encoded_labels[validate_size:]
    else:
        batches = features.shape[0]
        train_size_half = int((batches*split_frac)/2)
        validate_size_half = int(((train_size_half*2) + ((batches*split_frac_remain)/2))/2)
        test_size_half = int(validate_size_half + (((batches*split_frac_remain)/2)/2)) 

        #gettin the futures and labels for each dataset
        train_x = np.concatenate( (features[:train_size_half], features[-train_size_half:]))
        train_y = np.concatenate( (encoded_labels[:train_size_half], encoded_labels[-train_size_half:]))

        validate_x = np.concatenate( (features[train_size_half:validate_size_half], features[-validate_size_half:-train_size_half]))
        validate_y = np.concatenate( (encoded_labels[train_size_half:validate_size_half], 
                                      encoded_labels[-validate_size_half:-train_size_half]))                                               
        test_x = np.concatenate( (features[validate_size_half:test_size_half], features[-test_size_half:-validate_size_half]))
        test_y = np.concatenate( (encoded_labels[validate_size_half:test_size_half], encoded_labels[-test_size_half:-validate_size_half]))

    return train_x, validate_x, test_x, train_y, validate_y, test_y


def creat_dataloader(train_x, validate_x, test_x, train_y, validate_y, test_y, batch_size= 128):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(validate_x), torch.from_numpy(validate_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader