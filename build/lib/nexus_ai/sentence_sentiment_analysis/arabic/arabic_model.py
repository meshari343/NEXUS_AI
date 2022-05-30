import numpy as np
import pickle
import logging
import pandas as pd
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer
from nexus_ai.sentence_sentiment_analysis.bert import BertClassifier


print('starting arabic sentiment analysis model')
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')
# initialize the MAX_LEN
MAX_LEN = 128
# Loading the model 
# filename = 'nexus_ai/sentence_sentiment_analysis/arabic/models/binary_bert_arabic_01_acc_90.56.sav'
# f = open(filename, 'rb')
# bert_classifier = pickle.load(f)
bert_classifier = BertClassifier()
bert_classifier = bert_classifier.load_state_dict(torch.load('models/binary_bert_arabic_01_acc_90.50.pt', map_location='cpu'))
if torch.cuda.is_available():  
    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)


# function to tokenize a set of texts
def preprocessing_for_bert(data):
    """
    Perform required preprocessing steps for pretrained BERT tokenizer.
    
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            truncation=True,
            max_length=MAX_LEN,             # Max length to truncate/pad
            padding='max_length',           # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def text_preprocessing(text):
    
    text = normalize_unicode(text)
    # Normalizing alef variants to (ا)
    text = normalize_alef_ar(text)
    # Normalizing alef maksura (ى) to yeh (ي)
    text = normalize_alef_maksura_ar(text)
    # Normalizing teh marbuta (ة) to heh (ه)
    text = normalize_teh_marbuta_ar(text)
    # removing Arabic diacritical marks
    text = dediac_ar(text)
    
    return text


def bert_predict(model, dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # Put the model into the evaluation mode. The dropout layers are disabled 
    model.eval()

    all_logits = []

    # For each batch 
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    # get predictions from probabilities
    predictions = torch.Tensor(probs).data.max(1, keepdims=True)[1].flatten()
    # convert labels to classes
    predictions = predictions.cpu().detach().tolist()
    predictions = ['Positive' if prediction == 1 else('Negative' if prediction == 0 else None) for prediction in predictions]

    if None in predictions:     
        logging.warning('irregular output in arabic sentince sentiment analysis')

    return predictions


def pred(reviews):
    # To make sure to not return the preprocced text, 
    # while at the same time deleting reviews that do not meet the requirements.
    inputs, masks = preprocessing_for_bert(np.array(reviews))
    
    # Create DataLoader 
    batch_size = 128
    dataset = TensorDataset(inputs, masks)
    dataset_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=batch_size)

    predictions = bert_predict(bert_classifier, dataloader)

    return predictions