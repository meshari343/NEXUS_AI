from transformers import BertTokenizer
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from nexus_ai.ABSA.arabic.bert import bert_ATE, bert_APC
from nexus_ai.ABSA.arabic.dataset import pred_dataset_ATE
from nexus_ai.sentence_sentiment_analysis.preprocessing import pad_features

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_model_name = "nexus_ai/sentence_sentiment_analysis/arabic/models/bert_pretrained_01_acc_90.50"
tokenizer = BertTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-msa")

model_APC = bert_APC(pretrain_model_name).to(DEVICE)
model_APC.load_state_dict(torch.load('nexus_ai/ABSA/arabic/models/bert_APC.pt'))
model_APC.eval()

model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
model_ATE.load_state_dict(torch.load('nexus_ai/ABSA/arabic/models/bert_ATE.pt'))
model_ATE.eval() 

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False

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
    # split the text  into tokens
    text = simple_word_tokenize(text)  
    
    return text


def create_mini_batch_ATE(samples):
    ids_tensors = [s[0] for s in samples]

    ids_tensors = torch.Tensor(pad_features(ids_tensors, 128)).long()

    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, masks_tensors


def detokinize(tokens, tags):
    restored_tokens = []
    restored_tags = []
    # detokinize the results
    for review_tokens_idx in range(len(tokens)):
        review_restore_tokens = []
        review_restore_tags = []
        for token_idx in range(len(tokens[review_tokens_idx])):
            if not is_subtoken(tokens[review_tokens_idx][token_idx]) and (token_idx+1)<len(tokens[review_tokens_idx]) and is_subtoken(tokens[review_tokens_idx][token_idx+1]):
                review_restore_tokens.append(tokens[review_tokens_idx][token_idx] + tokens[review_tokens_idx][token_idx+1][2:])  
                j = 2
                while(token_idx+j<len(tokens[review_tokens_idx]) and len(tokens[review_tokens_idx]+[token_idx+j])<len(tokens) and is_subtoken(tokens[review_tokens_idx][token_idx+j])):
                    review_restore_tokens[-1] = review_restore_tokens[-1] + tokens[review_tokens_idx][token_idx+j][2:]
                    j += 1
                review_restore_tags.append(tags[review_tokens_idx][token_idx]) 
            elif not is_subtoken(tokens[review_tokens_idx][token_idx]):
                review_restore_tokens.append(tokens[review_tokens_idx][token_idx])      
                review_restore_tags.append(tags[review_tokens_idx][token_idx]) 
                
        restored_tokens.append(review_restore_tokens)
        restored_tags.append(review_restore_tags)
    
    return restored_tokens, restored_tags

def predict_model_APC(tokens, aspect):

    aspect = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += tokens
    word_pieces += ['[sep]']
    word_pieces += aspect

    segment = [0] + [0]*len(tokens) + [0] + [1]*len(aspect)
    segment_2d = []
    segment_2d.append(segment)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    ids_2d = []
    ids_2d.append(ids)
    input_tensor = torch.Tensor(pad_features(ids_2d, 128)).long()
    segment_tensor = torch.Tensor(pad_features(segment_2d, 128)).long()

    masks_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
    masks_tensor = input_tensor.masked_fill(input_tensor != 0, 1)  

    input_tensor = input_tensor.to(DEVICE)
    segment_tensor = segment_tensor.to(DEVICE)
    masks_tensor = masks_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model_APC(input_tensor, None, masks_tensor, segment_tensor)
        predictions = F.softmax(outputs, dim=1).max(dim=1)[1].cpu().tolist()[0]

    return predictions


def predict_model_ATE(pred_loader):

    for data in pred_loader:
        ids_tensors, mask_tensors = data
        ids_tensors = ids_tensors.to(DEVICE)
        mask_tensors = mask_tensors.to(DEVICE)
        outputs = model_ATE(ids_tensors, None, mask_tensors)
        _, predictions = torch.max(outputs, dim=2)


    return predictions.tolist()


def pred(reviews):
    if len(reviews)==0:
        return []
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    reviews = [text_preprocessing(review) for review in reviews] 

    pred_ds = pred_dataset_ATE(reviews, tokenizer)
    pred_loader = DataLoader(pred_ds, batch_size=100, collate_fn=create_mini_batch_ATE)
    
    # extract aspects for each review
    tags = predict_model_ATE(pred_loader)

    tokens = [sample[1] for sample in pred_ds]
    restored_tokens, restored_tags = detokinize(tokens, tags)

    aspects = []
    # extract terms/aspects
    for review_tags in range(len(restored_tags)):
        review_aspects = []
        # if no tags was found using ATE model add an empty array to the list
        if all(tag == 0 for tag in restored_tags[review_tags]):
            aspects.append(review_aspects)
            continue
        for tags in range(len(restored_tags[review_tags])):
            if restored_tags[review_tags][tags] == 1:
                review_aspects.append(restored_tokens[review_tags][tags])
            if restored_tags[review_tags][tags] == 2:
                if len(review_aspects) != 0:
                    review_aspects[-1] = review_aspects[-1] + restored_tokens[review_tags][tags]
                else:
                    review_aspects.append(restored_tokens[review_tags][tags])
        aspects.append(review_aspects)

    if len(tokens) != len(aspects):
        logging.warning('irregular output in arabic ABSA (ATE model)')
        return None

    aspects_polarity = []
    # predict polarity for each aspect 
    for review_aspects_idx in range(len(aspects)):
            review_aspects_polarity = []
            review_aspects = aspects[review_aspects_idx]
            if len(review_aspects) == 0:
                aspects_polarity.append(review_aspects_polarity)
                continue
            for aspect in review_aspects:
                pred= predict_model_APC(tokens[review_aspects_idx], aspect)
                review_aspects_polarity.append(pred)
            aspects_polarity.append(review_aspects_polarity)    
                
    aspects_polarity = [['Negative' if polarity == 0 else('Neutral' if polarity == 1 else ('Positive' if polarity == 2 else None)) for polarity in aspects] for aspects in  aspects_polarity]
    # dummy empty aspcets descriptions
    aspects_description = [[] for i in range(len(aspects))]
    results = [[ aspects[i], aspects_polarity[i], aspects_description[i] ] for i in  range(len(aspects))]


    return results
