import pandas as pd
from pyabsa import ATEPCCheckpointManager
import spacy
from math import ceil   
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

checkpoint_name = 'nexus_ai/ABSA/english/ATEPC_models/fast_lcf_atepc_custom_dataset_cdw_apcacc_88.84_apcf1_80.21_atef1_86.77'

# aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name,eval_batch_size=1000, auto_device='cpu')
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name, eval_batch_size=1000)

nlp = spacy.load("en_core_web_sm")


def pretokenize(tokens):
    '''
    return tokenized text to a string
    '''
    pretok_sent = ''
    for tok in tokens:
        if tok.startswith('##'):
            pretok_sent += tok[2:]
        else:
            pretok_sent += ' ' + tok
    return pretok_sent


def token_split(tokens, max_size):
    '''
    split tokens after toknezing them
    '''
    num_of_splits = ceil(len(tokens) / max_size)
    split_tokens_text = [pretokenize(tokens[(max_size*i):(max_size*(i+1))]) for i in range(num_of_splits)]
    return split_tokens_text


def split_inputs(input_texts, max_size = 64):
    '''
    input_texts: a list of input texts.
    max_size: the max size to truncate data to.
    '''
    nlp = spacy.load("en_core_web_sm")
    new_input_texts = []
    orginal_indexs = []
    
    for i, doc in enumerate(nlp.pipe(input_texts, disable=["tagger", "attribute_ruler", "lemmatizer"])):
        sentences = [sent.text for sent in doc.sents]
        new_sentences = []
        for sent in sentences:
            new_sentences.extend([split_sent for split_sent in token_split(tokenizer.tokenize(sent), max_size)] )
        orginal_indexs.extend([i for sent in new_sentences])
        new_input_texts.extend(new_sentences)

    return new_input_texts, orginal_indexs


def join_outputs(aspects, sentiments, descriptions, orginal_indexs):
    '''
    join splited outputs
    '''
    if not orginal_indexs:
        raise Exception('split inputs indexes are empty')
    if not aspects or not sentiments or not descriptions:
        raise Exception('split outputs are empty')
    new_aspects = aspects[0]
    new_sentiments = sentiments[0]
    new_descriptions = descriptions[0]
    joined_aspects = []
    joined_sentiments = []
    joined_descriptions = []
    len_idxs = len(orginal_indexs)
    # output = split_outputs[0]
    for i in range(len_idxs-1):
        past_index = orginal_indexs[i]
        current_index = orginal_indexs[i+1]
        if i == (len_idxs-2):
            if current_index ==  past_index:
                new_aspects.extend(aspects[i+1])
                new_sentiments.extend(sentiments[i+1])
                new_descriptions.extend(descriptions[i+1])
                joined_aspects.append(new_aspects)
                joined_sentiments.append(new_sentiments)
                joined_descriptions.append(new_descriptions)
            else:
                joined_aspects.append(new_aspects)
                joined_sentiments.append(new_sentiments)
                joined_descriptions.append(new_descriptions)
                new_aspects = aspects[i+1]
                new_sentiments = sentiments[i+1]
                new_descriptions = descriptions[i+1]
                joined_aspects.append(new_aspects)
                joined_sentiments.append(new_sentiments)
                joined_descriptions.append(new_descriptions)
        elif current_index ==  past_index:
            new_aspects.extend(aspects[i+1])
            new_sentiments.extend(sentiments[i+1])
            new_descriptions.extend(descriptions[i+1])
        else:
            joined_aspects.append(new_aspects)
            joined_sentiments.append(new_sentiments)
            joined_descriptions.append(new_descriptions)
            new_aspects = aspects[i+1]
            new_sentiments = sentiments[i+1]
            new_descriptions = descriptions[i+1]

    return joined_aspects, joined_sentiments, joined_descriptions


# add description for each aspect in each row or review
def add_desctiption(row):
    descriptions = []

    doc = nlp(row['sentence'])
    descriptive_terms = []
    # collect all descriptive terms in each review
    for token in doc:
        if token.pos_ == 'ADJ':     
            prepend = ''
            for child in token.children:
                if child.pos_ != 'ADV':
                    continue
                prepend += child.text + ' '
            descriptive_terms.append((prepend + token.text, token.i+1))
            
    if descriptive_terms:
        # for each aspect assign the closest descriptive term to it
        for position in row['position']:
            if not position:
                descriptions.append('')
                continue
            distances = [abs(position[0]-term[1]) for term in descriptive_terms]
            min_value_index = distances.index(min(distances))
            closest_term = descriptive_terms[min_value_index][0]
            descriptions.append(closest_term)

    # add descriptions to the row
    row['description'] = descriptions
    
    return row

def pred(reviews):
    # print('reviews before split')
    # print(reviews)
    reviews, orginal_indexs =  split_inputs(reviews)
    # print('reviews after split')
    # print(reviews)
    prediction = aspect_extractor.extract_aspect(
        inference_source=reviews,
        save_result=False,
        print_result=False,
        pred_sentiment=True
    )

    prediction = pd.DataFrame(prediction)
    
    prediction = prediction.apply(add_desctiption, axis=1)

    prediction.drop(['sentence','IOB', 'tokens', 'position'], axis=1, inplace=True)

    prediction = join_outputs(
        prediction['aspect'].tolist(),
        prediction['sentiment'].tolist(),
        prediction['description'].tolist(),
        orginal_indexs
    )
    # aspects = prediction['aspect'] 
    # sentiment = prediction['sentiment']
    # description = prediction['description']
    # prediction = prediction.values.tolist()
    
    return list(prediction)