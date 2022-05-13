import pandas as pd
from pyabsa import ATEPCCheckpointManager
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import spacy

checkpoint_name = 'nexus_ai/ABSA/ATEPC_models/fast_lcf_atepc_custom_dataset_cdw_apcacc_88.84_apcf1_80.21_atef1_86.77'

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name)

nlp = spacy.load("en_core_web_lg")

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

    df = pd.DataFrame(reviews, columns=['text'])
    deleted_idx = []

    # remove outliers (zreo length reviews)
    lengths = df['text'].apply(lambda x: len(x))
    zero_idx = df[lengths == 0].index
    deleted_idx.extend(list(zero_idx))
    df.drop(zero_idx, axis=0, inplace=True)

    lambda_ = lambda x: x if detect(x) == 'en' else None
    # detect non english reviews
    for i in range(len(df)):
        if(len(df.iloc[i, 0]) > 40):
            try:
                df.iloc[i, 0] = lambda_(df.iloc[i, 0])
            # if the number of words in the text is shorter than 10 
            # or if the text contain only numbers/symbols LangDetect would raise an exception
            except LangDetectException:
                pass
    # remove non english reviews           
    none_value = df[df['text'].isnull()].index
    df.drop(none_value, axis=0, inplace=True)
    deleted_idx.extend(list(none_value))

    prediction = aspect_extractor.extract_aspect(
        inference_source=list(df['text']),
        save_result=False,
        print_result=False,
        pred_sentiment=True
    )

    prediction = pd.DataFrame(prediction)
    
    prediction = prediction.apply(add_desctiption, axis=1)

    return deleted_idx, prediction