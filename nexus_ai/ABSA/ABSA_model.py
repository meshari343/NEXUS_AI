import pandas as pd
from pyabsa import ATEPCCheckpointManager
from nexus_ai.sentence_sentiment_analysis.preprocessing import clean_reviews_list
import spacy

checkpoint_name = 'nexus_ai/ABSA/ATEPC_models/fast_lcf_atepc_custom_dataset_cdw_apcacc_88.84_apcf1_80.21_atef1_86.77'

# aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name,eval_batch_size=1000, auto_device='cpu')
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint_name,eval_batch_size=1000)

nlp = spacy.load("en_core_web_sm")

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


def pred(reviews, source='Google Maps', sources=None):
    deleted_idx, processed_reviews = clean_reviews_list(reviews, source=source, sources=sources, transform_punct=False, remove_punct=True)

    prediction = aspect_extractor.extract_aspect(
        inference_source=processed_reviews,
        save_result=False,
        print_result=False,
        pred_sentiment=True
    )

    prediction = pd.DataFrame(prediction)
    
    prediction = prediction.apply(add_desctiption, axis=1)

    prediction.drop(['sentence','IOB', 'tokens', 'position'], axis=1, inplace=True)

    prediction = prediction.to_dict(orient='list')
    
    return deleted_idx, prediction