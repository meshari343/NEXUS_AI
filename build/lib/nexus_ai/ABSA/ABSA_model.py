import pandas as pd
from pyabsa import ATEPCCheckpointManager
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nexus_ai.utilities import process_google_reviews, remove_emoji, clean_review

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
    reviews = process_google_reviews(reviews, source=source, sources=sources, only_english=True)
    deleted_idx = []
    # remove emojis
    reviews = [remove_emoji(review) if review != None else None for review in reviews]
    # clean english reviews
    for i, review in enumerate(reviews):
        if review != None:
            try:
                if detect(review) == 'en':
                    reviews[i] = clean_review(review, transform_punct=False, remove_punct=True)
                # if the review is not english assign none to it for deleting later on
                else:
                    reviews[i] = None
            # if the text is short LangDetect would raise an exception
            # or if the text contain only numbers/symbols
            except LangDetectException:
                # if the review contain only numbers/symbols assign none to it for deleting later on
                reviews[i] = None

    processed_reviews = []
    # remove outliers (zreo length reviews)
    for i, review in enumerate(reviews):
        if review != None and review != '':
            processed_reviews.append(review)
        else: 
            deleted_idx.append(i)

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