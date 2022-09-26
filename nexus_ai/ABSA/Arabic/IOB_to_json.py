import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from nltk.stem.isri import ISRIStemmer
import csv

def write_json_IOB(texts, texts_extacted_labels, outputpath, normalize=True):
    '''
    texts: a list of input texts
    texts_extacted_labels: a list containing lists of labels in a text, for each input text
    outputpath: path to save the data
    normalize: boolean flag to indicate whatever to normalize input arabic text
    '''
    tokens = []
    tags = []
    for i in range(len(texts)):
        drop = False
        aspects = texts_extacted_labels[i]
        text = texts[i]
        # process the aspect so it can be used later on to create the target
        if normalize:
            for j, aspect in enumerate(aspects[:]):
                aspects[j] = normalize_unicode(aspects[j])
                # Normalizing alef variants to (ا)
                aspects[j] = normalize_alef_ar(aspects[j])
                # Normalizing alef maksura (ى) to yeh (ي)
                aspects[j] = normalize_alef_maksura_ar(aspects[j])
                # Normalizing teh marbuta (ة) to heh (ه)
                aspects[j] = normalize_teh_marbuta_ar(aspects[j])
                # removing Arabic diacritical marks
                aspects[j] = dediac_ar(aspects[j])
                # aspects[j] = [st.stem(word) for word in aspects[j]]
            # normalize input text
            text = normalize_unicode(text)
            # Normalizing alef variants to (ا)
            text = normalize_alef_ar(text)
            # Normalizing alef maksura (ى) to yeh (ي)
            text = normalize_alef_maksura_ar(text)
            # Normalizing teh marbuta (ة) to heh (ه)
            text = normalize_teh_marbuta_ar(text)
            # removing Arabic diacritical marks
            text = dediac_ar(text)

        # split each aspect
        for j, aspect in enumerate(aspects[:]):
            aspects[j] = simple_word_tokenize(aspects[j])  
        # split the text 
        text_split = simple_word_tokenize(text)
        # text_split = [st.stem(word) for word in text_split]
        
        # create target list where the start of the aspect is 1, the inside is 2, and non aspects are 0
        row_tags = np.zeros((len(text_split),), dtype=np.int16)
        for aspect in aspects:
            # assgin tags for the aspect
            for i, word in enumerate(aspect):
                try:
                    # assign one for the start of the aspect
                    if i == 0:
                        row_tags[text_split.index(word)] = 1
                    # assign 2 for the remaining words of the aspect
                    else:
                        row_tags[text_split.index(word)] = 2
                except ValueError:
                    drop = True
        if not drop:                         
            tokens.append(text_split)
            tags.append(row_tags.flatten().tolist())


    dict = {'Tokens': tokens, 'Tags': tags}     
    df = pd.DataFrame(dict) 

    df.to_json(outputpath) 