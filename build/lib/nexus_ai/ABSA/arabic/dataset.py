from torch.utils.data import Dataset
import pandas as pd
import torch

class pred_dataset_ATE(Dataset):
    def __init__(self, reviews, tokenizer):
        self.reviews = reviews
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        review = self.reviews[idx]

        bert_tokens = []
        for i in range(len(review)):
            t = self.tokenizer.tokenize(review[i])
            bert_tokens += t
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)

        return  ids_tensor, bert_tokens

    def __len__(self):
        return len(self.reviews)

class pred_dataset_APC(Dataset):
    def __init__(self, reviews, aspcets, tokenizer):
        self.reviews = reviews
        self.aspects = aspcets
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        review = self.reviews[idx]
        review_aspects = self.aspects[idx]

        bert_tokens = []
        for i in range(len(review)):
            t = self.tokenizer.tokenize(review[i])
            bert_tokens += t

        segment_tensor = [0] + [0]*len(bert_tokens) + [0] + [1]*len(review_aspects)

        bert_tokens = ['[cls]'] + bert_tokens + ['[sep]'] + review_aspects

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        
        ids_tensor = torch.tensor(bert_ids)
        segment_tensor = torch.tensor(segment_tensor)

        return ids_tensor, segment_tensor, bert_tokens

    def __len__(self):
        return len(self.reviews)

class dataset_ATE(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_tags = []
        bert_pols = []
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pols += [int(pols[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pols_tensor = torch.tensor(bert_pols)

        return bert_tokens, ids_tensor, tags_tensor, pols_tensor

    def __len__(self):
        return len(self.df)
    
    
class dataset_APC(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values
        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_att = []
        pols_label = 0
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            if int(pols[i]) != -1:
                bert_att += t
                pols_label = int(pols[i])

        segment_tensor = [0] + [0]*len(bert_tokens) + [0] + [1]*len(bert_att)
        bert_tokens = ['[cls]'] + bert_tokens + ['[sep]'] + bert_att
        

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        pols_tensor = torch.tensor(pols_label)
        segment_tensor = torch.tensor(segment_tensor)

        return bert_tokens, ids_tensor, segment_tensor, pols_tensor

    def __len__(self):
        return len(self.df)