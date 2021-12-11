import pandas as pd
from transformers import BertModel, BertTokenizerFast
import numpy as np
import torch

alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertModel.from_pretrained('onlplab/alephbert-base')


def get_word_vector(sent, word):
    ind = sent.split(" ").index(word)
    input_encodings = alephbert_tokenizer.encode_plus(sent)
    input_ids = torch.tensor(input_encodings['input_ids']).unsqueeze(0)  # Batch size 1
    outputs = alephbert(input_ids)
    last_hidden_state = outputs[0]
    return last_hidden_state[:, np.where(np.array(input_encodings.word_ids()) == ind), :].mean(dim=2)

CONTEXT_WINDOW = 4

def add_aleph_bert_vector(file_path):
    df = pd.read_pickle(file_path)
    df.word = df.word.apply(lambda word: word.replace("*", "").replace("_", "").replace(" ",""))
    df['word_context'] = [" ".join(list(w.values)) for w in list(df.word.rolling(CONTEXT_WINDOW))]
    df['ab_embedding'] = df.apply(lambda row: get_word_vector(row['word_context'], row['word']), axis=1)
    df.to_pickle(file_path)
import os
from tqdm import tqdm
files = os.listdir("word_lists")
for fp in tqdm(files):
    add_aleph_bert_vector(os.path.join("word_lists", fp))
