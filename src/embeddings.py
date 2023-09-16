import re
import torch
import logging
import nltk
from nltk.corpus import stopwords
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
embedding_model = RobertaModel.from_pretrained('roberta-base')
nltk.download("stopwords")


def clean_up_text(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub("\[.*\]", "", sentence)  # remove sections markings in brackets
    sentence = re.sub(r'[^\w\s]', ' ', sentence)  # remove punctuation
    sentence = re.sub(r'\d', '', sentence)  # remove numbers
    sentence = re.sub(r'\b\w\b', '', sentence)  # remove single characters
    sentence = re.sub(r'^\w\s', '', sentence)  # remove single characters from the start
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # remove extra spaces
    sentence = re.sub(r'.*lyrics', '', sentence)  # remove lyrics word as it is in all songs
  
    return sentence


def remove_stop_words(text):
    text = [word for word in text.split() 
            if word not in stopwords.words("english")]
    text = ' '.join(text)
    return text


def embed(sentence):
    tokens = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=512,
                                return_token_type_ids=True, padding="max_length",
                                truncation=True)
    input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
    with torch.no_grad():
        outputs = embedding_model(input_ids, attention_mask=attention_mask)
    return torch.mean(outputs[0], dim=1).squeeze()
