import os
import re
import nltk
import logging
import pandas as pd

from typing import Iterable

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


nltk.download("stopwords")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_up_text(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r'[^\w]', ' ', sentence)  # romove punctuation
    sentence = re.sub(r'[0-9]', '', sentence)  # remove numbers
    sentence = re.sub(r'\s[a-z]\s', ' ', sentence)  # remove single characters
    sentence = re.sub(r'^[a-z]\s', '', sentence)  # remove single characters from the start
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # remove extra spaces

    return sentence


def remove_stop_words(text):
    text = [word for word in text.split() 
            if word not in stopwords.words("english")]
    text = ' '.join(text)
    return text


def count_words(tokenizer) -> list:
    word_count = tokenizer.word_counts
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return word_count


def create_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer


def tokenize_text(text: Iterable[str], tokenizer: object) -> list:
    text = tokenizer.texts_to_sequences(text)
    return text

