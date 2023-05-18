import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from config import DEVICE, MANUAL_SEED
from embeddings import clean_up_text, remove_stop_words, embed
from dataset import MyDataset
from train import train_pass, test_pass, training_loop
from model_level_1 import Net
from accuracy import multi_class_accuracy

torch.manual_seed(MANUAL_SEED)

DATA_FOLDER = "data/genius_lyrics"
ARTISTS = ["Travis Scott", "Queen", "The Beatles"]
INPUT_UNITS = 768
OUTPUT_UNITS = 3
TRAIN_SIZE = 0.8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 60

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stt = {s: torch.eye(len(ARTISTS))[n] for n, s in enumerate(ARTISTS)}
tts = {str(v): k for k, v in stt.items()}



def encode_label(label: str) -> torch.tensor:
    return stt[label]


def decode_label(label: torch.tensor) -> str:
    return tts[str(label)]


def load_data() -> pd.DataFrame:
    logger.info('Loading data...')
    df = pd.DataFrame()
    for filename in os.listdir(DATA_FOLDER):
        artist_name = filename.split('.')[0]
        if artist_name in ARTISTS:
            temp_df = pd.read_csv(os.path.join(DATA_FOLDER, filename),
                    usecols=['artist', 'title', 'lyrics'])
            df = pd.concat([df, temp_df])

    logger.info('Data preprocessing...')
    df.lyrics = df.lyrics.apply(clean_up_text)
    df.lyrics = df.lyrics.apply(remove_stop_words)

    df_dataset = pd.DataFrame()
    df_dataset['vectors'] = df['lyrics'].apply(embed)
    df_dataset['artist'] = df['artist'].apply(encode_label)
    logger.info('Data loaded.')

    return df_dataset


def main():
    df_dataset = load_data()    
    net = Net(input_units=INPUT_UNITS, output_units=OUTPUT_UNITS).to(DEVICE)
    
    split = int(TRAIN_SIZE * len(df_dataset))
    train_dataset = MyDataset(df_dataset[:split])
    test_dataset = MyDataset(df_dataset[split:])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_results, test_results = training_loop(
        num_epochs=NUM_EPOCHS,
        model=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        accuracy_function=multi_class_accuracy,
        epoch_count=0
    )


if __name__ == '__main__':
    main()
