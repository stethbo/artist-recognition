import torch
import argparse

from config import DEVICE
from model_level_1 import Net
from embeddings import clean_up_text, embed
from level1 import decode_label, INPUT_UNITS, OUTPUT_UNITS

def predict(model, text, device=DEVICE):
    X = embed(clean_up_text(text))
    model.eval()
    with torch.no_grad():
        X = X.unsqueeze(dim=0).to(device).float()
        y_pred = model(X)
        y_pred = torch.zeros_like(y_pred).scatter_(
            1, torch.argmax(y_pred, dim=1).unsqueeze(dim=1), 1)
        return decode_label(y_pred.squeeze(dim=0).to(int))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--txt_path', type=str, required=True,
                        help='Path to textfile with tet to predict on')
    args = parser.parse_args()

    model = Net(input_units=INPUT_UNITS, output_units=OUTPUT_UNITS).to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(args.model))
    text = open(args.txt_path, 'r').read()
    print(predict(model, args.txt_path))
