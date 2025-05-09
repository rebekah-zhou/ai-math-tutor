import re
import os

TOKEN_RE = re.compile(r"""
    (\\[A-Za-z]+)       |  # backslash + letters, e.g. \frac, \theta
    (\d+\.\d+|\d+)      |  # numbers, ints or decimals
    ([A-Za-z])          |  # single letters (variables)
    (\^|_)              |  # superscript or subscript marker
    ([{}\(\)\[\]])      |  # braces or parentheses or brackets
    (<=|>=|<|>|=|!=)    |  # two-char comparison ops
    ([+\-*/=,:;])       |  # single-char operators & punctuation
    (\\,)               |  # escaped comma in LaTeX, if present
    (.)                     # anything else (catch-all)
""", re.VERBOSE)

def tokenize(latex: str):
    raw = TOKEN_RE.findall(latex)
    # each match is a tuple of groups; pick the non-empty one
    tokens = [ next(tok for tok in group if tok) for group in raw ]
    tokens = [t for t in tokens if not t.isspace()]
    return tokens


from collections import Counter

counter = Counter()
with open("data/images/labels.csv") as f:
    next(f)  # skip header
    for line in f:
        split,id,label,_ = line.rstrip("\n").split(",",3)
        if split=="train":
            counter.update(tokenize(label))

# sort by frequency and assign IDs
vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
for tok, _ in counter.most_common():
    if tok not in vocab:
        vocab[tok] = len(vocab)

# inverse map if you like
inv_vocab = {i:t for t,i in vocab.items()}

import torch
from torch.nn.utils.rnn import pad_sequence

def encode_label(label_str):
    toks = tokenize(label_str)
    ids  = [vocab.get(t, vocab["<unk>"]) for t in toks]
    return torch.tensor([vocab["<sos>"]] + ids + [vocab["<eos>"]], dtype=torch.long)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = [encode_label(lbl) for lbl in labels]
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=vocab["<pad>"])
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return imgs, seqs_padded, lengths


import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 1) Define your Dataset (re-using your tokenize/vocab code above)
class MathWritingDataset(Dataset):
    def __init__(self, csv_path, splits, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.split.isin(splits)].reset_index(drop=True)
        self.transform = transform or T.Compose([
            T.Resize((128,128)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert('L')
        return self.transform(img), row.label

# 2) Instantiate dataset + DataLoader
train_ds = MathWritingDataset("data/images/labels.csv", ["train"])
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,   # your collate_fn from above
)

# 3) Pull one batch and inspect
batch_imgs, batch_seqs, batch_lens = next(iter(train_loader))
print("Images:",  batch_imgs.shape)   # should be [4, 1, 128, 128]
print("Sequences:", batch_seqs.shape) # [4, max_seq_len]
print("Lengths:", batch_lens)         # tensor of 4 lengths

# 4) Decode one sequence back to tokens to verify:
for i in range(4):
    seq_ids = batch_seqs[i][: batch_lens[i]].tolist()
    # strip <sos> and <eos>
    seq_ids = seq_ids[1:-1]
    print("Decoded tokens:", [inv_vocab[id] for id in seq_ids])

print(f"Built vocab with {len(vocab)} tokens")