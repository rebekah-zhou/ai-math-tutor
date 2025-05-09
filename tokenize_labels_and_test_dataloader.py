import re
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import random
import os

# Common LaTeX math environments and commands from referenced documents
MATH_ENVIRONMENTS = [
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
    "array", "align", "aligned", "gather", "gathered"
]

# Simplified lines to avoid line length issues
MATH_COMMANDS = [
    # Greek letters
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", 
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", 
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", 
    "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", 
    "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    # Operators and functions
    "frac", "sum", "prod", "lim", "int", "oint", "infty", "nabla", "partial",
    "times", "div", "cdot", "pm", "mp", "leq", "geq", "equiv", "sim", "simeq",
    "approx", "neq", "propto", "in", "notin", "subset", "supset", "cup", "cap",
    "emptyset", "exists", "forall", "neg", "rightarrow", "leftarrow",
    "Rightarrow", "Leftarrow", "leftrightarrow", "Leftrightarrow",
    # Brackets and delimiters
    "left", "right", "lceil", "rceil", "lfloor", "rfloor", "langle", "rangle",
    "lvert", "rvert", "lVert", "rVert",
    # Accents and font styles
    "hat", "tilde", "bar", "vec", "overrightarrow", "dot", "ddot", "widehat", 
    "widetilde", "overline", "underline", "mathrm", "mathbf", "mathcal", 
    "mathsf", "mathtt", "mathit", "displaystyle", "textstyle", "scriptstyle",
    "scriptscriptstyle",
    # Spaces, dots, and other common commands
    "quad", "qquad", "hspace", "vspace", "ldots", "cdots", "vdots", "ddots",
    "text", "textrm", "textbf", "textit", "texttt", "textsf",
    "sqrt", "root", "binom", "pmod", "begin", "end", "fbox", "mbox"
]

# More comprehensive regex to handle LaTeX symbols properly
TOKEN_RE = re.compile(r"""
    (\\begin\{[a-zA-Z]+\})    |  # LaTeX environment start
    (\\end\{[a-zA-Z]+\})      |  # LaTeX environment end
    (\\[A-Za-z]+\{)           |  # Control sequences with opening brace
    (\\[A-Za-z]+)             |  # LaTeX commands
    (\d+\.\d+|\d+)            |  # numbers, ints or decimals
    ([A-Za-z][A-Za-z0-9]*)    |  # variables with multiple letters
    (\^|_)                    |  # superscript or subscript marker
    ([{}\(\)\[\]])            |  # braces, parentheses, brackets
    (<=|>=|<|>|=|!=|\+\+)     |  # multi-char operators
    ([+\-*/=,:;!|&%$@])       |  # single-char operators
    (\\,|\\;|\\:|\\!)         |  # LaTeX spaces
    (\\ )                     |  # escaped space
    (\\cdot|\\cdots|\\ldots)  |  # common dots commands
    (.)                          # anything else (catch-all)
""", re.VERBOSE)


def tokenize(latex: str):
    """Convert LaTeX string to tokens with improved handling of math environments."""
    # Clean input
    latex = latex.strip()
    
    # Find all tokens
    raw = TOKEN_RE.findall(latex)
    tokens = [next(tok for tok in group if tok) for group in raw]
    tokens = [t for t in tokens if not t.isspace()]
    
    # Process tokens with special handling for environments
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Handle environment start tags
        if token.startswith("\\begin{"):
            # Extract environment name
            env_name = token[7:-1]  # Remove \begin{ and }
            processed_tokens.append(token)
            i += 1
            
            # Special handling for matrix and similar environments
            if env_name in MATH_ENVIRONMENTS:
                # Continue until we find the matching \end
                while (i < len(tokens) and 
                       not tokens[i].startswith(f"\\end{{{env_name}")):
                    # Add content tokens, keeping structural tokens whole
                    processed_tokens.append(tokens[i])
                    i += 1
                
                # Add the end token if found
                if i < len(tokens):
                    processed_tokens.append(tokens[i])
                    i += 1
            else:
                # For non-matrix environments, continue normally
                continue
        else:
            processed_tokens.append(token)
            i += 1
    
    return processed_tokens


def build_vocab(csv_path, min_freq=1):
    """Build vocabulary from training data with minimum frequency threshold."""
    # Count token frequencies
    counter = Counter()
    train_labels = []
    
    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) >= 4:
                split, id, label, _ = parts
                if split == "train":
                    counter.update(tokenize(label))
                    train_labels.append(label)
    
    # Create vocabulary with special tokens
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    
    # Add tokens that appear at least min_freq times
    for tok, count in counter.most_common():
        if count >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    
    # Create inverse vocabulary
    inv_vocab = {i: t for t, i in vocab.items()}

    return vocab, inv_vocab, train_labels


# Build vocabulary if this module is imported
vocab_built = False
vocab, inv_vocab, train_labels = {}, {}, []


def get_vocab():
    """Get or build vocabulary."""
    global vocab, inv_vocab, train_labels, vocab_built
    if not vocab_built:
        vocab, inv_vocab, train_labels = build_vocab("data/images/labels.csv")
        vocab_built = True
    return vocab, inv_vocab, train_labels


# Initialize on import
vocab, inv_vocab, train_labels = get_vocab()


def encode_label(label_str):
    """Convert label string to token IDs."""
    toks = tokenize(label_str)
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
    return torch.tensor(
        [vocab["<sos>"]] + ids + [vocab["<eos>"]], 
        dtype=torch.long
    )


def collate_fn(batch):
    """Collate function for DataLoader."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = [encode_label(lbl) for lbl in labels]
    
    # Get sequence lengths for packing
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    
    # Pad sequences
    seqs_padded = pad_sequence(
        seqs, 
        batch_first=True, 
        padding_value=vocab["<pad>"]
    )
    
    return imgs, seqs_padded, lengths


class MathWritingDataset(Dataset):
    """Dataset for math writing recognition with augmentation."""
    def __init__(self, csv_path, splits, transform=None, augment=True):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.split.isin(splits)].reset_index(drop=True)
        self.augment = augment and 'train' in splits
        
        # Default transforms
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        
        # Augmentation transforms (used only for training)
        self.aug_transforms = T.Compose([
            T.RandomApply([
                T.RandomAffine(
                    degrees=2,  # Slight rotation
                    translate=(0.02, 0.02),  # Small translation
                    scale=(0.98, 1.02),  # Slight scaling
                    fill=255,  # Fill with white
                )
            ], p=0.5),
            T.RandomApply([
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),
        ])
        
        # For noise augmentation (applied after tensor conversion)
        self.tensor_aug = AddGaussianNoise(mean=0., std=0.01) if self.augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load image as PIL Image
            img = Image.open(row.image_path).convert('L')
            
            # Apply PIL-based augmentations to PIL Image
            if self.augment:
                img = self.aug_transforms(img)
                
            # Apply standard transformations (conversion to tensor)
            img = self.transform(img)
            
            # Apply tensor-based augmentations
            if self.augment and self.tensor_aug and torch.rand(1).item() < 0.2:
                img = self.tensor_aug(img)
            
            return img, row.label
        except Exception as e:
            print(f"Error loading {row.image_path}: {e}")
            # Return a blank image and empty label in case of error
            placeholder = torch.zeros(1, 256, 256)
            return placeholder, ""


class AddGaussianNoise:
    """Add Gaussian noise to tensor images."""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


if __name__ == "__main__":
    # Test tokenizer and dataset with original labels
    print("Testing tokenizer with example LaTeX expressions...")
    test_expressions = [
        r"\tilde{U}_{S}^{\dagger}",
        r"y(x)=(\begin{matrix}y_{1}(x)\\ y_{2}(x)\end{matrix})",
        r"D_{i}(g_{t})=\frac{d}{dt}g_{t}",
        r"B=b_{0}+b_{1}x+\cdot\cdot\cdot"
    ]
    
    for i, expr in enumerate(test_expressions):
        tokens = tokenize(expr)
        print(f"\nTest {i+1}:")
        print(f"  Original: {expr}")
        print(f"  Tokens: {tokens}")
    
    # Now test dataset loading
    print("\nTesting dataset loading...")
    train_ds = MathWritingDataset("data/images/labels.csv", ["train"])
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,  # Don't shuffle for testing
        collate_fn=collate_fn,
    )

    batch_imgs, batch_seqs, batch_lens = next(iter(train_loader))
    print("Images shape:", batch_imgs.shape)
    print("Sequences shape:", batch_seqs.shape)
    print("Lengths:", batch_lens)

    # Compare direct tokenization vs loaded data
    print("\nComparing tokenization methods:")
    for i in range(min(4, len(batch_seqs))):
        original_label = train_ds.df.iloc[i].label
        direct_tokens = tokenize(original_label)
        
        print(f"\nExample {i+1}:")
        print(f"  Original label: {original_label}")
        print(f"  Direct tokens: {direct_tokens}")
        
        # Show encoded/decoded tokens 
        seq_ids = batch_seqs[i][1:batch_lens[i]-1].tolist()  # Remove <sos>/<eos>
        decoded_tokens = [inv_vocab[id] for id in seq_ids]
        print(f"  Decoded from dataloader: {decoded_tokens}")
        
        # Debug: check if encoding directly matches
        direct_ids = encode_label(original_label)[1:-1].tolist()  # Remove <sos>/<eos>
        direct_decoded = [inv_vocab[id] for id in direct_ids]
        if direct_decoded != decoded_tokens:
            print("  ⚠️ MISMATCH between direct encoding and dataloader!")
            print(f"  Direct encoding: {direct_decoded}")

    # Print vocabulary statistics
    print("\nVocabulary statistics:")
    print(f"  Total size: {len(vocab)} tokens")
    print(f"  Special tokens: {[t for t in vocab if t.startswith('<')]}")
    
    # Check for tokens that might need special handling
    special_chars = [t for t, i in vocab.items() 
                    if len(t) == 1 and not t.isalnum() 
                    and not t.startswith('<')]
    if special_chars:
        print(f"  Special characters: {special_chars[:10]}")
        if len(special_chars) > 10:
            print("    ... and more")