import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from tokenize_labels_and_test_dataloader import (
    MathWritingDataset,
    collate_fn,
    vocab,
)

@torch.no_grad()
def validate(encoder, decoder, valid_loader, vocab, device):
    encoder.eval()
    decoder.eval()
    total, correct = 0, 0

    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        feats = encoder(imgs)

        B = imgs.size(0)
        # start every sequence with <sos>
        preds = torch.full((B, 1), vocab["<sos>"], dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        max_steps = seqs.size(1) - 1  # at most generate this many tokens
        for _ in range(max_steps):
            logits   = decoder(feats, preds)               # [B, seq_len, V]
            next_tok = logits[:, -1].argmax(dim=-1).unsqueeze(1)  # [B,1]
            preds    = torch.cat([preds, next_tok], dim=1)       # extend each seq

            # mark sequences that just produced <eos>
            eos_hits = next_tok.squeeze(1) == vocab["<eos>"]
            finished = finished | eos_hits

            # if every sequence has emitted <eos>, stop early
            if finished.all():
                break

        # now compare each pred vs true seq
        for i in range(B):
            p = preds[i].tolist()
            t = seqs[i].tolist()

            # strip <sos> and everything after the first <eos>
            if vocab["<eos>"] in p:
                p = p[1 : p.index(vocab["<eos>"])]
            else:
                p = p[1:]

            if vocab["<eos>"] in t:
                t = t[1 : t.index(vocab["<eos>"])]
            else:
                t = t[1:]

            total += 1
            if p == t:
                correct += 1

    return correct / total

@torch.no_grad()
def token_accuracy(decoder, encoder, valid_loader, vocab, device):
    correct, total = 0, 0
    decoder.eval(); encoder.eval()
    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        feats = encoder(imgs)

        # teacher‐forced logits on the true seqs
        logits = decoder(feats, seqs[:, :-1])  # predict all next tokens
        preds  = logits.argmax(-1)              # [B, T]

        # compare preds vs seqs[:,1:] mask out padding
        mask   = seqs[:,1:] != vocab["<pad>"]
        correct += ((preds == seqs[:,1:]) & mask).sum().item()
        total   += mask.sum().item()

    return correct/total



# ─── 1) Hyperparameters ────────────────────────────────────
BATCH_SIZE    = 16
LR            = 1e-3
NUM_EPOCHS    = 50
EMBED_DIM     = 256
HIDDEN_DIM    = 256
VOCAB_SIZE    = len(vocab)
IMG_CHANNELS  = 1
IMG_SIZE      = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE   = 5   # stop if no improvement after this many epochs
best_val   = 0.0
epochs_no_improve = 0

# ─── 2) DataLoaders ───────────────────────────────────────
train_ds = train_ds = MathWritingDataset("data/images/labels.csv", ["train"])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)
valid_ds = MathWritingDataset("data/images/labels.csv", ["valid"])
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn)

# ─── 3) Model Definition ──────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, 3, stride=2, padding=1),  # 64×64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # 32×32
            nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1),       # 16×16
            nn.ReLU(),
        )
    def forward(self, x):
        # x: [B,1,128,128] → [B, C, H', W']
        feats = self.cnn(x)
        B,C,H,W = feats.shape
        # flatten spatial dims → sequence length L=H*W
        return feats.view(B, C, H*W).permute(0,2,1)  # → [B, L, C]

class SeqDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim + EMBED_DIM, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, feats, tgt_seq):
        """
        feats: [B, L, EMBED_DIM]  (encoder outputs)
        tgt_seq: [B, T]           (input token IDs, including <sos> but excluding last <eos>)
        """
        emb = self.embedding(tgt_seq)                        # [B, T, embed_dim]
        # repeat a global summary vector (or mean-pool feats) at each time-step:
        ctx = feats.mean(dim=1, keepdim=True).expand(-1, emb.size(1), -1)
        inp = torch.cat([emb, ctx], dim=-1)                  # [B, T, embed+emb_dim]
        out, _ = self.lstm(inp)                              # [B, T, hidden_dim]
        logits = self.fc(out)                                # [B, T, vocab_size]
        return logits

# assemble
encoder = CNNEncoder(out_dim=EMBED_DIM).to(DEVICE)
decoder = SeqDecoder(EMBED_DIM, HIDDEN_DIM, VOCAB_SIZE).to(DEVICE)

# ─── 4) Training Setup ────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)
# ─── 5) Training Loop ────────────────────────────────────
for epoch in range(1, NUM_EPOCHS+1):
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for imgs, seqs, lengths in train_loader:
        imgs, seqs = imgs.to(DEVICE), seqs.to(DEVICE)
        optimizer.zero_grad()

        feats = encoder(imgs)
        input_seq  = seqs[:, :-1]
        target_seq = seqs[:, 1:]
        logits = decoder(feats, input_seq)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train = running_loss / len(train_loader)
    val_acc   = validate(encoder, decoder, valid_loader, vocab, DEVICE)

    print(f"Epoch {epoch:02d}: train_loss={avg_train:.4f}, valid_acc={val_acc:.2%}")

    # early-stopping logic
    if val_acc > best_val:
        best_val = val_acc
        epochs_no_improve = 0
        # save checkpoint
        torch.save({
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": vocab,
            "valid_acc": val_acc,
        }, "best_checkpoint.pth")
        print(f"  ↳ New best! checkpoint saved.")
    else:
        epochs_no_improve += 1
        print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        print(f"Stopping early after {epoch} epochs (no improvement in {PATIENCE}).")
        break

tok_acc = token_accuracy(decoder, encoder, valid_loader, vocab, DEVICE)
print("Training complete. Best valid_acc:", best_val)
print(f"Valid token‐accuracy: {tok_acc:.2%}")