import os
import sys
import time
import warnings
from multiprocessing import freeze_support
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tokenize_labels_and_test_dataloader import (
    MathWritingDataset,
    collate_fn,
    vocab,
)

# Silence warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


@torch.no_grad()
def validate(encoder, decoder, valid_loader, vocab, device):
    encoder.eval()
    decoder.eval()
    total_tokens = 0
    correct_tokens = 0
    
    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        feats = encoder(imgs)
        
        B = imgs.size(0)
        preds = torch.full(
            (B, 1), vocab["<sos>"], 
            dtype=torch.long, device=device
        )
        
        # Generate tokens
        max_steps = seqs.size(1) - 1
        for _ in range(max_steps):
            logits = decoder(feats, preds)
            next_tok = logits[:, -1].argmax(dim=-1).unsqueeze(1)
            preds = torch.cat([preds, next_tok], dim=1)
        
        # Compare tokens individually
        for i in range(B):
            p = preds[i, 1:].tolist()  # Remove <sos>
            t = seqs[i, 1:].tolist()   # Remove <sos>
            
            # Only compare up to <eos> or end
            if vocab["<eos>"] in p:
                p = p[:p.index(vocab["<eos>"])]
            if vocab["<eos>"] in t:
                t = t[:t.index(vocab["<eos>"])]
                
            # Compare each token
            min_len = min(len(p), len(t))
            total_tokens += min_len
            correct_tokens += sum(1 for j in range(min_len) if p[j] == t[j])
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def token_accuracy(decoder, encoder, valid_loader, vocab, device):
    correct, total = 0, 0
    decoder.eval()
    encoder.eval()
    
    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        feats = encoder(imgs)

        # teacher‐forced logits on the true seqs
        logits = decoder(feats, seqs[:, :-1])  # predict all next tokens
        preds = logits.argmax(-1)  # [B, T]

        # compare preds vs seqs[:,1:] mask out padding
        mask = seqs[:, 1:] != vocab["<pad>"]
        correct += ((preds == seqs[:, 1:]) & mask).sum().item()
        total += mask.sum().item()

    return correct/total


# ─── 1) Hyperparameters ────────────────────────────────────
# Hyperparameters
BATCH_SIZE = 32  # Balanced for MPS
LR = 1e-3        # Higher initial learning rate
NUM_EPOCHS = 50
EMBED_DIM = 256
HIDDEN_DIM = 256
VOCAB_SIZE = len(vocab)
IMG_CHANNELS = 1
IMG_SIZE = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PATIENCE = 3     # Reduced patience for faster early stopping
PRINT_FREQ = 10  # Print frequency (every N batches)
best_val = 0.0
epochs_no_improve = 0


def main():
    # Minimize console output
    if not os.environ.get('VERBOSE', False):
        # Redirect stdout to a null device
        original_stdout = sys.stdout

        class MinimalOutput:
            def write(self, x):
                # Only write if it contains important training info
                if any(s in x for s in [
                    'Epoch', 'valid_acc', 'New best', 'Training complete'
                ]):
                    original_stdout.write(x)

            def flush(self):
                original_stdout.flush()
        
        # Only use minimal output after initial setup
        print(f"Using device: {DEVICE}")
        print(f"Training with batch size: {BATCH_SIZE}, learning rate: {LR}")
        print("Starting training (warnings and dataset info suppressed)...")
        sys.stdout = MinimalOutput()
    else:
        print(f"Using device: {DEVICE}")
        
    # Enable metal performance optimization for Apple Silicon
    if DEVICE.type == 'mps':
        # MPS-specific optimizations
        torch.mps.empty_cache()  # Clear GPU memory before starting

    # ─── 2) DataLoaders with Apple-optimized settings ───────────────
    train_ds = MathWritingDataset("data/images/labels.csv", ["train"])
    
    # Apple Silicon optimized DataLoader settings
    # Fewer workers and no persistent workers to reduce memory usage
    num_workers = 2 if DEVICE.type == 'mps' else 4
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,      # ← faster host→device copies
        prefetch_factor=2,    # ← each worker preloads 2 batches
        collate_fn=collate_fn,
        # No persistent workers for MPS to avoid memory leaks
        persistent_workers=False,
    )

    valid_ds = MathWritingDataset("data/images/labels.csv", ["valid"])
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers // 2,  # Fewer workers for validation
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    # ─── 3) Model Definition ──────────────────────────────────
    class CNNEncoder(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            
            # Initial convolution with smaller kernel for fine details
            self.conv1 = nn.Sequential(
                nn.Conv2d(IMG_CHANNELS, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            # Deeper residual blocks
            self.block1 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            
            # Spatial attention module
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
            
            # Final convolution
            self.conv_final = nn.Sequential(
                nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            
            # Global average pooling with learned weights
            self.gap_weights = nn.Parameter(torch.ones(1, out_dim, 1, 1))
            self.gap = nn.AdaptiveAvgPool2d((8, 8))
            
            # Skip connections
            self.skip1 = nn.Sequential(
                nn.Conv2d(32, 64, 1, stride=2),
                nn.BatchNorm2d(64)
            )
            self.skip2 = nn.Sequential(
                nn.Conv2d(64, 128, 1, stride=2),
                nn.BatchNorm2d(128)
            )
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(
                        m.weight, 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    # Xavier/Glorot for linear layers
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            # Initial conv with residual
            x1 = self.conv1(x)
            
            # First residual block with skip connection
            identity = self.skip1(x1)
            x2 = self.block1(x1)
            x2 = x2 + identity
            
            # Second residual block with skip connection
            identity = self.skip2(x2)
            x3 = self.block2(x2)
            x3 = x3 + identity
            
            # Apply spatial attention
            att = self.spatial_attention(x3)
            x3 = x3 * att
            
            # Final processing
            x4 = self.conv_final(x3)
            x4 = self.gap(x4)
            x4 = x4 * self.gap_weights
            
            # Reshape for decoder
            B, C, H, W = x4.shape
            return x4.view(B, C, H*W).permute(0, 2, 1)  # → [B, L, C]

    # More efficient attention with matrix operations
    class MultiHeadAttention(nn.Module):
        def __init__(self, feat_dim, hidden_dim, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            
            self.feat_proj = nn.Linear(feat_dim, hidden_dim)
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, feats, hidden_state):
            B = feats.size(0)
            
            # Project features and hidden state
            feats = self.feat_proj(feats)
            query = (
                self.query_proj(hidden_state)
                .view(B, 1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            key = (
                self.key_proj(feats)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value = (
                self.value_proj(feats)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            
            # Scaled dot-product attention with memory-efficient computation
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, value)
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(B, 1, -1)
            )
            context = self.output_proj(context)
            
            return context.squeeze(1), attn_weights

    class SeqDecoder(nn.Module):
        def __init__(self, feat_dim, embed_dim, hidden_dim, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.attention = MultiHeadAttention(feat_dim, hidden_dim)
            
            # More efficient LSTM setup
            self.lstm = nn.LSTM(
                embed_dim + feat_dim,
                hidden_dim,
                num_layers=2,
                dropout=0.2,
                batch_first=True
            )
            
            # Output layers with regularization
            self.dropout = nn.Dropout(0.2)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, vocab_size)
            
            # Initialize weights
            self._init_weights()
            
            # Initial states
            self.h0 = nn.Parameter(
                torch.zeros(2, 1, hidden_dim),
                requires_grad=False
            )
            self.c0 = nn.Parameter(
                torch.zeros(2, 1, hidden_dim),
                requires_grad=False
            )
        
        def _init_weights(self):
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
            
            # Initialize LSTM weights
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        def forward(self, feats, tgt_seq):
            B, L, feat_dim = feats.size()
            seq_len = tgt_seq.size(1)
            
            # Get embeddings with dropout
            embedded = self.dropout(self.embedding(tgt_seq))
            
            # Initialize hidden states
            h = self.h0.expand(2, B, self.lstm.hidden_size).contiguous()
            c = self.c0.expand(2, B, self.lstm.hidden_size).contiguous()
            
            outputs = []
            
            for t in range(seq_len):
                current_emb = embedded[:, t, :]
                
                # Get context vector using multi-head attention
                context, _ = self.attention(feats, h[-1])
                
                # Combine embedding and context
                lstm_input = torch.cat(
                    [current_emb, context], dim=1
                ).unsqueeze(1)
                
                # Process through LSTM
                output, (h, c) = self.lstm(lstm_input, (h, c))
                
                # Apply layer norm and dropout
                output = self.layer_norm(output.squeeze(1))
                output = self.dropout(output)
                
                # Generate logits
                logits = self.fc(output)
                outputs.append(logits.unsqueeze(1))
            
            return torch.cat(outputs, dim=1)

    # ─── Create models ────────────────────────────────────────
    encoder = CNNEncoder(out_dim=EMBED_DIM).to(DEVICE)
    decoder = SeqDecoder(
        feat_dim=EMBED_DIM,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE
    ).to(DEVICE)

    # ─── 4) Training Setup ────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab["<pad>"],
        label_smoothing=0.1
    )
    
    # Optimizer with weight decay and layer-wise learning rates
    parameter_groups = [
        {'params': encoder.parameters(), 'lr': LR},
        {'params': decoder.parameters(), 'lr': LR * 1.5}  # Higher LR for decoder
    ]
    
    optimizer = optim.AdamW(
        parameter_groups,
        lr=LR,
        weight_decay=0.05,  # Increased weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler with warmup and cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR, LR * 1.5],  # Different max LRs for encoder/decoder
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Longer warm up (20% of training)
        div_factor=25.0,  # Initial LR is LR/25
        final_div_factor=1000.0,  # Final LR is LR/1000
        anneal_strategy='cos'  # Cosine annealing
    )

    # ─── 5) Training Loop ────────────────────────────────────
    global best_val, epochs_no_improve
    
    # Pre-fetch first batch for benchmark
    data_iter = iter(train_loader)
    try:
        next(data_iter)  # Just prefetch, don't store
    except StopIteration:
        pass
    
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()
        print(f'Epoch {epoch:02d} starting...')
        
        # Training phase
        encoder.train()
        decoder.train()
        running_loss = 0.0
        
        # Determine whether to use tqdm based on environment variable
        use_tqdm = os.environ.get('VERBOSE', False)
        if use_tqdm:
            train_iter = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        else:
            train_iter = train_loader
        
        # Periodically clear MPS cache to prevent memory buildup
        if DEVICE.type == 'mps' and epoch % 2 == 0:
            torch.mps.empty_cache()
            
        for batch_idx, (imgs, seqs, lengths) in enumerate(train_iter):
            # Transfer to device with non-blocking for better performance
            imgs = imgs.to(DEVICE, non_blocking=True)
            seqs = seqs.to(DEVICE, non_blocking=True)
            
            # Zero gradients - more efficient way
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
            # Forward pass
            feats = encoder(imgs)
            input_seq = seqs[:, :-1]   # Remove last token (<eos> or pad)
            target_seq = seqs[:, 1:]   # Remove first token (<sos>)
            
            # Compute outputs and loss
            logits = decoder(feats, input_seq)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1)
            )
            
            # Backward pass and optimization with MPS-friendly approach
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            # Step optimizer and scheduler
            optimizer.step()
            scheduler.step()
            
            # Track statistics
            curr_loss = loss.item()
            running_loss += curr_loss
            
            # Update progress bar if using tqdm
            if use_tqdm:
                train_iter.set_postfix({"loss": f"{curr_loss:.4f}"})
            elif batch_idx % PRINT_FREQ == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {curr_loss:.4f}"
                )
        
        # Calculate epoch statistics
        avg_train = running_loss / len(train_loader)
        
        # Validation phase
        val_acc = validate(encoder, decoder, valid_loader, vocab, DEVICE)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch:02d}: train_loss={avg_train:.4f}, "
            f"valid_acc={val_acc:.2%}, time={epoch_time:.1f}s"
        )
        
        # Early stopping logic with more detailed tracking
        if val_acc > best_val:
            best_val = val_acc
            epochs_no_improve = 0
            # Save checkpoint with more information
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "vocab": vocab,
                "valid_acc": val_acc,
                "train_loss": avg_train,
                "hyperparams": {
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM
                }
            }, "best_checkpoint.pth")
            print("  ↳ New best! Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(
                f"  ↳ No improvement for {epochs_no_improve} epoch(s). "
                f"Best: {best_val:.2%}"
            )

        if epochs_no_improve >= PATIENCE:
            print(
                f"Stopping early after {epoch} epochs. "
                f"Best validation accuracy: {best_val:.2%}"
            )
            break

    # Final validation with token accuracy
    tok_acc = token_accuracy(decoder, encoder, valid_loader, vocab, DEVICE)
    print("Training complete. Best valid_acc:", best_val)
    print(f"Valid token‐accuracy: {tok_acc:.2%}")


if __name__ == "__main__":
    # On macOS it can help to call freeze_support()
    freeze_support()
    
    # Add command-line options for easier control
    import argparse
    parser = argparse.ArgumentParser(
        description="Train math writing recognition model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=LR, help="Learning rate"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--fp16", action="store_true", 
        help="Use float16 precision (MPS only)"
    )
    parser.add_argument(
        "--help-mac", action="store_true", help="Show Mac optimization tips"
    )
    args = parser.parse_args()
    
    # Update settings from command line
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.lr:
        LR = args.lr
    if args.verbose:
        os.environ['VERBOSE'] = "1"
    if args.fp16:
        os.environ['USE_FP16'] = "1"
        
    # Display Mac optimization tips if requested
    if args.help_mac:
        print("\nApple Silicon Optimization Tips:")
        print(
            "1. Install libjpeg to fix warnings: "
            "conda install -c conda-forge libjpeg-turbo"
        )
        print("2. Try smaller batch sizes (16-32) if you get memory errors")
        print("3. Use --fp16 flag for potentially faster training")
        print(
            "4. Empty MPS cache if experiencing memory issues: "
            "python -c 'import torch; torch.mps.empty_cache()'"
        )
        print("5. Restart your Python environment if MPS performance degrades")
        print("\nExample command: python train.py --batch_size 32 --fp16")
        exit(0)
    
    # Pre-training cleanup for MPS
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Time the full training process
    total_start = time.time()
    main()
    total_time = time.time() - total_start
    
    # Print final timing information
    print("\nTraining complete!")
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")