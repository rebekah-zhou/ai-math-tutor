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
from collections import defaultdict
import heapq
import contextlib

from tokenize_labels_and_test_dataloader import (
    MathWritingDataset,
    collate_fn,
    vocab,
    inv_vocab,
)

# Silence warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


class BeamSearchNode:
    """Node in beam search."""
    def __init__(self, hiddens, prev_node, token_id, logp, length):
        self.h = hiddens[0]
        self.c = hiddens[1]
        self.prev_node = prev_node
        self.token_id = token_id
        self.logp = logp
        self.length = length
        
    def eval(self, alpha=1.0):
        """Evaluate node score."""
        # Length normalization to prevent bias towards shorter sequences
        # Dividing by length^alpha where alpha is typically 0.6-1.0
        return self.logp / (self.length ** alpha)
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.eval() < other.eval()


@torch.no_grad()
def beam_search_decode(encoder, decoder, imgs, vocab, device, beam_width=5, max_len=50):
    """Beam search decoding for better sequence generation."""
    encoder.eval()
    decoder.eval()
    batch_size = imgs.size(0)
    
    # Encode images
    feats = encoder(imgs)
    
    # Store final sequences
    final_sequences = []
    
    # Process each image individually
    for b in range(batch_size):
        # Get features for current image
        img_feats = feats[b:b+1]  # Keep batch dimension
        
        # Initialize with <sos> token
        token = torch.full((1, 1), vocab["<sos>"], dtype=torch.long, device=device)
        
        # Initialize LSTM hidden states
        h = decoder.h0.expand(2, 1, decoder.lstm.hidden_size).contiguous()
        c = decoder.c0.expand(2, 1, decoder.lstm.hidden_size).contiguous()
        
        # Initial node
        start_node = BeamSearchNode((h, c), None, vocab["<sos>"], 0, 1)
        
        # Priority queue for beam search (will keep top-scored candidates)
        nodes = []
        heapq.heappush(nodes, (-start_node.eval(), id(start_node), start_node))
        
        # Queue of completed sequences
        end_nodes = []
        
        # Start beam search
        step = 0
        while step < max_len:
            step += 1
            
            # Get all candidates from current beam
            curr_nodes = []
            for _ in range(len(nodes)):
                if nodes:
                    score, _, n = heapq.heappop(nodes)
                    curr_nodes.append(n)
            
            # No candidates left
            if not curr_nodes:
                break
                
            # Expand all current candidates
            for node in curr_nodes:
                # Stop if <eos> encountered
                if node.token_id == vocab["<eos>"] and node.prev_node is not None:
                    end_nodes.append(node)
                    continue
                
                # Generate token sequence so far
                tokens = [node.token_id]
                n = node.prev_node
                while n is not None and n.prev_node is not None:
                    tokens.append(n.token_id)
                    n = n.prev_node
                tokens = tokens[::-1]  # Reverse to get correct order
                
                current_token = torch.tensor([tokens[-1]], device=device).view(1, 1)
                
                # Get context vector using attention
                hidden_state = node.h[-1].unsqueeze(1)  # [1, 1, hidden_dim]
                context, _ = decoder.attention(img_feats, hidden_state)
                context = context.squeeze(1)  # [1, hidden_dim]
                
                # Get embedding
                current_emb = decoder.embedding(current_token).squeeze(1)  # [1, embed_dim]
                
                # Combine embedding and context
                lstm_input = torch.cat([current_emb, context], dim=1).unsqueeze(1)
                
                # Forward through LSTM
                h, c = node.h, node.c
                output, (h_new, c_new) = decoder.lstm(lstm_input, (h, c))
                
                # Get logits
                output = output.squeeze(1)
                normed_output = decoder.layer_norm1(output)
                ffn_output = decoder.ffn(normed_output)
                output = decoder.layer_norm2(normed_output + ffn_output)
                output = decoder.dropout(output)
                logits = decoder.fc(output)
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=1)
                
                # Get top K candidates
                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                
                # Create new nodes
                for k in range(beam_width):
                    token_id = topk_indices[0, k].item()
                    log_prob = topk_log_probs[0, k].item()
                    
                    # Create new node
                    new_node = BeamSearchNode(
                        (h_new, c_new),
                        node,
                        token_id,
                        node.logp + log_prob,
                        node.length + 1
                    )
                    
                    # Add to queue
                    heapq.heappush(
                        nodes, 
                        (-new_node.eval(), id(new_node), new_node)
                    )
            
            # Limit beam width
            nodes = nodes[:beam_width]
            
            # If we have enough end nodes or no more candidates
            if len(end_nodes) >= beam_width or not nodes:
                break
        
        # If no complete sequences, take best partial sequences
        if not end_nodes and nodes:
            end_nodes = [heapq.heappop(nodes)[2] for _ in range(min(beam_width, len(nodes)))]
        
        # Get the best sequence
        if end_nodes:
            # Sort by score
            end_nodes.sort(key=lambda n: -n.eval())
            best_node = end_nodes[0]
            
            # Backtrack to get the sequence
            tokens = []
            node = best_node
            while node is not None:
                if node.token_id != vocab["<sos>"]:  # Skip <sos>
                    tokens.append(node.token_id)
                node = node.prev_node
            
            tokens = tokens[::-1]  # Reverse to get correct order
            if tokens and tokens[-1] == vocab["<eos>"]:
                tokens = tokens[:-1]  # Remove <eos>
        else:
            tokens = []
        
        final_sequences.append(tokens)
    
    return final_sequences


@torch.no_grad()
def validate(encoder, decoder, valid_loader, vocab, device, use_beam_search=True):
    encoder.eval()
    decoder.eval()
    total_tokens = 0
    correct_tokens = 0
    
    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        
        B = imgs.size(0)
        
        if use_beam_search:
            # Use beam search decoding
            pred_seqs = beam_search_decode(encoder, decoder, imgs, vocab, device)
            
            # Compare with ground truth
            for i in range(B):
                p = pred_seqs[i]  # Already processed (no <sos>/<eos>)
                t = seqs[i, 1:].tolist()  # Remove <sos>
                
                # Only compare up to <eos> or end
                if vocab["<eos>"] in t:
                    t = t[:t.index(vocab["<eos>"])]
                
                # Compare each token
                min_len = min(len(p), len(t))
                total_tokens += min_len
                correct_tokens += sum(1 for j in range(min_len) if p[j] == t[j])
        else:
            # Use greedy decoding
            feats = encoder(imgs)
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


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        """Register EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


@contextlib.contextmanager
def amp_autocast(enabled=True):
    """Context manager for mixed precision training."""
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


# ─── 1) Hyperparameters ────────────────────────────────────
BATCH_SIZE = 32
LR = 5e-4  # Slightly lower learning rate for better stability
NUM_EPOCHS = 50
EMBED_DIM = 256
HIDDEN_DIM = 512  # Increased hidden dimension for more capacity
VOCAB_SIZE = len(vocab)
IMG_CHANNELS = 1
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
PATIENCE = 5  # Increased patience for more reliable early stopping
PRINT_FREQ = 10
best_val = 0.0
epochs_no_improve = 0
WEIGHT_DECAY = 0.01  # Reduced weight decay for better convergence
DROPOUT_RATE = 0.1  # Lower dropout rate to prevent underfitting early
LABEL_SMOOTHING = 0.05  # Lower label smoothing to prevent too much uncertainty
USE_EMA = True  # Use Exponential Moving Average for more stable models
USE_AMP = torch.cuda.is_available()  # Use mixed precision when possible
USE_BEAM_SEARCH = True  # Use beam search for validation


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
    train_ds = MathWritingDataset("data/images/labels.csv", ["train"], augment=True)
    
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

    valid_ds = MathWritingDataset("data/images/labels.csv", ["valid"], augment=False)
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
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 256 -> 128
            )
            
            # Deeper residual blocks
            self.block1 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 128 -> 64
            )
            
            self.block2 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 64 -> 32
            )
            
            self.block3 = nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 32 -> 16
            )
            
            # Spatial attention module
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(512, 1, 1),
                nn.Sigmoid()
            )
            
            # Final convolution
            self.conv_final = nn.Sequential(
                nn.Conv2d(512, out_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            
            # Global average pooling with learned weights
            self.gap_weights = nn.Parameter(torch.ones(1, out_dim, 1, 1))
            self.gap = nn.AdaptiveAvgPool2d((8, 8))  # Output 8x8 feature map
            
            # Skip connections
            self.skip1 = nn.Sequential(
                nn.Conv2d(64, 128, 1, stride=1),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2)
            )
            self.skip2 = nn.Sequential(
                nn.Conv2d(128, 256, 1, stride=1),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2)
            )
            self.skip3 = nn.Sequential(
                nn.Conv2d(256, 512, 1, stride=1),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2)
            )
            
            # Dropout for regularization
            self.dropout = nn.Dropout2d(DROPOUT_RATE)
            
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
            # Initial convolutional layers
            x1 = self.conv1(x)
            x1 = self.dropout(x1)  # Apply dropout for regularization
            
            # First residual block with skip connection
            identity = self.skip1(x1)
            x2 = self.block1(x1)
            x2 = x2 + identity
            x2 = self.dropout(x2)
            
            # Second residual block with skip connection
            identity = self.skip2(x2)
            x3 = self.block2(x2)
            x3 = x3 + identity
            x3 = self.dropout(x3)
            
            # Third residual block with skip connection
            identity = self.skip3(x3)
            x4 = self.block3(x3)
            x4 = x4 + identity
            
            # Apply spatial attention
            att = self.spatial_attention(x4)
            x4 = x4 * att
            
            # Final processing
            x5 = self.conv_final(x4)
            x5 = self.gap(x5)
            x5 = x5 * self.gap_weights
            
            # Reshape for decoder: [batch, channels, height, width] -> [batch, seq_len, features]
            B, C, H, W = x5.shape
            return x5.view(B, C, H*W).permute(0, 2, 1)  # → [B, L, C]

    # Position Encoding for transformer-style attention
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super(PositionalEncoding, self).__init__()
            
            # Create position encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            # Register as buffer (not a parameter, but part of the module)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            # x is [batch, seq_len, d_model]
            return x + self.pe[:, :x.size(1), :]

    # More efficient attention with matrix operations
    class MultiHeadAttention(nn.Module):
        def __init__(self, feat_dim, hidden_dim, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
            
            self.feat_proj = nn.Linear(feat_dim, hidden_dim)
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            
            self.attn_dropout = nn.Dropout(DROPOUT_RATE)
            self.output_dropout = nn.Dropout(DROPOUT_RATE)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
        def forward(self, feats, hidden_state):
            B = feats.size(0)
            
            # Project features and hidden state
            feats = self.feat_proj(feats)
            query = (
                self.query_proj(hidden_state)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)  # [B, num_heads, query_len, head_dim]
            )
            key = (
                self.key_proj(feats)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)  # [B, num_heads, key_len, head_dim]
            )
            value = (
                self.value_proj(feats)
                .view(B, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)  # [B, num_heads, value_len, head_dim]
            )
            
            # Scaled dot-product attention with memory-efficient computation
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale  # [B, num_heads, query_len, key_len]
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            context = torch.matmul(attn_weights, value)  # [B, num_heads, query_len, head_dim]
            context = (
                context.transpose(1, 2)  # [B, query_len, num_heads, head_dim]
                .contiguous()
                .view(B, -1, self.num_heads * self.head_dim)  # [B, query_len, hidden_dim]
            )
            
            output = self.output_proj(context)
            output = self.output_dropout(output)
            
            # Add residual connection and layer normalization for stability
            if hidden_state.size(-1) == output.size(-1):
                output = self.layer_norm(output + hidden_state)
            else:
                output = self.layer_norm(output)
            
            return output, attn_weights

    class SeqDecoder(nn.Module):
        def __init__(self, feat_dim, embed_dim, hidden_dim, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = PositionalEncoding(embed_dim)
            self.attention = MultiHeadAttention(feat_dim, hidden_dim, num_heads=8)
            
            # Deeper LSTM with gradient clipping built in
            self.lstm = nn.LSTM(
                embed_dim + hidden_dim,  # Concatenated embedding and context
                hidden_dim,
                num_layers=2,
                dropout=DROPOUT_RATE,
                batch_first=True
            )
            
            # Feed-forward network after LSTM
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            # Output layers with regularization
            self.dropout = nn.Dropout(DROPOUT_RATE)
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, vocab_size)
            
            # Initialize weights
            self._init_weights()
            
            # Initial states
            self.h0 = nn.Parameter(torch.zeros(2, 1, hidden_dim))
            self.c0 = nn.Parameter(torch.zeros(2, 1, hidden_dim))
        
        def _init_weights(self):
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
            
            # Initialize LSTM weights with orthogonal initialization
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                
            # Initialize FFN weights
            for m in self.ffn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, feats, tgt_seq):
            B, seq_len = tgt_seq.size()
            
            # Get embeddings with positional encoding
            embedded = self.embedding(tgt_seq)  # [B, seq_len, embed_dim]
            embedded = self.pos_encoding(embedded)  # Add positional encoding
            embedded = self.dropout(embedded)
            
            # Initialize hidden states
            h = self.h0.expand(2, B, self.lstm.hidden_size).contiguous()
            c = self.c0.expand(2, B, self.lstm.hidden_size).contiguous()
            
            outputs = []
            
            for t in range(seq_len):
                current_emb = embedded[:, t, :]  # [B, embed_dim]
                
                # Use hidden state for attention query
                query = h[-1].unsqueeze(1)  # [B, 1, hidden_dim]
                
                # Get context vector using multi-head attention
                context, _ = self.attention(feats, query)  # [B, 1, hidden_dim]
                context = context.squeeze(1)  # [B, hidden_dim]
                
                # Combine embedding and context
                lstm_input = torch.cat([current_emb, context], dim=1).unsqueeze(1)  # [B, 1, embed_dim+hidden_dim]
                
                # Process through LSTM
                output, (h, c) = self.lstm(lstm_input, (h, c))  # output: [B, 1, hidden_dim]
                output = output.squeeze(1)  # [B, hidden_dim]
                
                # Apply residual connection and layer normalization
                normed_output = self.layer_norm1(output)
                
                # Apply feed-forward network with residual connection
                ffn_output = self.ffn(normed_output)
                output = self.layer_norm2(normed_output + ffn_output)
                
                # Apply dropout
                output = self.dropout(output)
                
                # Generate logits
                logits = self.fc(output)  # [B, vocab_size]
                outputs.append(logits.unsqueeze(1))  # Append [B, 1, vocab_size]
            
            return torch.cat(outputs, dim=1)  # [B, seq_len, vocab_size]

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
        label_smoothing=LABEL_SMOOTHING
    )
    
    # Optimizer with weight decay and layer-wise learning rates
    parameter_groups = [
        {'params': encoder.parameters(), 'lr': LR},
        {'params': decoder.parameters(), 'lr': LR * 2.0}  # Higher learning rate for decoder
    ]
    
    optimizer = optim.AdamW(
        parameter_groups,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),  # Increased beta2 for more stable updates
        eps=1e-8
    )

    # Learning rate scheduler with warmup and cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR, LR * 2.0],  # Different max LRs for encoder/decoder
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Shorter warm up (10% of training)
        div_factor=10.0,  # Initial LR is LR/10
        final_div_factor=100.0,  # Final LR is LR/100
        anneal_strategy='cos'  # Cosine annealing
    )

    # ─── 5) Training Loop ────────────────────────────────────
    global best_val, epochs_no_improve
    
    # Create EMA if enabled
    if USE_EMA:
        encoder_ema = EMA(encoder, decay=0.9998)
        decoder_ema = EMA(decoder, decay=0.9998)
        encoder_ema.register()
        decoder_ema.register()
    
    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
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
                
            # Use mixed precision for forward pass if enabled
            with amp_autocast(enabled=USE_AMP):
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
            if USE_AMP:
                # AMP: scale loss, backward, unscale, clip, step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard: backward, clip, step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Step scheduler
            scheduler.step()
            
            # Update EMA parameters
            if USE_EMA:
                encoder_ema.update()
                decoder_ema.update()
            
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
        
        # Validation phase with EMA model if enabled
        if USE_EMA:
            # Apply EMA parameters for validation
            encoder_ema.apply_shadow()
            decoder_ema.apply_shadow()
            
        # Validate with beam search if enabled
        val_acc = validate(encoder, decoder, valid_loader, vocab, DEVICE, 
                         use_beam_search=USE_BEAM_SEARCH)
        
        # Restore original parameters if using EMA
        if USE_EMA:
            encoder_ema.restore()
            decoder_ema.restore()
        
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
            if USE_EMA:
                # Apply EMA parameters for saving
                encoder_ema.apply_shadow()
                decoder_ema.apply_shadow()
                
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "vocab": vocab,
                "inv_vocab": inv_vocab,
                "valid_acc": val_acc,
                "train_loss": avg_train,
                "hyperparams": {
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM
                }
            }, "best_checkpoint.pth")
            
            if USE_EMA:
                # Restore original parameters after saving
                encoder_ema.restore()
                decoder_ema.restore()
                
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
    
    # Save a sample of generated sequences for inspection
    if os.environ.get('VERBOSE', False):
        print("\nGenerating sample predictions...")
        # Load best model
        checkpoint = torch.load("best_checkpoint.pth", map_location=DEVICE)
        encoder.load_state_dict(checkpoint["encoder_state"])
        decoder.load_state_dict(checkpoint["decoder_state"])
        
        # Generate sample predictions
        sample_predictions(encoder, decoder, valid_loader, vocab, inv_vocab, DEVICE)


def sample_predictions(encoder, decoder, valid_loader, vocab, inv_vocab, device, n_samples=5):
    """Generate and display sample predictions."""
    encoder.eval()
    decoder.eval()
    
    # Get a batch
    for imgs, seqs, lengths in valid_loader:
        imgs, seqs = imgs.to(device), seqs.to(device)
        break
    
    # Limit to n_samples
    imgs = imgs[:n_samples]
    seqs = seqs[:n_samples]
    lengths = lengths[:n_samples]
    
    # Generate predictions with beam search
    with torch.no_grad():
        pred_seqs = beam_search_decode(encoder, decoder, imgs, vocab, device)
    
    # Display results
    print("\nSample Predictions:")
    print("=" * 50)
    
    for i in range(len(imgs)):
        # Get ground truth sequence
        true_seq = []
        for t in seqs[i, 1:lengths[i]-1].tolist():  # Remove <sos> and <eos>
            true_seq.append(inv_vocab[t])
        
        # Get predicted sequence
        pred_seq = []
        for t in pred_seqs[i]:
            pred_seq.append(inv_vocab[t])
        
        # Print comparison
        print(f"Example {i+1}:")
        print(f"  Truth: {' '.join(true_seq)}")
        print(f"  Pred:  {' '.join(pred_seq)}")
        print("-" * 50)


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