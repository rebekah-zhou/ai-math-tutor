import torch
import pandas as pd
import os
import argparse
from tokenize_labels_and_test_dataloader import (
    MathWritingDataset,
    collate_fn,
    vocab,
)
from torch.utils.data import DataLoader

def load_model_from_checkpoint(checkpoint_path, device):
    """Load encoder and decoder models from a checkpoint file."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # We need to define model classes exactly as they were during training
    # This should match your model definitions from your training script
    class CNNEncoder(torch.nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, stride=2, padding=1),  
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),            
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),           
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, out_dim, 3, stride=2, padding=1),      
                torch.nn.BatchNorm2d(out_dim),
                torch.nn.ReLU(inplace=True),
            )
            
        def forward(self, x):
            feats = self.cnn(x)
            B, C, H, W = feats.shape
            return feats.view(B, C, H*W).permute(0, 2, 1)

    class Attention(torch.nn.Module):
        def __init__(self, feat_dim, hidden_dim):
            super().__init__()
            self.feat_proj = torch.nn.Linear(feat_dim, hidden_dim)
            self.hidden_proj = torch.nn.Linear(hidden_dim, hidden_dim)
            self.v = torch.nn.Linear(hidden_dim, 1)

        def forward(self, feats, hidden_state):
            feat_proj = self.feat_proj(feats)                  
            hidden_proj = self.hidden_proj(hidden_state)       
            energy = torch.tanh(feat_proj + hidden_proj.unsqueeze(1))
            attention_scores = self.v(energy)                        
            attention_weights = torch.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.transpose(1, 2), feats)
            return context.squeeze(1), attention_weights

    class SeqDecoder(torch.nn.Module):
        def __init__(self, feat_dim, embed_dim, hidden_dim, vocab_size):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.attention = Attention(feat_dim, hidden_dim)
            self.lstm = torch.nn.LSTM(embed_dim + feat_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)
            self.h0 = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=False)
            self.c0 = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=False)

        def forward(self, feats, tgt_seq):
            B, L, feat_dim = feats.size()
            T = tgt_seq.size(1)
            embedded = self.embedding(tgt_seq)
            
            h = self.h0.expand(1, B, self.lstm.hidden_size).contiguous()
            c = self.c0.expand(1, B, self.lstm.hidden_size).contiguous()
            
            outputs = []
            for t in range(T):
                current_emb = embedded[:, t, :]
                context, _ = self.attention(feats, h.squeeze(0))
                lstm_input = torch.cat([current_emb, context], dim=1).unsqueeze(1)
                output, (h, c) = self.lstm(lstm_input, (h, c))
                logits = self.fc(output.squeeze(1))
                outputs.append(logits.unsqueeze(1))
            
            return torch.cat(outputs, dim=1)

    # Get the model dimensions from the checkpoint
    embed_dim = 256  # Default values, update if your model uses different values
    hidden_dim = 256
    vocab_size = len(checkpoint.get('vocab', vocab))

    # Create models
    encoder = CNNEncoder(out_dim=embed_dim).to(device)
    decoder = SeqDecoder(
        feat_dim=embed_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size
    ).to(device)

    # Load state dictionaries
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['valid_acc']:.2%}")
    
    return encoder, decoder, checkpoint.get('vocab', vocab)

@torch.no_grad()
def generate_predictions(encoder, decoder, data_loader, vocab, device, max_len=50):
    """Generate predictions for a dataset and return as list of (image_id, predicted_latex)."""
    encoder.eval()
    decoder.eval()
    
    # Inverse vocab mapping (index to token)
    idx_to_token = {idx: token for token, idx in vocab.items()}
    
    all_predictions = []
    image_ids = []
    
    for batch_idx, (imgs, seqs, lengths) in enumerate(data_loader):
        print(f"Processing batch {batch_idx+1}/{len(data_loader)}", end="\r")
        
        # Get batch size
        batch_size = imgs.size(0)
        
        # Store image IDs (if available in your dataset)
        # This assumes the dataset has a way to get image IDs, modify as needed
        if hasattr(data_loader.dataset, 'get_image_ids'):
            batch_image_ids = data_loader.dataset.get_image_ids(batch_idx * batch_size, 
                                                              (batch_idx + 1) * batch_size)
            image_ids.extend(batch_image_ids)
        else:
            # If no image IDs are available, just use sequential numbers
            image_ids.extend([f"img_{batch_idx * batch_size + i}" for i in range(batch_size)])
        
        # Move tensors to device
        imgs = imgs.to(device)
        
        # Get image features
        features = encoder(imgs)
        
        # Initialize predictions with <sos> token
        preds = torch.full((batch_size, 1), vocab["<sos>"], dtype=torch.long, device=device)
        
        # Track which sequences are finished (generated <eos> token)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len):
            # Get predictions for next token
            logits = decoder(features, preds)
            next_token = logits[:, -1].argmax(dim=-1).unsqueeze(1)
            
            # Add predicted token to sequence
            preds = torch.cat([preds, next_token], dim=1)
            
            # Check which sequences produced <eos>
            eos_hits = next_token.squeeze(1) == vocab["<eos>"]
            finished = finished | eos_hits
            
            # Stop if all sequences have produced <eos>
            if finished.all():
                break
        
        # Convert token indices to tokens
        for i in range(batch_size):
            tokens = preds[i].tolist()
            
            # Remove <sos> token at beginning
            tokens = tokens[1:]
            
            # Truncate at <eos> if present
            if vocab["<eos>"] in tokens:
                tokens = tokens[:tokens.index(vocab["<eos>"])]
            
            # Convert indices to tokens
            pred_tokens = [idx_to_token.get(idx, "<unk>") for idx in tokens]
            
            # Join tokens to create LaTeX string
            latex_string = "".join(pred_tokens)
            
            # Add to predictions
            all_predictions.append(latex_string)
    
    print("\nFinished generating predictions")
    return image_ids, all_predictions

def main():
    parser = argparse.ArgumentParser(description="Generate predictions and export to CSV")
    parser.add_argument("--checkpoint", default="best_checkpoint.pth", 
                      help="Path to model checkpoint")
    parser.add_argument("--dataset", default="test", choices=["train", "valid", "test"],
                      help="Which dataset split to use")
    parser.add_argument("--output", default="predictions.csv",
                      help="Output CSV file path")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=50,
                      help="Maximum sequence length to generate")
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Please train a model first or specify the correct checkpoint path")
        return
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else
                         "cpu")
    print(f"Using device: {device}")
    
    # Load model
    encoder, decoder, vocab = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create dataset and dataloader
    dataset = MathWritingDataset("data/images/labels.csv", [args.dataset])
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Generate predictions
    image_ids, predictions = generate_predictions(
        encoder, decoder, data_loader, vocab, device, args.max_len
    )
    
    # Create DataFrame with predictions
    df = pd.DataFrame({
        "image_id": image_ids,
        "latex_prediction": predictions
    })
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    
    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Image {image_ids[i]}: {predictions[i]}")

if __name__ == "__main__":
    main()