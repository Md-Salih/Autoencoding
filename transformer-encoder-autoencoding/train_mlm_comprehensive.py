"""
Train Student Model on Comprehensive Dataset
Uses diverse text sources for better generalization
"""

import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam

# Add parent to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

from encoder import TransformerEncoder
from data_loader_comprehensive import get_comprehensive_mlm_dataset
from utils import build_vocab, encode_batch


def train_comprehensive_model(
    epochs: int = 100,
    batch_size: int = 16,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    lr: float = 0.0005,
    device: str = "cpu"
):
    """Train student model on comprehensive dataset"""
    
    print("=" * 60)
    print("COMPREHENSIVE STUDENT MODEL TRAINING")
    print("=" * 60)
    
    # Load comprehensive dataset
    print("\n[1/6] Loading comprehensive dataset...")
    pairs = get_comprehensive_mlm_dataset()
    print(f"✓ Loaded {len(pairs)} training examples")
    
    # Build vocabulary
    print("\n[2/6] Building vocabulary...")
    all_sentences = []
    for masked, original in pairs:
        all_sentences.append(masked)
        all_sentences.append(original)
    
    vocab = build_vocab(all_sentences)
    print(f"✓ Vocabulary size: {len(vocab)} tokens")
    
    # Determine max sequence length
    max_len = max(len(s.split()) for s, _ in pairs)
    max_len = min(max_len, 50)  # Cap at 50 for efficiency
    print(f"✓ Max sequence length: {max_len}")
    
    # Initialize model
    print(f"\n[3/6] Initializing model...")
    print(f"  - Embedding dimension: {d_model}")
    print(f"  - Attention heads: {n_heads}")
    print(f"  - Encoder layers: {n_layers}")
    print(f"  - Feed-forward dimension: {d_ff}")
    
    model = TransformerEncoder(len(vocab), d_model, n_heads, n_layers, d_ff, max_len)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Setup training
    print(f"\n[4/6] Setting up training...")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get("[PAD]", 0))
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    print(f"✓ Using Adam optimizer with lr={lr}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Training epochs: {epochs}")
    print(f"\n⏱️  Estimated training time: 10-15 minutes")
    print(f"💡 Progress will be shown every 5 epochs")
    
    # Training loop
    print(f"\n[5/6] Training model...")
    print("-" * 60)
    print("🚀 Training started... Please wait, this may take 10-15 minutes")
    print("📊 Progress indicator: ░ = not started, ▓ = completed")
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    # Progress bar setup
    total_epochs = epochs
    progress_bar_width = 50
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle data each epoch
        import random
        random.shuffle(pairs)
        
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Prepare batch
            masked_sentences = [masked for masked, _ in batch_pairs]
            target_sentences = [target for _, target in batch_pairs]
            
            # Encode sentences
            x = torch.tensor(encode_batch(masked_sentences, vocab, max_len)).to(device)
            y = torch.tensor(encode_batch(target_sentences, vocab, max_len)).to(device)
            
            # Forward pass
            logits, _ = model(x)  # [B, T, vocab_size]
            
            # Reshape for loss
            logits = logits.view(-1, len(vocab))
            y = y.view(-1)
            
            # Compute loss
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Print progress with visual progress bar
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Calculate progress percentage
            progress = (epoch + 1) / total_epochs
            filled_length = int(progress_bar_width * progress)
            bar = '▓' * filled_length + '░' * (progress_bar_width - filled_length)
            percent = progress * 100
            
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Progress: |{bar}| {percent:.1f}%")
            
            # Time estimation
            if epoch > 0:
                import time
                elapsed_epochs = epoch + 1
                remaining_epochs = total_epochs - elapsed_epochs
                print(f"Remaining: ~{remaining_epochs} epochs\n")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(os.path.dirname(__file__), "results", "model_comprehensive.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab,
                'max_len': max_len,
                'config': {
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'n_layers': n_layers,
                    'd_ff': d_ff
                }
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print("-" * 60)
    print(f"✓ Training completed!")
    print(f"✓ Best loss: {best_loss:.4f}")
    
    # Test predictions
    print(f"\n[6/6] Testing predictions...")
    model.eval()
    test_cases = [
        "She drinks [MASK] in the morning",
        "The earth revolves around the [MASK]",
        "Computers process data using [MASK] code",
        "Trees produce [MASK] through photosynthesis",
        "Students learn mathematics using [MASK] methods"
    ]
    
    with torch.no_grad():
        for test_sent in test_cases:
            x = torch.tensor(encode_batch([test_sent], vocab, max_len)).to(device)
            logits, _ = model(x)
            
            # Find mask position
            words = test_sent.split()
            mask_idx = words.index("[MASK]")
            
            # Get prediction
            pred_logits = logits[0, mask_idx]
            pred_idx = pred_logits.argmax().item()
            
            # Get word
            idx_to_word = {idx: word for word, idx in vocab.items()}
            pred_word = idx_to_word.get(pred_idx, "<UNK>")
            
            # Show top 3 predictions
            top_k = 3
            top_probs, top_indices = torch.softmax(pred_logits, dim=0).topk(top_k)
            top_words = [idx_to_word.get(idx.item(), "<UNK>") for idx in top_indices]
            
            print(f"\nInput: {test_sent}")
            print(f"Top {top_k} predictions:")
            for j, (word, prob) in enumerate(zip(top_words, top_probs), 1):
                print(f"  {j}. {word} ({prob.item()*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("MODEL SAVED: results/model_comprehensive.pth")
    print("=" * 60)
    print("\nThis comprehensive student model can now handle:")
    print("  • General knowledge and facts")
    print("  • Daily life situations")
    print("  • Technology concepts")
    print("  • Nature and animals")
    print("  • Education topics")
    print("  • Health and wellness")
    print("  • Business and economics")
    print("  • Travel and geography")
    print("  • Sports and athletics")
    print("  • And much more!")
    print()


if __name__ == "__main__":
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_comprehensive_model(
        epochs=100,
        batch_size=16,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        lr=0.0005,
        device=device
    )
