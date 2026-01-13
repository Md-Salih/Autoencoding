import torch
import torch.nn as nn
import torch.optim as optim
import os
from encoder import TransformerEncoder
from utils import build_vocab, encode_batch, decode_batch
from data_loader import get_wiki_sentences
import random
from tqdm import tqdm

def mask_sentence_random(sentence, mask_prob=0.15):
    """Randomly mask multiple words in a sentence like BERT"""
    words = sentence.split()
    if len(words) < 3:
        return None, None
    
    masked = words.copy()
    num_to_mask = max(1, int(len(words) * mask_prob))
    
    # Don't mask first and last words
    maskable_indices = list(range(1, len(words)-1))
    if not maskable_indices:
        return None, None
    
    mask_indices = random.sample(maskable_indices, min(num_to_mask, len(maskable_indices)))
    
    for idx in mask_indices:
        masked[idx] = '[MASK]'
    
    return ' '.join(masked), ' '.join(words)

def create_training_data(sentences, num_samples=2000):
    """Create training data with multiple masked versions per sentence"""
    data = []
    for s in sentences:
        # Create multiple masked versions of each sentence
        for _ in range(3):  # 3 different masked versions
            masked, orig = mask_sentence_random(s)
            if masked and orig:
                data.append((masked, orig))
    
    # Shuffle and limit
    random.shuffle(data)
    return data[:num_samples]

def compute_mlm_loss(predictions, targets, vocab):
    """Compute loss only on masked positions"""
    mask_id = vocab['[MASK]']
    
    # Flatten predictions and targets
    predictions = predictions.view(-1, predictions.size(-1))
    targets = targets.view(-1)
    
    # Create mask for [MASK] tokens in input
    # We need to track which positions were masked
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
    loss = loss_fn(predictions, targets)
    
    return loss

def main():
    print("Loading Wikipedia sentences...")
    sentences = get_wiki_sentences(2000)  # More data
    
    print("Creating training data...")
    data = create_training_data(sentences, num_samples=3000)
    print(f"Prepared {len(data)} masked sentences.")
    
    # Build vocabulary
    all_text = [s for s, _ in data] + [t for _, t in data]
    vocab = build_vocab(all_text)
    vocab_size = len(vocab)
    max_len = min(64, max(len(s.split()) for s, _ in data) + 2)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_len}")
    
    # IMPROVED MODEL ARCHITECTURE - Much larger and more powerful
    d_model = 256      # Increased from 64
    n_heads = 8        # Increased from 4
    num_layers = 6     # Increased from 2
    d_ff = 1024        # Increased from 128
    
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "max_len": max_len,
        "task": "mlm",
    }
    
    print("\nModel Configuration:")
    print(f"- d_model: {d_model}")
    print(f"- n_heads: {n_heads}")
    print(f"- num_layers: {num_layers}")
    print(f"- d_ff: {d_ff}")
    
    model = TransformerEncoder(vocab_size, d_model, n_heads, num_layers, d_ff, max_len)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Use regular Adam for compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = 20  # More epochs
    batch_size = 16  # Process multiple samples at once
    
    print("\nStarting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        
        # Create batches
        num_batches = len(data) // batch_size
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = data[start_idx:end_idx]
                
                # Prepare batch
                src_sentences = [s for s, _ in batch_data]
                tgt_sentences = [t for _, t in batch_data]
                
                src_ids = encode_batch(src_sentences, vocab, max_len)
                tgt_ids = encode_batch(tgt_sentences, vocab, max_len)
                
                src_tensor = torch.tensor(src_ids)
                tgt_tensor = torch.tensor(tgt_ids)
                
                # Forward pass
                out, _ = model(src_tensor)
                
                # Compute loss
                out = out.view(-1, vocab_size)
                tgt_tensor = tgt_tensor.view(-1)
                loss = criterion(out, tgt_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Test on some examples every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("\n--- Sample Predictions ---")
            test_samples = random.sample(data, min(3, len(data)))
            for src, tgt in test_samples:
                src_ids = encode_batch([src], vocab, max_len)
                src_tensor = torch.tensor(src_ids)
                with torch.no_grad():
                    out, _ = model(src_tensor)
                    pred_ids = out.argmax(-1)
                    output = decode_batch(pred_ids.tolist(), vocab)[0]
                print(f"Input:  {src}")
                print(f"Target: {tgt}")
                print(f"Output: {output}")
                print()
            model.train()
    
    # Save model
    print("\nSaving model...")
    from save_load_model import save_model
    save_model(model, vocab, path="results/model_wiki_improved.pth", config=config)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    test_samples = random.sample(data, min(10, len(data)))
    correct_words = 0
    total_words = 0
    
    for src, tgt in test_samples:
        src_ids = encode_batch([src], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        with torch.no_grad():
            out, _ = model(src_tensor)
            pred_ids = out.argmax(-1)
            output = decode_batch(pred_ids.tolist(), vocab)[0]
        
        # Count correct predictions
        src_words = src.split()
        tgt_words = tgt.split()
        pred_words = output.split()
        
        for i, (s, t) in enumerate(zip(src_words, tgt_words)):
            if s == '[MASK]' and i < len(pred_words):
                total_words += 1
                if pred_words[i] == t:
                    correct_words += 1
        
        print(f"Input:  {src}")
        print(f"Target: {tgt}")
        print(f"Output: {output}")
        print()
    
    if total_words > 0:
        accuracy = (correct_words / total_words) * 100
        print(f"Masked Token Accuracy: {accuracy:.2f}% ({correct_words}/{total_words})")
    
    print(f"\nModel saved to: results/model_wiki_improved.pth")

if __name__ == "__main__":
    main()
