import torch
import torch.nn as nn
import torch.optim as optim
import os
from encoder import TransformerEncoder
from utils import build_vocab, encode_batch, decode_batch
import random
from tqdm import tqdm

def load_large_dataset():
    """Load a large diverse dataset for better training"""
    # Use the existing data loader that works
    try:
        from data_loader import get_wiki_sentences
        print("Loading Wikipedia sentences...")
        sentences = get_wiki_sentences(5000)  # Get 5000 sentences
        print(f"Loaded {len(sentences)} sentences from Wikipedia")
        
        # Add common sentence patterns for better generalization
        common_sentences = [
            "The cat sits on the mat",
            "I love to eat pizza",
            "She drinks coffee in the morning",
            "He plays football with friends",
            "The dog runs in the park",
            "They go to school every day",
            "We watch movies on weekends",
            "The sun shines brightly",
            "Birds fly in the sky",
            "Children play in the garden",
            "The car drives on the road",
            "People walk on the street",
            "She reads books at night",
            "He writes letters to his friend",
            "The teacher teaches students",
            "The doctor helps patients",
            "The chef cooks delicious food",
            "The artist paints beautiful pictures",
            "The musician plays wonderful music",
            "The athlete runs very fast",
            "The student studies hard",
            "The worker builds houses",
            "The farmer grows vegetables",
            "The driver delivers packages",
            "The nurse cares for sick people",
        ]
        
        # Add more diverse variations
        sentence_templates = []
        for base in common_sentences:
            sentence_templates.append(base)
            # Create variations
            words = base.split()
            if len(words) >= 4:
                # Create different word order variations (where it makes sense)
                sentence_templates.append(base.replace("in the", "at the"))
                sentence_templates.append(base.replace("on the", "near the"))
                sentence_templates.append(base.replace("to", "for"))
        
        sentences.extend(sentence_templates * 20)  # Repeat for more training data
        
        return sentences
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def mask_sentence_mlm(sentence, mask_prob=0.15):
    """Mask multiple words in a sentence (BERT-style)"""
    words = sentence.split()
    if len(words) < 3:
        return None, None
    
    masked = words.copy()
    targets = words.copy()
    
    # Decide how many words to mask
    num_to_mask = max(1, int(len(words) * mask_prob))
    
    # Don't mask first and last words
    maskable_indices = list(range(0, len(words)))
    if len(maskable_indices) == 0:
        return None, None
    
    # Randomly select positions to mask
    num_to_mask = min(num_to_mask, len(maskable_indices))
    mask_indices = random.sample(maskable_indices, num_to_mask)
    
    for idx in mask_indices:
        rand = random.random()
        if rand < 0.8:  # 80% replace with [MASK]
            masked[idx] = '[MASK]'
        elif rand < 0.9:  # 10% replace with random word
            masked[idx] = random.choice(words)
        # 10% keep original (else clause)
    
    return ' '.join(masked), ' '.join(targets)

def create_diverse_training_data(sentences, augmentation_factor=3):
    """Create diverse training data with augmentation"""
    data = []
    for sentence in tqdm(sentences, desc="Creating training data"):
        # Create multiple masked versions
        for _ in range(augmentation_factor):
            masked, target = mask_sentence_mlm(sentence)
            if masked and target:
                data.append((masked, target))
    
    random.shuffle(data)
    return data

def main():
    # Load large diverse dataset
    sentences = load_large_dataset()
    
    if len(sentences) < 100:
        print("Error: Not enough sentences loaded. Please check dataset availability.")
        return
    
    print(f"\nCreating training data from {len(sentences)} sentences...")
    data = create_diverse_training_data(sentences, augmentation_factor=2)
    print(f"Created {len(data)} training samples")
    
    # Build comprehensive vocabulary
    print("\nBuilding vocabulary...")
    all_text = []
    for src, tgt in data:
        all_text.append(src)
        all_text.append(tgt)
    
    vocab = build_vocab(all_text)
    vocab_size = len(vocab)
    max_len = min(64, max(len(s.split()) for s, _ in data[:1000]) + 2)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_len}")
    
    # POWERFUL MODEL ARCHITECTURE
    d_model = 384       # Larger embedding
    n_heads = 8         # 8 attention heads
    num_layers = 8      # Deeper network
    d_ff = 1536         # Larger feed-forward
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    epochs = 25
    batch_size = 32
    
    print("\nStarting training...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        num_batches = len(data) // batch_size
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = data[start_idx:end_idx]
                
                src_sentences = [s for s, _ in batch_data]
                tgt_sentences = [t for _, t in batch_data]
                
                src_ids = encode_batch(src_sentences, vocab, max_len)
                tgt_ids = encode_batch(tgt_sentences, vocab, max_len)
                
                src_tensor = torch.tensor(src_ids)
                tgt_tensor = torch.tensor(tgt_ids)
                
                out, _ = model(src_tensor)
                out = out.view(-1, vocab_size)
                tgt_tensor = tgt_tensor.view(-1)
                
                loss = criterion(out, tgt_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best model! Saving...")
            from save_load_model import save_model
            save_model(model, vocab, path="results/model_wiki_robust.pth", config=config)
        
        # Test samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("\n--- Sample Predictions ---")
            test_samples = random.sample(data, min(5, len(data)))
            for src, tgt in test_samples:
                src_ids = encode_batch([src], vocab, max_len)
                src_tensor = torch.tensor(src_ids)
                with torch.no_grad():
                    out, _ = model(src_tensor)
                    pred_ids = out.argmax(-1)
                    output = decode_batch(pred_ids.tolist(), vocab)[0]
                print(f"Input:  {src}")
                print(f"Target: {tgt}")
                print(f"Output: {output}\n")
            model.train()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    test_samples = random.sample(data, min(20, len(data)))
    correct_tokens = 0
    total_masked_tokens = 0
    
    for src, tgt in test_samples:
        src_ids = encode_batch([src], vocab, max_len)
        src_tensor = torch.tensor(src_ids)
        with torch.no_grad():
            out, _ = model(src_tensor)
            pred_ids = out.argmax(-1)
            output = decode_batch(pred_ids.tolist(), vocab)[0]
        
        src_words = src.split()
        tgt_words = tgt.split()
        pred_words = output.split()
        
        for i, (s, t) in enumerate(zip(src_words, tgt_words)):
            if s == '[MASK]' and i < len(pred_words):
                total_masked_tokens += 1
                if pred_words[i] == t:
                    correct_tokens += 1
        
        print(f"Input:  {src}")
        print(f"Target: {tgt}")
        print(f"Output: {output}\n")
    
    if total_masked_tokens > 0:
        accuracy = (correct_tokens / total_masked_tokens) * 100
        print(f"Masked Token Accuracy: {accuracy:.2f}% ({correct_tokens}/{total_masked_tokens})")
    
    print(f"\nBest model saved to: results/model_wiki_robust.pth")

if __name__ == "__main__":
    main()
