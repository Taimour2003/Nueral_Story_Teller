# part 1

# ============================================================================
# PART 1: FEATURE EXTRACTION (FIXED VERSION)
# ============================================================================

import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# ============================================================================
# STEP 1: FIND IMAGE DIRECTORY (FIXED)
# ============================================================================

def find_image_dir():
    """
    Find the Flickr30k image directory in Kaggle input
    """
    # Common Kaggle paths
    possible_paths = [
        '/kaggle/input/flickr30k/Images',
        '/kaggle/input/flickr30k/flickr30k_images/flickr30k_images',
        '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images',
        '/kaggle/input/flickr30k-images/flickr30k_images/flickr30k_images'
    ]
    
    # Try common paths first
    for path in possible_paths:
        if os.path.exists(path):
            jpg_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg'))])
            if jpg_count > 1000:
                return path
    
    # If not found, walk through the input directory
    base_input = '/kaggle/input'
    for root, dirs, files in os.walk(base_input):
        # Look for folder with many jpg files
        jpg_files = [f for f in files if f.endswith(('.jpg', '.jpeg'))]
        if len(jpg_files) > 1000:
            return root
    
    return None  # ← NOW properly outside the loop


IMAGE_DIR = find_image_dir()
OUTPUT_FILE = 'flickr30k_features.pkl'

if IMAGE_DIR:
    print(f"✓ Found images at: {IMAGE_DIR}")
    # Count total images
    total_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg'))])
    print(f"✓ Total images found: {total_images}")
else:
    raise FileNotFoundError(
        "Could not find the Flickr30k image directory.\n"
        "Please ensure the dataset is added to the notebook."
    )

# ============================================================================
# STEP 2: DATASET CLASS
# ============================================================================

class FlickrDataset(Dataset):
    """
    PyTorch Dataset for loading Flickr30k images
    """
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [
            f for f in os.listdir(img_dir) 
            if f.endswith(('.jpg', '.jpeg'))
        ]
        print(f"✓ Dataset initialized with {len(self.img_names)} images")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), name

# ============================================================================
# STEP 3: SETUP MODEL AND TRANSFORMS
# ============================================================================

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load ResNet50 pre-trained on ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove the final classification layer (we only want features)
model = nn.Sequential(*list(model.children())[:-1])

# Move to GPU and use DataParallel for faster processing
if torch.cuda.is_available():
    model = nn.DataParallel(model)
model = model.to(device)
model.eval()

print("✓ ResNet50 loaded (feature extractor mode)")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================================
# STEP 4: CREATE DATALOADER
# ============================================================================

dataset = FlickrDataset(IMAGE_DIR, transform)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"✓ DataLoader created with batch_size=128")
print(f"✓ Total batches: {len(loader)}")

# ============================================================================
# STEP 5: EXTRACT FEATURES (FIXED!)
# ============================================================================

features_dict = {}

print("\nExtracting features...")
with torch.no_grad():
    for imgs, names in tqdm(loader, desc="Processing batches"):
        # Extract features
        feats = model(imgs.to(device)).view(imgs.size(0), -1)
        
        # Save each image's features (FIXED INDENTATION!)
        for i, name in enumerate(names):
            features_dict[name] = feats[i].cpu().numpy()

print(f"\n✓ Feature extraction complete!")
print(f"✓ Processed {len(features_dict)} images")

# ============================================================================
# STEP 6: SAVE FEATURES
# ============================================================================

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(features_dict, f)

print(f"✓ Features saved to {OUTPUT_FILE}")

# ============================================================================
# STEP 7: VERIFY SAVED DATA
# ============================================================================

# Load and verify
with open(OUTPUT_FILE, 'rb') as f:
    loaded_features = pickle.load(f)

print(f"\n{'='*60}")
print("VERIFICATION:")
print(f"{'='*60}")
print(f"Total images saved: {len(loaded_features)}")
print(f"Feature vector shape: {list(loaded_features.values())[0].shape}")
print(f"Feature vector size: {list(loaded_features.values())[0].shape[0]}")
print(f"\nSample image names:")
for i, name in enumerate(list(loaded_features.keys())[:5]):
    print(f"  {i+1}. {name}")

print(f"\n{'='*60}")
print("✓ PART 1 COMPLETE!")
print(f"{'='*60}")





#part 2


#1. Load the file

CAPTIONS_FILE = '/kaggle/input/flickr30k/captions.txt'

if CAPTIONS_FILE:
    print(f"✓ Found captions at: {CAPTIONS_FILE}")
else:
    raise FileNotFoundError("Could not find captions.txt. Please check the dataset.")



#2.Load And Parse Captions 
def load_captions(captions_file):
    captions_dict = {}
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        # Skip header if present
        first_line = f.readline()
        if not first_line.strip().endswith(('.jpg', '.jpeg', '.png')):
            # First line is likely a header, skip it
            pass
        else:
            # First line is data, process it
            f.seek(0)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split on first comma only (caption might contain commas)
            parts = line.split(',', 1)
            
            if len(parts) != 2:
                continue
            
            img_name = parts[0].strip()
            caption = parts[1].strip()
            
            # Skip if caption is empty or too short
            if len(caption) < 3:
                continue
            
            if img_name not in captions_dict:
                captions_dict[img_name] = []
            
            captions_dict[img_name].append(caption)
    
    return captions_dict





captions_dict = load_captions(CAPTIONS_FILE)
print(f"✓ Loaded captions for {len(captions_dict)} images")
print(f"✓ Total captions: {sum(len(caps) for caps in captions_dict.values())}")

# Show example
sample_img = list(captions_dict.keys())[0]
print(f"\nExample - Image: {sample_img}")
for i, cap in enumerate(captions_dict[sample_img][:3], 1):
    print(f"  Caption {i}: {cap}")


# STEP 3: TEXT CLEANING AND TOKENIZATION
def clean_caption(caption):
    
    import re
    
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove special characters and digits, keep only letters and spaces
    caption = re.sub(r'[^a-z\s]', '', caption)
    
    # Remove extra whitespace
    
    caption = ' '.join(caption.split())
    
    return caption

def tokenize_captions(captions_dict):
    """
    Tokenize all captions and return cleaned version
    """
    tokenized_captions = {}
    
    for img_name, captions in captions_dict.items():
        tokenized_captions[img_name] = []
        
        for caption in captions:
            # Clean the caption
            cleaned = clean_caption(caption)
            
            # Skip if cleaning resulted in empty string
            if not cleaned:
                continue
            
            # Split into words (tokens)
            tokens = cleaned.split()
            
            # Only keep captions with at least 2 words
            if len(tokens) >= 2:
                tokenized_captions[img_name].append(tokens)
    
    # Remove images with no valid captions
    tokenized_captions = {k: v for k, v in tokenized_captions.items() if v}
    
    return tokenized_captions

tokenized_captions = tokenize_captions(captions_dict)
print(f"\n✓ Tokenized all captions")
print(f"✓ Valid images after tokenization: {len(tokenized_captions)}")

# Show example
print(f"\nExample - Image: {sample_img}")
if sample_img in tokenized_captions and tokenized_captions[sample_img]:
    print(f"  Original: {captions_dict[sample_img][0]}")
    print(f"  Tokenized: {tokenized_captions[sample_img][0]}")


from collections import Counter
# ============================================================================
# STEP 4: BUILD VOCABULARY
# ============================================================================

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        Initialize vocabulary with special tokens
        
        Args:
            freq_threshold: Minimum frequency for a word to be included in vocab
        """
        # Special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"
        
        # Mappings
        self.word2idx = {}
        self.idx2word = {}
        self.freq_threshold = freq_threshold
        
        # Initialize with special tokens
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.start_token] = 1
        self.word2idx[self.end_token] = 2
        self.word2idx[self.unk_token] = 3
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.idx = 4  # Next available index
    
    def build_vocabulary(self, tokenized_captions):
        """
        Build vocabulary from tokenized captions
        """
        # Count word frequencies
        word_freq = Counter()
        
        for img_name, captions_list in tokenized_captions.items():
            for tokens in captions_list:
                word_freq.update(tokens)
        
        print(f"✓ Total unique words before filtering: {len(word_freq)}")
        
        # Add words that meet frequency threshold
        for word, freq in word_freq.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        
        print(f"✓ Vocabulary size (freq >= {self.freq_threshold}): {len(self.word2idx)}")
        print(f"✓ Special tokens: {self.pad_token}, {self.start_token}, {self.end_token}, {self.unk_token}")
        
        # Show most common words
        print(f"\n✓ Top 10 most common words:")
        for word, freq in word_freq.most_common(10):
            if word in self.word2idx:
                print(f"   '{word}': {freq} occurrences")
    
    def numericalize(self, tokens):
        """
        Convert list of tokens to list of indices
        """
        indices = [self.word2idx[self.start_token]]
        
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx[self.unk_token])
        
        indices.append(self.word2idx[self.end_token])
        
        return indices
    
    def denumericalize(self, indices):
        """
        Convert list of indices back to tokens
        """
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                # Stop at end token
                if token == self.end_token:
                    break
                # Don't include start, pad tokens in output
                if token not in [self.start_token, self.pad_token]:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)

# Build the vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(tokenized_captions)

# Show examples
print(f"\n{'='*60}")
print("VOCABULARY EXAMPLES:")
print(f"{'='*60}")
print(f"Vocab size: {len(vocab)}")
print(f"\nFirst 20 words in vocabulary:")
for idx in range(min(20, len(vocab))):
    print(f"  {idx}: {vocab.idx2word[idx]}")


# STEP 5: NUMERICALIZE ALL CAPTIONS

def numericalize_all_captions(tokenized_captions, vocab):
    """
    Convert all captions to numerical indices
    """
    numericalized = {}
    
    for img_name, captions_list in tokenized_captions.items():
        numericalized[img_name] = []
        
        for tokens in captions_list:
            indices = vocab.numericalize(tokens)
            numericalized[img_name].append(indices)
    
    return numericalized



numericalized_captions = numericalize_all_captions(tokenized_captions, vocab)

# Show example
print(f"\n{'='*60}")
print("NUMERICALIZATION EXAMPLE:")
print(f"{'='*60}")
print(f"Image: {sample_img}")
if sample_img in tokenized_captions and tokenized_captions[sample_img]:
    print(f"Original caption: {captions_dict[sample_img][0]}")
    print(f"Tokenized: {tokenized_captions[sample_img][0]}")
    print(f"Numericalized: {numericalized_captions[sample_img][0]}")
    print(f"Back to text: {vocab.denumericalize(numericalized_captions[sample_img][0])}")


# STEP 6: STATISTICS AND ANALYSIS

#import os
#import pickle
#import pandas as pd
import numpy as np

def get_caption_statistics(numericalized_captions):
    """
    Get statistics about caption lengths
    """
    all_lengths = []
    
    for img_name, captions_list in numericalized_captions.items():
        for indices in captions_list:
            all_lengths.append(len(indices))
    
    return {
        'min_length': min(all_lengths),
        'max_length': max(all_lengths),
        'mean_length': np.mean(all_lengths),
        'median_length': np.median(all_lengths),
        'std_length': np.std(all_lengths)
    }

stats = get_caption_statistics(numericalized_captions)

print(f"\n{'='*60}")
print("CAPTION LENGTH STATISTICS:")
print(f"{'='*60}")
print(f"Minimum length: {stats['min_length']}")
print(f"Maximum length: {stats['max_length']}")
print(f"Mean length: {stats['mean_length']:.2f}")
print(f"Median length: {stats['median_length']:.2f}")
print(f"Std deviation: {stats['std_length']:.2f}")

# Suggested max length for padding (mean + 2*std covers ~95% of data)
MAX_LENGTH = int(stats['mean_length'] + 2 * stats['std_length'])
print(f"\n✓ Suggested MAX_LENGTH for padding: {MAX_LENGTH}")


# ============================================================================
# STEP 7: SAVE PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(vocab, numericalized_captions, tokenized_captions, 
                           output_file='flickr30k_preprocessed.pkl'):
 
    data = {
        'vocab': vocab,
        'numericalized_captions': numericalized_captions,
        'tokenized_captions': tokenized_captions,
        'vocab_size': len(vocab),
        'pad_idx': vocab.word2idx[vocab.pad_token],
        'start_idx': vocab.word2idx[vocab.start_token],
        'end_idx': vocab.word2idx[vocab.end_token],
        'unk_idx': vocab.word2idx[vocab.unk_token],
        'max_length': MAX_LENGTH
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Preprocessed data saved to {output_file}")
    return data

preprocessed_data = save_preprocessed_data(vocab, numericalized_captions, tokenized_captions)


# Load the features from Part 1
with open('flickr30k_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

print(f"\n{'='*60}")
print("DATA INTEGRITY CHECK:")
print(f"{'='*60}")
print(f"Images with features: {len(features_dict)}")
print(f"Images with captions: {len(numericalized_captions)}")

# Find intersection
common_images = set(features_dict.keys()) & set(numericalized_captions.keys())
print(f"✓ Common images (features + captions): {len(common_images)}")

# Images with features but no captions
only_features = set(features_dict.keys()) - set(numericalized_captions.keys())
if only_features:
    print(f"⚠ Warning: {len(only_features)} images have features but no captions")
    print(f"  Example: {list(only_features)[:3]}")

# Images with captions but no features
only_captions = set(numericalized_captions.keys()) - set(features_dict.keys())
if only_captions:
    print(f"⚠ Warning: {len(only_captions)} images have captions but no features")
    print(f"  Example: {list(only_captions)[:3]}")

print(f"\n✓ Part 2 Complete! Ready for Part 3.")


# ============================================================================
# PART 3: SEQ2SEQ ARCHITECTURE FOR IMAGE CAPTIONING
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================

print("Loading preprocessed data...")

# Load features from Part 1
with open('flickr30k_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)
print(f"✓ Loaded features for {len(features_dict)} images")

# Load vocabulary and captions from Part 2
with open('flickr30k_preprocessed.pkl', 'rb') as f:
    preprocessed_data = pickle.load(f)

vocab = preprocessed_data['vocab']
numericalized_captions = preprocessed_data['numericalized_captions']
MAX_LENGTH = preprocessed_data['max_length']
vocab_size = preprocessed_data['vocab_size']
pad_idx = preprocessed_data['pad_idx']
start_idx = preprocessed_data['start_idx']
end_idx = preprocessed_data['end_idx']

print(f"✓ Vocabulary size: {vocab_size}")
print(f"✓ Max caption length: {MAX_LENGTH}")
print(f"✓ Special tokens - PAD: {pad_idx}, START: {start_idx}, END: {end_idx}")


# ============================================================================
# STEP 2: ENCODER - IMAGE FEATURE ENCODER
# ============================================================================

class ImageEncoder(nn.Module):
    """
    Encodes ResNet50 features (2048-dim) into a hidden state for the decoder
    """
    def __init__(self, resnet_feature_size=2048, hidden_size=512, dropout=0.5):
        """
        Args:
            resnet_feature_size: Size of ResNet50 features (2048)
            hidden_size: Size of LSTM hidden state (512)
            dropout: Dropout probability for regularization
        """
        super(ImageEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Linear layer to transform ResNet features to hidden state
        self.fc = nn.Linear(resnet_feature_size, hidden_size)
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, image_features):
        """
        Args:
            image_features: (batch_size, 2048) - ResNet50 features
        
        Returns:
            hidden: (batch_size, hidden_size) - Encoded features
        """
        # Transform features
        hidden = self.fc(image_features)  # (batch_size, hidden_size)
        hidden = self.bn(hidden)          # Batch norm
        hidden = self.relu(hidden)        # Activation
        hidden = self.dropout(hidden)     # Dropout
        
        return hidden


# ============================================================================
# STEP 3: DECODER - CAPTION GENERATOR WITH LSTM
# ============================================================================

class CaptionDecoder(nn.Module):
    """
    Generates captions word-by-word using LSTM
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, 
                 num_layers=1, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CaptionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        #batch_first=True  -> In an input tensor , there exists an order of dimensions (sequence_Length, Batch_size ,Hidden_size), The batch_first=True specifies that batchsize will come first
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to project LSTM output to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize embedding and linear layer weights
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
    
    def forward(self, captions, hidden_state):
        """
        Args:
            captions: (batch_size, seq_len) - Input caption tokens
            hidden_state: (batch_size, hidden_size) - Initial hidden state from encoder
        
        Returns:
            outputs: (batch_size, seq_len, vocab_size) - Logits for each word
        """
        # Get word embeddings
        embeddings = self.embedding(captions)  # (batch_size, seq_len, embed_size)
        embeddings = self.dropout(embeddings)
        
        # Prepare hidden state for LSTM
        # LSTM expects hidden state as (num_layers, batch_size, hidden_size)
        batch_size = hidden_state.size(0)
        h0 = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)  # Cell state initialized to zeros
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embeddings, (h0, c0))  # (batch_size, seq_len, hidden_size)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary size
        outputs = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate_caption(self, hidden_state, max_length=50, method='greedy', beam_size=3):
        """
        Generate caption using greedy search or beam search
        
        Args:
            hidden_state: (1, hidden_size) - Encoded image features
            max_length: Maximum caption length
            method: 'greedy' or 'beam'
            beam_size: Beam size for beam search
        
        Returns:
            List of word indices
        """
        if method == 'greedy':
            return self._greedy_search(hidden_state, max_length)
        elif method == 'beam':
            return self._beam_search(hidden_state, max_length, beam_size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _greedy_search(self, hidden_state, max_length):
        """
        Greedy search: Always pick the word with highest probability
        """
        result = []
        
        # Start with <start> token
        input_word = torch.tensor([start_idx]).unsqueeze(0).to(device)  # (1, 1)
        
        # Prepare hidden state
        h = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        for _ in range(max_length):
            # Get embedding
            embed = self.embedding(input_word)  # (1, 1, embed_size)
            
            # LSTM forward
            lstm_out, (h, c) = self.lstm(embed, (h, c))  # (1, 1, hidden_size)
            
            # Project to vocabulary
            output = self.fc(lstm_out.squeeze(1))  # (1, vocab_size)
            
            # Get word with highest probability
            predicted = output.argmax(dim=1)  # (1,)
            word_idx = predicted.item()
            
            # Stop if <end> token
            if word_idx == end_idx:
                break
            
            result.append(word_idx)
            
            # Use predicted word as next input
            input_word = predicted.unsqueeze(1)  # (1, 1)
        
        return result
    
    def _beam_search(self, hidden_state, max_length, beam_size):
        """
        Beam search: Keep top-k candidates at each step
        """
        # Prepare initial hidden state
        h = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        # Initialize beams: [(sequence, score, hidden, cell)]
        beams = [([start_idx], 0.0, h, c)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            for seq, score, h_state, c_state in beams:
                # If sequence already ended, keep it as is
                if seq[-1] == end_idx:
                    completed.append((seq, score))
                    continue
                
                # Get last word
                input_word = torch.tensor([[seq[-1]]]).to(device)  # (1, 1)
                
                # Get embedding
                embed = self.embedding(input_word)  # (1, 1, embed_size)
                
                # LSTM forward
                lstm_out, (h_new, c_new) = self.lstm(embed, (h_state, c_state))
                
                # Project to vocabulary
                output = self.fc(lstm_out.squeeze(1))  # (1, vocab_size)
                
                # Get log probabilities
                log_probs = F.log_softmax(output, dim=1)  # (1, vocab_size)
                
                # Get top-k words
                topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=1)
                
                # Create new candidates
                for i in range(beam_size):
                    word_idx = topk_indices[0, i].item()
                    word_score = topk_log_probs[0, i].item()
                    
                    new_seq = seq + [word_idx]
                    new_score = score + word_score
                    
                    candidates.append((new_seq, new_score, h_new, c_new))
            
            # Keep top-k candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # If all beams have ended, stop
            if all(seq[-1] == end_idx for seq, _, _, _ in beams):
                completed.extend([(seq, score) for seq, score, _, _ in beams])
                break
        
        # Add remaining beams to completed
        if not completed:
            completed = [(seq, score) for seq, score, _, _ in beams]
        
        # Return sequence with highest score (excluding <start> and <end>)
        best_seq = max(completed, key=lambda x: x[1])[0]
        return [idx for idx in best_seq if idx not in [start_idx, end_idx, pad_idx]]


# ============================================================================
# STEP 4: COMBINED MODEL - IMAGE TO CAPTION
# ============================================================================

class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model combining encoder and decoder
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, 
                 num_layers=1, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = ImageEncoder(
            resnet_feature_size=2048,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, image_features, captions):
        """
        Forward pass for training
        
        Args:
            image_features: (batch_size, 2048) - ResNet features
            captions: (batch_size, seq_len) - Caption tokens
        
        Returns:
            outputs: (batch_size, seq_len, vocab_size) - Logits
        """
        # Encode image
        hidden_state = self.encoder(image_features)
        
        # Decode to caption
        outputs = self.decoder(captions, hidden_state)
        
        return outputs
    
    def generate_caption(self, image_features, max_length=50, method='greedy', beam_size=3):
        """
        Generate caption for a single image
        
        Args:
            image_features: (2048,) or (1, 2048) - ResNet features
            max_length: Maximum caption length
            method: 'greedy' or 'beam'
            beam_size: Beam size for beam search
        
        Returns:
            caption: String caption
        """
        self.eval()
        
        with torch.no_grad():
            # Ensure correct shape
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            
            # Encode image
            hidden_state = self.encoder(image_features)
            
            # Generate caption
            word_indices = self.decoder.generate_caption(
                hidden_state, 
                max_length=max_length,
                method=method,
                beam_size=beam_size
            )
            
            # Convert indices to words
            caption = vocab.denumericalize(word_indices)
        
        return caption


# ============================================================================
# STEP 5: INITIALIZE MODEL
# ============================================================================

# Hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5

# Initialize model
model = ImageCaptioningModel(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*60}")
print("MODEL ARCHITECTURE:")
print(f"{'='*60}")
print(model)
print(f"\n{'='*60}")
print("MODEL STATISTICS:")
print(f"{'='*60}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nHyperparameters:")
print(f"  - Vocabulary size: {vocab_size}")
print(f"  - Embedding size: {EMBED_SIZE}")
print(f"  - Hidden size: {HIDDEN_SIZE}")
print(f"  - Number of LSTM layers: {NUM_LAYERS}")
print(f"  - Dropout: {DROPOUT}")


# ============================================================================
# STEP 6: TEST MODEL WITH SAMPLE DATA
# ============================================================================

print(f"\n{'='*60}")
print("TESTING MODEL:")
print(f"{'='*60}")

# Get a sample image
sample_img_name = list(features_dict.keys())[0]
sample_features = torch.tensor(features_dict[sample_img_name]).float().unsqueeze(0).to(device)
#.float() --> mostly the raw data is in integer or double precision but the pytorch expects the data in float32 , without this it will gives me Runtime error 
# .unsqueeze(0)--> without unsqueeze() the data inlcudes just features , but the pytorch expects the data in the batch format , thats why we apply unsqueeze
sample_caption_indices = numericalized_captions[sample_img_name][0]

print(f"Sample image: {sample_img_name}")
print(f"Ground truth caption: {vocab.denumericalize(sample_caption_indices)}")
print(f"Image features shape: {sample_features.shape}")

# Test forward pass
model.eval()  # enabling the evaluation mode 
with torch.no_grad():  # Dont keep track of my code , Turning of the tracker make the code run faster 
    # Test encoder
    hidden = model.encoder(sample_features)
    print(f"\nEncoder output shape: {hidden.shape}")
    
    # Test greedy search
    print(f"\nGenerating caption with GREEDY SEARCH...")
    greedy_caption = model.generate_caption(sample_features, method='greedy', max_length=20)
    print(f"Generated caption (greedy): {greedy_caption}")
    
    # Test beam search
    print(f"\nGenerating caption with BEAM SEARCH (beam_size=3)...")
    beam_caption = model.generate_caption(sample_features, method='beam', beam_size=3, max_length=20)
    print(f"Generated caption (beam): {beam_caption}")

# ============================================================================
# STEP 7: SAVE MODEL ARCHITECTURE
# ============================================================================

# Save model architecture and hyperparameters
model_config = {
    'vocab_size': vocab_size,
    'embed_size': EMBED_SIZE,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'max_length': MAX_LENGTH,
    'pad_idx': pad_idx,
    'start_idx': start_idx,
    'end_idx': end_idx,
}

with open('model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

print(f"\n✓ Model configuration saved to model_config.pkl")
print(f"\n{'='*60}")
print("✓ PART 3 COMPLETE!")
print(f"{'='*60}")
print("Next steps:")
print("  1. Implement training loop (Part 4)")
print("  2. Train the model")
print("  3. Generate captions and evaluate")
print("  4. Deploy with Gradio")


# ============================================================================
# PART 4: TRAINING & INFERENCE
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# STEP 1: LOAD ALL PREPROCESSED DATA AND MODEL
# ============================================================================

print("Loading preprocessed data...")

# Load features from Part 1
with open('flickr30k_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Load vocabulary and captions from Part 2
with open('flickr30k_preprocessed.pkl', 'rb') as f:
    preprocessed_data = pickle.load(f)

vocab = preprocessed_data['vocab']
numericalized_captions = preprocessed_data['numericalized_captions']
MAX_LENGTH = preprocessed_data['max_length']
vocab_size = preprocessed_data['vocab_size']
pad_idx = preprocessed_data['pad_idx']
start_idx = preprocessed_data['start_idx']
end_idx = preprocessed_data['end_idx']

# Load model config
with open('model_config.pkl', 'rb') as f:
    model_config = pickle.load(f)

print(f"✓ Loaded features for {len(features_dict)} images")
print(f"✓ Vocabulary size: {vocab_size}")
print(f"✓ Max caption length: {MAX_LENGTH}")

# Find common images (have both features and captions)
common_images = list(set(features_dict.keys()) & set(numericalized_captions.keys()))
print(f"✓ Training on {len(common_images)} images with both features and captions")

# ============================================================================
# STEP 2: CREATE DATASET CLASS
# ============================================================================

class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset for Flickr30k image captioning
    """
    def __init__(self, image_names, features_dict, captions_dict, vocab):
        """
        Args:
            image_names: List of image filenames
            features_dict: Dict mapping image names to ResNet features
            captions_dict: Dict mapping image names to list of caption indices
            vocab: Vocabulary object
        """
        self.image_names = image_names
        self.features_dict = features_dict
        self.captions_dict = captions_dict
        self.vocab = vocab
        
        # Create flat list of (image_name, caption_indices) pairs
        self.samples = []
        for img_name in image_names:
            if img_name in features_dict and img_name in captions_dict:
                for caption in captions_dict[img_name]:
                    self.samples.append((img_name, caption))
        
        print(f"  Dataset created with {len(self.samples)} image-caption pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_features: (2048,) tensor
            caption: List of word indices
            img_name: Image filename (for debugging)
        """
        img_name, caption_indices = self.samples[idx]
        
        # Get image features
        image_features = torch.tensor(self.features_dict[img_name], dtype=torch.float32)
        
        # Convert caption to tensor
        caption = torch.tensor(caption_indices, dtype=torch.long)
        
        return image_features, caption, img_name


def collate_fn(batch):
    """
    Custom collate function to pad captions to same length
    
    Args:
        batch: List of (image_features, caption, img_name) tuples
    
    Returns:
        image_features: (batch_size, 2048)
        captions_padded: (batch_size, max_len)
        lengths: List of caption lengths
        img_names: List of image names
    """
    # Sort batch by caption length (descending) for packed sequences
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    
    image_features, captions, img_names = zip(*batch)
    
    # Stack image features
    image_features = torch.stack(image_features, dim=0)
    
    # Get caption lengths
    lengths = [len(cap) for cap in captions]
    
    # Pad captions
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    
    return image_features, captions_padded, lengths, img_names


# ============================================================================
# STEP 3: TRAIN/VAL SPLIT
# ============================================================================

# Shuffle and split data
random.seed(42)
random.shuffle(common_images)

train_size = int(0.8 * len(common_images))
val_size = int(0.1 * len(common_images))

train_images = common_images[:train_size]
val_images = common_images[train_size:train_size + val_size]
test_images = common_images[train_size + val_size:]

print(f"\n{'='*60}")
print("DATA SPLIT:")
print(f"{'='*60}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# Create datasets
train_dataset = Flickr30kDataset(train_images, features_dict, numericalized_captions, vocab)
val_dataset = Flickr30kDataset(val_images, features_dict, numericalized_captions, vocab)
test_dataset = Flickr30kDataset(test_images, features_dict, numericalized_captions, vocab)

# Create dataloaders
BATCH_SIZE = 64
NUM_WORKERS = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"\n✓ Created dataloaders:")
print(f"  - Training batches: {len(train_loader)}")
print(f"  - Validation batches: {len(val_loader)}")
print(f"  - Test batches: {len(test_loader)}")


# ============================================================================
# STEP 4: IMPORT MODEL FROM PART 3
# ============================================================================

# Import the model classes from Part 3
# (Assuming you've run Part 3 and classes are defined)

#from part3_seq2seq_architecture import ImageCaptioningModel

# Initialize model
model = ImageCaptioningModel(
    vocab_size=vocab_size,
    embed_size=model_config['embed_size'],
    hidden_size=model_config['hidden_size'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout']
).to(device)

print(f"\n✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================================
# STEP 5: LOSS FUNCTION AND OPTIMIZER
# ============================================================================

# Loss function: CrossEntropyLoss with ignore_index for padding
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Optimizer: Adam with weight decay
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler: Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    #verbose=True
)

print(f"\n✓ Loss function: CrossEntropyLoss (ignore_index={pad_idx})")
print(f"✓ Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"✓ Scheduler: ReduceLROnPlateau")

# ============================================================================
# STEP 6: TRAINING FUNCTION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for image_features, captions, lengths, _ in progress_bar:
        # Move to device
        image_features = image_features.to(device)
        captions = captions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Input: all tokens except last (we don't predict after <end>)
        # Target: all tokens except first (we don't predict <start>)
        outputs = model(image_features, captions[:, :-1])
        
        # Reshape for loss calculation
        # outputs: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        outputs = outputs.reshape(-1, vocab_size)
        targets = captions[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# STEP 7: VALIDATION FUNCTION
# ============================================================================

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for image_features, captions, lengths, _ in tqdm(dataloader, desc="Validating"):
            # Move to device
            image_features = image_features.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(image_features, captions[:, :-1])
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, vocab_size)
            targets = captions[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# STEP 8: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 35
EARLY_STOP_PATIENCE = 5

# For tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0

print(f"\n{'='*60}")
print("STARTING TRAINING:")
print(f"{'='*60}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
print(f"{'='*60}\n")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'-'*60}")
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Print epoch summary
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': vocab,
            'model_config': model_config
        }, 'best_model.pth')
        
        print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s)")
    
    # Early stopping
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        break
    
    # Generate sample captions every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"\n{'='*60}")
        print("SAMPLE CAPTIONS:")
        print(f"{'='*60}")
        
        model.eval()
        sample_images = random.sample(val_images, min(3, len(val_images)))
        
        for img_name in sample_images:
            # Get features
            img_features = torch.tensor(features_dict[img_name]).float().unsqueeze(0).to(device)
            
            # Get ground truth
            ground_truth = vocab.denumericalize(numericalized_captions[img_name][0])
            
            # Generate caption
            with torch.no_grad():
                generated = model.generate_caption(img_features, method='beam', beam_size=3)
            
            print(f"\nImage: {img_name}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Generated:    {generated}")

print(f"\n{'='*60}")
print("✓ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Total epochs trained: {len(train_losses)}")

# ============================================================================
# STEP 9: PLOT TRAINING CURVES
# ============================================================================

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Training curve saved to 'training_curve.png'")

# ============================================================================
# STEP 10: LOAD BEST MODEL FOR EVALUATION
# ============================================================================

# Load best model
checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

# ============================================================================
# STEP 11: EVALUATION METRICS
# ============================================================================

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("⚠ Warning: Could not download NLTK data")

def calculate_metrics(model, dataloader, vocab, device, max_samples=None):
    """
    Calculate BLEU scores and other metrics
    
    Returns:
        metrics: Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    model.eval()
    
    references = []  # List of lists of reference captions
    hypotheses = []  # List of generated captions
    
    sample_count = 0
    
    with torch.no_grad():
        for image_features, captions, lengths, img_names in tqdm(dataloader, desc="Evaluating"):
            image_features = image_features.to(device)
            
            for i in range(len(img_names)):
                # Get all reference captions for this image
                img_name = img_names[i]
                refs = []
                
                if img_name in numericalized_captions:
                    for cap_indices in numericalized_captions[img_name]:
                        # Convert to words (skip special tokens)
                        ref_words = []
                        for idx in cap_indices:
                            if idx not in [start_idx, end_idx, pad_idx]:
                                if idx in vocab.idx2word:
                                    ref_words.append(vocab.idx2word[idx])
                        refs.append(ref_words)
                
                if not refs:
                    continue
                
                # Generate caption
                img_feat = image_features[i:i+1]
                generated_caption = model.generate_caption(img_feat, method='beam', beam_size=3)
                hyp_words = generated_caption.split()
                
                references.append(refs)
                hypotheses.append(hyp_words)
                
                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    metrics = {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'num_samples': sample_count
    }
    
    return metrics, references, hypotheses

# Calculate metrics on test set
print(f"\n{'='*60}")
print("CALCULATING METRICS ON TEST SET:")
print(f"{'='*60}")

test_metrics, test_refs, test_hyps = calculate_metrics(
    model, test_loader, vocab, device, max_samples=500
)

print(f"\nTest Set Metrics (on {test_metrics['num_samples']} samples):")
print(f"  BLEU-1: {test_metrics['bleu1']:.4f}")
print(f"  BLEU-2: {test_metrics['bleu2']:.4f}")
print(f"  BLEU-3: {test_metrics['bleu3']:.4f}")
print(f"  BLEU-4: {test_metrics['bleu4']:.4f}")

# ============================================================================
# STEP 12: GENERATE SAMPLE CAPTIONS FOR VISUALIZATION
# ============================================================================

def visualize_predictions(model, image_names, features_dict, captions_dict, vocab, device, num_samples=10):
    """
    Generate and display sample predictions
    """
    model.eval()
    
    samples = random.sample(image_names, min(num_samples, len(image_names)))
    
    results = []
    
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS:")
    print(f"{'='*60}")
    
    for img_name in samples:
        # Get features
        img_features = torch.tensor(features_dict[img_name]).float().unsqueeze(0).to(device)
        
        # Get ground truth captions
        ground_truths = []
        for cap_indices in captions_dict[img_name]:
            gt = vocab.denumericalize(cap_indices)
            ground_truths.append(gt)
        
        # Generate captions
        with torch.no_grad():
            greedy_caption = model.generate_caption(img_features, method='greedy')
            beam_caption = model.generate_caption(img_features, method='beam', beam_size=5)
        
        # Store results
        result = {
            'image': img_name,
            'ground_truth': ground_truths,
            'greedy': greedy_caption,
            'beam': beam_caption
        }
        results.append(result)
        
        # Print
        print(f"\n📷 Image: {img_name}")
        print(f"   Ground Truth:")
        for i, gt in enumerate(ground_truths[:3], 1):
            print(f"     {i}. {gt}")
        print(f"   Generated (Greedy): {greedy_caption}")
        print(f"   Generated (Beam):   {beam_caption}")
    
    return results

# Generate samples
sample_results = visualize_predictions(
    model, test_images, features_dict, numericalized_captions, vocab, device, num_samples=10
)

# ============================================================================
# STEP 13: SAVE RESULTS
# ============================================================================

# Save evaluation results
evaluation_results = {
    'test_metrics': test_metrics,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'sample_results': sample_results,
    'best_val_loss': best_val_loss
}

with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(evaluation_results, f)

print(f"\n✓ Evaluation results saved to 'evaluation_results.pkl'")

# ============================================================================
# STEP 14: CREATE INFERENCE FUNCTION FOR DEPLOYMENT
# ============================================================================

def generate_caption_from_image(image_path, model, device, method='beam', beam_size=3):
    """
    Complete pipeline: Load image → Extract features → Generate caption
    
    For deployment with Gradio
    """
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    
    # Load ResNet50 for feature extraction
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
    resnet = resnet.to(device)
    resnet.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path.convert('RGB')  # Already PIL image
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = resnet(image_tensor)
        features = features.squeeze().cpu()
    
    # Move to device
    features = features.unsqueeze(0).to(device)
    
    # Generate caption
    model.eval()
    with torch.no_grad():
        caption = model.generate_caption(features, method=method, beam_size=beam_size)
    
    return caption

print(f"\n{'='*60}")
print("✓ PART 4 COMPLETE!")
print(f"{'='*60}")
print("\nFiles created:")
print("  - best_model.pth")
print("  - training_curve.png")
print("  - evaluation_results.pkl")
print("\nMetrics:")
print(f"  - Best validation loss: {best_val_loss:.4f}")
print(f"  - Test BLEU-4: {test_metrics['bleu4']:.4f}")
print("\nNext step: Deploy with Gradio!")