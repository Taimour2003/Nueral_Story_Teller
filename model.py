import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# VOCABULARY CLASS (ADD THIS!)
# ============================================================================

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        Vocabulary class for word-to-index and index-to-word mapping
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
        self.idx = 4
    
    def __len__(self):
        return len(self.word2idx)
    
    def denumericalize(self, indices):
        """
        Convert list of indices back to text
        """
        if isinstance(indices, int):
            indices = [indices]
        
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                # Skip special tokens
                if token not in [self.start_token, self.end_token, self.pad_token]:
                    tokens.append(token)
        
        return ' '.join(tokens)

# ============================================================================
# MODEL CLASSES (YOUR EXISTING CODE)
# ============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, resnet_feature_size=2048, hidden_size=512, dropout=0.5):
        super(ImageEncoder, self).__init__()
        self.fc = nn.Linear(resnet_feature_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, image_features):
        return self.dropout(self.relu(self.bn(self.fc(image_features))))


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, 
                 num_layers=1, dropout=0.5, pad_idx=0, start_idx=1, end_idx=2):
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
    
    def forward(self, captions, hidden_state):
        embeddings = self.dropout(self.embedding(captions))
        h0 = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        return self.fc(self.dropout(lstm_out))
    
    def generate_caption(self, hidden_state, vocab, max_length=50, method='beam', beam_size=3):
        if method == 'greedy':
            return self._greedy_search(hidden_state, vocab, max_length)
        return self._beam_search(hidden_state, vocab, max_length, beam_size)
    
    def _greedy_search(self, hidden_state, vocab, max_length):
        device = hidden_state.device
        result = []
        input_word = torch.tensor([self.start_idx]).unsqueeze(0).to(device)
        h = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        for _ in range(max_length):
            lstm_out, (h, c) = self.lstm(self.embedding(input_word), (h, c))
            predicted = self.fc(lstm_out.squeeze(1)).argmax(dim=1)
            word_idx = predicted.item()
            if word_idx == self.end_idx:
                break
            result.append(word_idx)
            input_word = predicted.unsqueeze(1)
        
        return vocab.denumericalize(result)
    
    def _beam_search(self, hidden_state, vocab, max_length, beam_size):
        device = hidden_state.device
        h = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        beams = [([self.start_idx], 0.0, h, c)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            for seq, score, h_state, c_state in beams:
                if seq[-1] == self.end_idx:
                    completed.append((seq, score))
                    continue
                
                input_word = torch.tensor([[seq[-1]]]).to(device)
                lstm_out, (h_new, c_new) = self.lstm(self.embedding(input_word), (h_state, c_state))
                log_probs = F.log_softmax(self.fc(lstm_out.squeeze(1)), dim=1)
                topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    candidates.append((seq + [topk_indices[0, i].item()], 
                                     score + topk_log_probs[0, i].item(), h_new, c_new))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(seq[-1] == self.end_idx for seq, _, _, _ in beams):
                completed.extend([(seq, score) for seq, score, _, _ in beams])
                break
        
        if not completed:
            completed = [(seq, score) for seq, score, _, _ in beams]
        
        best_seq = max(completed, key=lambda x: x[1])[0]
        indices = [idx for idx in best_seq if idx not in [self.start_idx, self.end_idx, self.pad_idx]]
        return vocab.denumericalize(indices)


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, 
                 num_layers=1, dropout=0.5, pad_idx=0, start_idx=1, end_idx=2):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(2048, hidden_size, dropout)
        self.decoder = CaptionDecoder(vocab_size, embed_size, hidden_size, 
                                     num_layers, dropout, pad_idx, start_idx, end_idx)
    
    def forward(self, image_features, captions):
        return self.decoder(captions, self.encoder(image_features))
    
    def generate_caption(self, image_features, vocab, max_length=50, method='beam', beam_size=3):
        self.eval()
        with torch.no_grad():
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            hidden_state = self.encoder(image_features)
            return self.decoder.generate_caption(hidden_state, vocab, max_length, method, beam_size)