import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Load dataset
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.split("=", 1)[-1].strip()
    return eval(content)

dataset_path = r"D:\McCabe\translate.txt"
dataset = load_dataset(dataset_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenization
class Vocabulary:
    def __init__(self, sentences):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.build_vocab(sentences)
    
    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence.lower().split())
        for word in counter:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, sentence):
        return [self.word2idx.get(word, 3) for word in sentence.lower().split()] + [2]
    
    def decode(self, indices):
        return ' '.join([self.idx2word[idx] for idx in indices if idx not in {0, 1, 2}])

# Build vocab
english_sentences, french_sentences = zip(*dataset)
eng_vocab = Vocabulary(english_sentences)
fr_vocab = Vocabulary(french_sentences)

# Convert dataset to indexed format
pairs = [(eng_vocab.encode(en), fr_vocab.encode(fr)) for en, fr in dataset]

# Pad sequences
def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

max_eng_len = max(len(p[0]) for p in pairs)
max_fr_len = max(len(p[1]) for p in pairs)
pairs = [(pad_sequence(en, max_eng_len), pad_sequence(fr, max_fr_len)) for en, fr in pairs]

# Train-test split
train_data, val_data = train_test_split(pairs, test_size=0.2, random_state=42)

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

train_loader = DataLoader(TranslationDataset(train_data), batch_size=32, shuffle=True)
val_loader = DataLoader(TranslationDataset(val_data), batch_size=32, shuffle=False)

# GRU Encoder-Decoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x).unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        hidden = self.encoder(src)
        outputs = []
        dec_input = torch.tensor([1] * src.size(0)).to(src.device)  # <SOS>
        for _ in range(trg.size(1)):
            output, hidden = self.decoder(dec_input, hidden)
            outputs.append(output.unsqueeze(1))
            dec_input = output.argmax(1)
        return torch.cat(outputs, dim=1)

# Initialize model
embed_size = 256
hidden_size = 512
encoder = Encoder(len(eng_vocab.word2idx), embed_size, hidden_size)
decoder = Decoder(len(fr_vocab.word2idx), embed_size, hidden_size)
model = Seq2Seq(encoder, decoder).to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg)
            loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
            total_loss += loss.item()
            
            preds = output.argmax(dim=2)
            mask = (trg != 0)
            correct += (preds == trg) * mask
            total += mask.sum().item()
    
    avg_val_loss = total_loss / len(val_loader)
    val_acc = (correct.sum().item() / total) * 100
    return avg_val_loss, val_acc

def train(model, train_loader, val_loader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

# Start training
train(model, train_loader, val_loader, epochs=100)
