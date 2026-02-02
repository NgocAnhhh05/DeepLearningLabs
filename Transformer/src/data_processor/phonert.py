import json
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.text_preprocess import TextProcessor
from src.data.vocab import Vocab

class PhonerDataset(Dataset):
    def __init__(self, file_path, vocab:Vocab, label:str, max_len=100):
        super.__init__()
        self.vocab = vocab
        self.path = file_path
        self.label = label
        self.max_len = max_len
        self.data = []
        with open (file_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)

        sentences = [item['words'] for item in self.data]
        labels = [item[self.label] for item in self.data]
        self.vocab.build_label_vocab(labels)
        self.vocab.build_vocab(sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        words = item['words']
        labels = item[self.label]
        encoded_sentence = self.vocab.encode_sentence(words)
        encoded_label = self.vocab.encode_label(labels)
        return {
            "input_ids": encoded_sentence,
            "label": encoded_label
        }
    @staticmethod
    def collate_fn(samples):
        samples = {
            "input_ids": torch.stack([sample['input_ids'] for sample in samples], dim=0),
            "label": torch.stack([sample['label'] for sample in samples], dim=0)
        }
        return samples
if __name__ == "__main__":
    file_path = "../data/PhoNER_COVID19/word/train_word.json"
    dataset = PhonerDataset(file_path, Vocab(), "tags")
    loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    for item in enumerate(loader):
        print(item)
        break
