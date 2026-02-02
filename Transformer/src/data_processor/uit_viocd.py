import json
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.text_preprocess import TextProcessor
from src.data.vocab import Vocab

class ViOCDDataset(Dataset):
    def __init__(self, file_path, vocab:Vocab, label:str, max_len=100):
        super().__init__()
        self.path = file_path
        self.label = label
        self.vocab = vocab
        self.max_len=max_len
        self.tokenizer = TextProcessor()
        with open(self.path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        sentences = [self.tokenizer.preprocess(item["sentence"]) for item in self.data]
        labels =[item[self.label] for item in self.data]
        self.vocab.build_label_vocab(labels)
        self.vocab.build_vocab(sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int):
        "Take as individual"
        item = self.data[index]
        sentence = item['sentence']
        labels = item[self.label]
        tokens = self.tokenizer.preprocess(sentence)

        encoded_sentence = self.vocab.encode_sentence(tokens=tokens)
        encoded_label = self.vocab.encode_label(label_input=labels)

        return {
            "input_ids": encoded_sentence,
            "label": encoded_label
        }

    @staticmethod
    def collate_fn(samples: list[dict]):
        "Stack individuals into a batch"
        samples = {
            "input_ids": torch.stack([sample["input_ids"] for sample in samples], dim=0),
            'label': torch.stack([sample['label'] for sample in samples], dim=0)
        }
        return samples

if __name__ == "__main__":
    file_path = "../data/uit-viocd/train.json"
    dataset = ViOCDDataset(file_path, Vocab(), 'topic')
    loader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)
    for i, item in enumerate(loader):
        print(item)
        break