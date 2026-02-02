import json
import torch
from torch.utils.data import Dataset
from src.utils.text_preprocessor import TextProcessor

class PhoMTDataset(Dataset):
    """
    Dataset for English-Vietnamese Translation.
    Each item contains source_ids (English) and target_ids (Vietnamese).
    """
    def __init__(self, file_path, src_vocab=None, tgt_vocab=None, max_len=50):
        self.processor = TextProcessor()
        self.max_len = max_len

        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # Pre-tokenize
        self.src_sents = [self.processor.preprocess(item['english']).split() for item in self.raw_data]
        self.tgt_sents = [self.processor.preprocess(item['vietnamese']).split() for item in self.raw_data]

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # Source (English): Add SOS and EOS? Usually only EOS is enough for encoder
        src_ids = self.src_vocab.encode(self.src_sents[idx], max_len=self.max_len, add_eos=True)
        # Target (Vietnamese): MUST have SOS (for decoder input) and EOS (for target)
        tgt_ids = self.tgt_vocab.encode(self.tgt_sents[idx], max_len=self.max_len, add_sos=True, add_eos=True)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }

if __name__ == "__main__":
    # Mock test: Requires actual file or mock JSON
    print("Dataset logic ready for English-Vietnamese pairs.")