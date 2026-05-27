# dataset.py
import torch
from tqdm import tqdm

class DatasetIterator:
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.n_batches + (1 if self.residue else 0)

def build_dataset(config, tokenizer):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin: continue
                try:
                    content, label = lin.split('\t')
                    token = tokenizer.tokenize(content)
                    token = ['[CLS]'] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = []
                    
                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token) + [0] * (pad_size - len(token))
                            token_ids = tokenizer.convert_tokens_to_ids(token)
                            token_ids += [0] * (pad_size - len(token))
                        else:
                            mask = [1] * pad_size
                            token_ids = tokenizer.convert_tokens_to_ids(token[:pad_size])
                            seq_len = pad_size
                    contents.append((token_ids, int(config.class_list.index(label)), seq_len, mask))
                except:
                    continue
        return contents

    print("Loading Train...")
    train = load_dataset(config.train_path, config.pad_size)
    print("Loading Dev...")
    dev = load_dataset(config.dev_path, config.pad_size)
    print("Loading Test...")
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test