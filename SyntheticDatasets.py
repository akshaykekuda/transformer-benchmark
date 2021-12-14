from torch.utils.data import Dataset
import torch

class SyntheticDataset(Dataset):

  def __init__(self, vocab_size, num_examples, num_labels, max_seq_len, out_max_seq_len):
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.out_max_seq_len = out_max_seq_len
  
  def __len__(self):
    return self.num_examples

  def __getitem__(self, idx):
        i = torch.randint(self.vocab_size, (self.max_seq_len,))
        l = torch.randint(self.num_labels, (self.out_max_seq_len,))
        sample = {"input":i, "labels":l}
        return sample

class SyntheticMCQDataset(SyntheticDataset):

  def __init__(self, vocab_size, num_examples, num_labels, max_seq_len):
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len

  def __getitem__(self, idx):
        i = torch.randint(self.vocab_size, (self.num_labels, self.max_seq_len))
        l = torch.randint(self.num_labels, (1,)).squeeze()
        sample = {"input":i, "labels":l}
        return sample