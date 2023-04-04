import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    def __init__(self, processed_text, word_index_mapping, post_classes):
        self.processed_text = processed_text
        self.post_classes = post_classes

        self.max_length = max([len(desc) for desc in self.processed_text])
        self.padding_index = max(word_index_mapping.values()) + 1

        self.encoded_text = list(
            [
                list([word_index_mapping[word] for word in description])
                for description in self.processed_text
            ]
        )
        self.encoded_text = torch.LongTensor(np.array([
            np.pad(
                desc,
                (0, self.max_length - len(desc)),
                "constant",
                constant_values=(self.padding_index,),
            )
            for desc in self.encoded_text
        ]))
        
        self.mapping = {k: v for (v, k) in enumerate(list(set(self.post_classes)))}
        self.num_post_classes = len(self.mapping)
        self.encoded_post_classes = torch.tensor([self.mapping[i] for i in self.post_classes])

        assert len(self.post_classes) == len(self.encoded_text)

    def __len__(self):
        return len(self.post_classes)

    def __getitem__(self, idx):
        return self.encoded_text[idx], self.encoded_post_classes[idx]
