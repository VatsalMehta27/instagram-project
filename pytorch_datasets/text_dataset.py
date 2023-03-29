from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, processed_text, word_index_mapping, post_classes):
        self.processed_text = processed_text
        self.post_classes = post_classes

        self.encoded_text = [[word_index_mapping[word] for word in description] for description in self.processed_text]
        
        assert len(self.post_classes) == len(self.encoded_text)

    def __len__(self):
        return len(self.post_classes)

    def __get_item__(self, idx):
        return self.encoded_text[idx], self.post_classes[idx]
