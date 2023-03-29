from torch.utils.data import Dataset

class MVAEDataset(Dataset):
    def __init__(self, text, img_embeddings, post_classes):
        self.text = text
        self.img_embeddings = img_embeddings
        self.post_classes = post_classes

    def __len__(self):
        return len(self.post_classes)

    def __get_item__(self, idx):
        return self.text[idx], self.img_embeddings[idx], self.post_classes[idx]
