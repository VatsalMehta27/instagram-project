import torch
from torch.utils.data import Dataset
from preprocessing.get_image_embeddings import get_image_embeddings
import numpy as np
import pandas as pd


class MVAEDataset(Dataset):
    def __init__(self, processed_text, word_index_mapping, image_usernames, vgg_file_ids, vgg_encoding_folder_path, post_classes):     
        # Image processing
        
        vgg_file_names = list([f"{x}.jpg" for x in vgg_file_ids])
        usernames_with_image_files = pd.DataFrame(list(zip(image_usernames, vgg_file_names, processed_text, post_classes)), columns=["username", "photo_name", "description", "post_classes"])
        
        vgg_encodings = get_image_embeddings(vgg_encoding_folder_path)
        vgg_encodings["flattenPhoto"] = vgg_encodings["flattenPhoto"].apply(np.asarray)
        
        no_image_encodings = vgg_encodings.index[vgg_encodings["flattenPhoto"].apply(lambda arr: arr.size) == 0].tolist()
        # print(no_image_encodings)
        vgg_encodings = vgg_encodings.drop(no_image_encodings).reset_index()
        
        self.matched_vgg_encodings = usernames_with_image_files.merge(vgg_encodings, on=["username", "photo_name"], how="inner")
        print(f"{len(vgg_file_names) - len(self.matched_vgg_encodings)} of data removed because matching VGG image embeddings not found.")
        
        self.image_embeddings = self.matched_vgg_encodings["flattenPhoto"].tolist()
        self.post_classes = self.matched_vgg_encodings["post_classes"].tolist()
        
        self.mapping = {k: v for (v, k) in enumerate(list(set(self.post_classes)))}
        self.num_post_classes = len(self.mapping)
        self.encoded_post_classes = torch.tensor([self.mapping[i] for i in self.post_classes])

        assert len(self.encoded_post_classes) == len(self.image_embeddings)
        
        # Text processing
        
        self.processed_text = self.matched_vgg_encodings["description"].tolist()

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

        assert len(self.encoded_post_classes) == len(self.encoded_text)

    def __len__(self):
        return len(self.encoded_post_classes)

    def __getitem__(self, idx):
        return self.encoded_text[idx], self.image_embeddings[idx], self.encoded_post_classes[idx]
