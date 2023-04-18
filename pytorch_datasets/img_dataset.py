import torch
from torch.utils.data import Dataset
from preprocessing.get_image_embeddings import get_image_embeddings
import numpy as np
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, processed_text, word_index_mapping, image_usernames, vgg_file_ids, vgg_encoding_folder_path, post_classes):
        vgg_file_names = list([f"{x}.jpg" for x in vgg_file_ids])
        usernames_with_image_files = pd.DataFrame(list(zip(image_usernames, vgg_file_names, post_classes)), columns=["username", "photo_name", "post_classes"])
        
        vgg_encodings = get_image_embeddings(vgg_encoding_folder_path)
        vgg_encodings["flattenPhoto"] = vgg_encodings["flattenPhoto"].apply(np.asarray)
        
        no_image_encodings = vgg_encodings.index[vgg_encodings["flattenPhoto"].apply(lambda arr: arr.size) == 0].tolist()
        # print(no_image_encodings)
        vgg_encodings = vgg_encodings.drop(no_image_encodings).reset_index()
        
        self.matched_vgg_encodings = usernames_with_image_files.merge(vgg_encodings, on=["username", "photo_name"], how="inner")
        print(f"{len(vgg_file_names) - len(self.matched_vgg_encodings)} of data removed because matching VGG image embeddings not found.")
        
        # self.image_usernames = self.matched_vgg_encodings["username"].tolist()
        self.image_embeddings = self.matched_vgg_encodings["flattenPhoto"].tolist()
        self.post_classes = self.matched_vgg_encodings["post_classes"].tolist()
        
        self.mapping = {k: v for (v, k) in enumerate(list(set(self.post_classes)))}
        self.num_post_classes = len(self.mapping)
        self.encoded_post_classes = torch.tensor([self.mapping[i] for i in self.post_classes])

        assert len(self.encoded_post_classes) == len(self.image_embeddings)

    def __len__(self):
        return len(self.post_classes)

    def __getitem__(self, idx):
        # image_username = self.image_usernames[idx]
        # image_encoding = self.vgg_encodings.loc[(self.vgg_encodings['username'] == image_username) & (self.vgg_encodings['photo_name'] == image_id), "flattenPhoto"]
        
#         print(image_username, image_id, image_encoding)
        
        return self.image_embeddings[idx], self.encoded_post_classes[idx]
