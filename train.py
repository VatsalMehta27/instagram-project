import pandas as pd
# from preprocessing.preprocess_text import process_posts
from preprocessing.preprocess_text import clean_text
from preprocessing.gen_text_embeddings import generate_text_embeddings
import torch
from pytorch_datasets.text_dataset import TextDataset
from torch.utils.data import DataLoader, random_split
from models.vae_text import TextVAE
from torch.optim import Adam

print("Loading data...")
instagram_data = pd.read_csv("data/instagram_data.csv")
data = instagram_data.dropna(subset=["description"])
print(f"Removed {len(instagram_data) - len(data)} rows due to N/A descriptions.")

print("Processing post descriptions...")
# post_descriptions = process_posts(data["description"].tolist())

post_descriptions = data["description"].apply(lambda text: clean_text(text) if type(text) == str else text).tolist()
post_classes = data["Party"]

print("Generating text embeddings...")
text_embeddings, word_index_mapping = generate_text_embeddings(post_descriptions, 32)

print("Preparing datasets...")
dataset = TextDataset(post_descriptions, word_index_mapping, post_classes)

# Start of abstractable code

train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

num_epochs = 10
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

model = TextVAE(device, 32, 32, text_embeddings).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_idx, (text, party) in enumerate(train_loader):
        text = text.to(device)

        decoded_text, mu, logvar, classifier_result = model(text)

        optimizer.zero_grad()

        loss = model.loss_function(text, party, decoded_text, mu, logvar, classifier_result)
        loss.backward()

        train_loss += loss.data()
        optimizer.step()

    print(f"Train Epoch: {epoch} \tLoss: {loss.data:.6f}")