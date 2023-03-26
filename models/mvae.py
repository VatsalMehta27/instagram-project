import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

class MVAE(nn.Module):
    def __init__(self, hidden_dim=32, latent_size=64, text_embeddings, img_embedding_size=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.img_embedding_size = img_embedding_size

        self.word_embeddings = nn.Embedding(text_embeddings.shape[0], text_embeddings.shape[1])
        self.word_embeddings.weight = nn.Parameter(text_embeddings)
        self.word_embeddings.requires_grad = False

        self.stacked_bi_lstm = nn.Sequential(
            nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True),
            nn.Tanh(),
            nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True),
            nn.Tanh(),
        )

        self.text_fc = nn.Sequntial(
            nn.Linear(self.hidden_dim),
            nn.Tanh(),
        )

        self.gen_text_feature_rep = nn.Sequential(...)
        self.gen_img_feature_rep = nn.Sequential(
            nn.Linear(self.img_embedding_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )

        self.latent_fc = nn.Sequential(
            nn.Linear(2*self.hidden_dim, 2*self.hidden_dim),
            nn.Tanh(),
        )

        self.mu_layer = nn.Linear(2*self.hidden_dim, self.latent_size)
        self.var_layer = nn.Linear(2*self.hidden_dim, self.latent_size)

        self.decoder = None

        

    def forward(self, text_input, img_input):


        encoder_result = self.encoder(torch.cat([text_feature_rep, img_feature_rep], dim=1))



        decoder_result = self.decoder(latent_vector)
