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

        # Generate Text Feature Representation

        # Creates a layer using the word emebedding vectors learned from the Word2Vec model
        self.word_embeddings = nn.Embedding(text_embeddings.shape[0], text_embeddings.shape[1])
        self.word_embeddings.weight = nn.Parameter(text_embeddings)
        self.word_embeddings.requires_grad = False

        # Stacked Bi-directional LSTM layers
        self.stacked_bi_lstm = nn.Sequential(
            nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True),
            nn.Tanh(),
            nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True),
            nn.Tanh(),
        )

        # Fully connected layer that produces the final text feature representation
        self.text_fc = nn.Sequntial(
            nn.Linear(self.hidden_dim),
            nn.Tanh(),
        )

        # Generate Image Feature Representation

        # Takes the VGG-19 image embeddings and passes through 2 fully connected layers 
        # to produce the final image feature representation
        self.gen_img_feature_rep = nn.Sequential(
            nn.Linear(self.img_embedding_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )

        # Fully connected layer that takes the concatenated feature represenation (text + img)
        # to produce the shared representation
        self.latent_fc = nn.Sequential(
            nn.Linear(2*self.hidden_dim, 2*self.hidden_dim),
            nn.Tanh(),
        )


        self.mu_layer = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.latent_size), 
            nn.ReLU(),
        )
        self.logvar_layer = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.latent_size), 
            nn.ReLU(),
        )

        self.decoder = None

    def forward(self, text_input, img_input):
        word_embeddings = self.word_embeddings(text_input)
        bi_lstm_result = self.stacked_bi_lstm(word_embeddings)
        text_feature_rep = self.text_fc(bi_lstm_result)
        print("Text Feature Rep:", text_feature_rep.shape)
        
        img_feature_rep = self.gen_img_feature_rep(img_input)
        print("Img Feature Rep:", img_feature_rep.shape)

        encoder_result = self.latent_fc(torch.cat([text_feature_rep, img_feature_rep], dim=-1))

        mu = self.mu_layer(encoder_result)
        logvar = self.logvar_layer(encoder_result)

        sampled_latent_vector = self.reparametrize(mu, logvar)

        decoder_result = self.decoder(latent_vector)

    def reparametrize(mu, logvar):
        std_gauss_sample = torch.randn(size=mu.shape).to(device)

        sampled_latent_vector = mu + (logvar * std_gauss_sample)

        return sampled_latent_vector

    def loss_function(text, img, decoded_text, decoded_img, mu, logvar):
        img_recon_loss = nn.MSELoss()(img, decoded_img)

        text_recon_loss = nn.CrossEntropyLoss()(text, decoded_text)

        kl_divergence = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))

        print(img_recon_loss + text_recon_loss + kl_divergence)

        assert decoded_text.shape[0] == decoded_img.shape[0]

        loss = (img_recon_loss + text_recon_loss + kl_divergence) / decoded_text.shape[0]

        return loss