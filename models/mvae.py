import torch
from torch import nn
from time_distributed import TimeDistributed

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

class MVAE(nn.Module):
    def __init__(self, device, hidden_dim, latent_size, text_embeddings, img_embedding_size):
        super(MVAE, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.img_embedding_size = img_embedding_size
        self.vocab_size = text_embeddings.shape[0]
        self.text_embedding_size = text_embeddings.shape[1]

        # Generate Text Feature Representation

        # Creates a layer using the word emebedding vectors learned from the Word2Vec model
        self.word_embeddings = nn.Embedding(self.vocab_size, self.text_embedding_size).from_pretrained(embeddings=torch.FloatTensor(text_embeddings), freeze=True)

        # Stacked Bi-directional LSTM layers
        self.stacked_bi_lstm = nn.Sequential(
            nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True),
            nn.Tanh(),
            nn.LSTM(self.hidden_dim*2, self.hidden_dim, bidirectional=True, batch_first=True),
            nn.Tanh(),
        )

        # Fully connected layer that produces the final text feature representation
        self.enc_text_fc = nn.Sequntial(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.Tanh(),
        )

        # Generate Image Feature Representation

        # Takes the VGG-19 image embeddings and passes through 2 fully connected layers 
        # to produce the final image feature representation
        self.gen_img_feature_rep = nn.Sequential(
            nn.Linear(self.img_embedding_size, self.img_embedding_size // 4),
            nn.Tanh(),
            nn.Linear(self.img_embedding_size // 4, self.hidden_dim),
            nn.Tanh()
        )

        # Fully connected layer that takes the concatenated feature represenation (text + img)
        # to produce the shared representation
        self.latent_fc = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.latent_size),
            nn.Tanh(),
        )


        self.mu_layer = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size), 
            nn.ReLU(),
        )
        self.logvar_layer = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size), 
            nn.ReLU(),
        )

        # Text Reconstruction

        self.dec_text_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.Tanh()
        )

        self.dec_stacked_bi_lstm = TimeDistributed(
            nn.Sequential(
                nn.LSTM(self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True),
                nn.Tanh(),
                nn.LSTM(self.hidden_dim*2, self.hidden_dim, bidirectional=True, batch_first=True),
                nn.Tanh(),
            )
        )

        self.dec_text = nn.Softmax()

        # Image Reconstruction
        self.recon_img = nn.Sequential(
            # dec_vis_fc1
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.Tanh(),
            # dec_vis_fc2
            nn.Linear(self.hidden_dim, self.img_embedding_size // 4),
            nn.Tanh(),
            # final fc layer
            nn.Linear(self.img_embedding_size // 4, self.img_embedding_size),
            nn.Sigmoid()
        )

        # Binary Classifier
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.Tanh(),
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_input, img_input):
        # Encoder Layers

        word_embeddings = self.word_embeddings(text_input)
        bi_lstm_result, _ = self.stacked_bi_lstm(word_embeddings)
        print(bi_lstm_result.shape)
        text_feature_rep = self.enc_text_fc(bi_lstm_result)
        print("Text Feature Rep:", text_feature_rep.shape)
        
        img_feature_rep = self.gen_img_feature_rep(img_input)
        print("Img Feature Rep:", img_feature_rep.shape)

        encoder_result = self.latent_fc(torch.cat([text_feature_rep, img_feature_rep], dim=-1))

        mu = self.mu_layer(encoder_result)
        logvar = self.logvar_layer(encoder_result)

        sampled_latent_vector = self.reparametrize(mu, logvar)

        # Classification
        classifier_result = self.binary_classifier(sampled_latent_vector)

        # Decoder Layers
        dec_text_fc_result = self.dec_text_fc(sampled_latent_vector)
        dec_text_fc_result = dec_text_fc_result.repeat(self.text_embedding_size).view(self.text_embedding_size, -1)
        dec_stacked_bi_lstm_result, _ = self.dec_stacked_bi_lstm(dec_text_fc_result)
        decoded_text = self.dec_text(dec_stacked_bi_lstm_result)

        decoded_img = self.recon_img(sampled_latent_vector)

        return decoded_text, decoded_img, mu, logvar, classifier_result


    def reparametrize(mu, logvar):
        std_gauss_sample = torch.randn(size=mu.shape).to(self.device)

        sampled_latent_vector = mu + (logvar * std_gauss_sample)

        return sampled_latent_vector

    def loss_function(text, img, post_class, decoded_text, decoded_img, mu, logvar, predicted_class):
        img_recon_loss = nn.MSELoss()(img, decoded_img)

        text_recon_loss = nn.CrossEntropyLoss()(text, decoded_text)

        kl_divergence = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))

        binary_classifier_loss = nn.BCELoss()(post_class, predicted_class)

        print(img_recon_loss + text_recon_loss + kl_divergence)
        print(binary_classifier_loss)

        assert decoded_text.shape[0] == decoded_img.shape[0]

        loss = (img_recon_loss + text_recon_loss + kl_divergence + binary_classifier_loss) / decoded_text.shape[0]

        return loss