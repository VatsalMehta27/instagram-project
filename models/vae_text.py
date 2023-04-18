import torch
from torch import nn
from models.time_distributed import TimeDistributed
import numpy as np


class TextVAE(nn.Module):
    def __init__(self, device, hidden_dim, latent_size, text_embeddings, padding_index, num_classes):
        super(TextVAE, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.padding_index = padding_index
        self.num_classes = num_classes

        self.vocab_size = text_embeddings.shape[0] + 1
        self.text_embedding_size = text_embeddings.shape[1]
        
        print(self.vocab_size)
        print(padding_index)

        # Generate Text Feature Representation

        # Creates a layer using the word emebedding vectors learned from the Word2Vec model
        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.text_embedding_size, padding_idx=self.padding_index
        ).from_pretrained(embeddings=torch.FloatTensor(np.vstack((text_embeddings, np.zeros(self.text_embedding_size)))), freeze=True)

        # Stacked Bi-directional LSTM layers
        self.stacked_bi_lstm_1 = nn.LSTM(
                self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True
            )
        # nn.Tanh(),
        self.stacked_bi_lstm_2 = nn.LSTM(
                self.hidden_dim * 2,
                self.hidden_dim,
                bidirectional=True,
                batch_first=True,
            )
            # nn.Tanh(),

        # Fully connected layer that produces the final text feature representation
        self.enc_text_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
        )

        # Fully connected layer that takes the concatenated feature represenation (text + img)
        # to produce the shared representation
        self.latent_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_size),
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
            nn.Linear(self.latent_size, self.hidden_dim), nn.Tanh()
        )

        self.dec_stacked_bi_lstm_1 = nn.LSTM(
                    self.hidden_dim,
                    self.hidden_dim,
                    bidirectional=True,
                    batch_first=True,
                )
                # nn.Tanh(),
        # self.dec_stacked_bi_lstm_2 = TimeDistributed(
            # nn.Sequential(
        self.dec_stacked_bi_lstm_2 = nn.LSTM(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    bidirectional=True,
                    batch_first=True,
                )
                # nn.Tanh(),
            # )
        # )

         # nn.Sequential(
        self.dec_text = nn.Linear(
                self.hidden_dim*2,
                self.vocab_size
            )
            # nn.Softmax(dim=-1)
        # )

        # Binary Classifier
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.Tanh(),
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, text_input):
        # Encoder Layers
        
        batch_size = text_input.shape[0]
        sequence_length = text_input.shape[1]

        word_embeddings = self.word_embeddings(text_input)
        bi_lstm_result_1, _ = self.stacked_bi_lstm_1(word_embeddings)
        bi_lstm_result_2, _ = self.stacked_bi_lstm_2(bi_lstm_result_1)
        text_feature_rep = self.enc_text_fc(bi_lstm_result_2)

        encoder_result = self.latent_fc(text_feature_rep).mean(dim=1)

        mu = self.mu_layer(encoder_result)
        logvar = self.logvar_layer(encoder_result)

        sampled_latent_vector = self.reparametrize(mu, logvar)

        # Classification
        classifier_result = self.binary_classifier(sampled_latent_vector)

        # Decoder Layers
        
        dec_text_fc_result = self.dec_text_fc(sampled_latent_vector)
        repeated_context = dec_text_fc_result.unsqueeze(1).repeat(1, sequence_length, 1)

        dec_stacked_bi_lstm_result_1, _ = self.dec_stacked_bi_lstm_1(repeated_context)
        dec_stacked_bi_lstm_result_2, _ = self.dec_stacked_bi_lstm_2(dec_stacked_bi_lstm_result_1)
        decoded_text = self.dec_text(dec_stacked_bi_lstm_result_2)

        return decoded_text, mu, logvar, classifier_result

    def reparametrize(self, mu, logvar):
        std_gauss_sample = torch.randn(size=mu.shape).to(self.device)

        sampled_latent_vector = mu + (logvar * std_gauss_sample)
        
        return sampled_latent_vector

    def loss_function(
        self, text, post_class, decoded_text, mu, logvar, predicted_class
    ):
        one_hot_input_text = nn.functional.one_hot(text, num_classes = self.vocab_size).float().to(self.device)
        one_hot_input_text[text == self.padding_index] = -100
        text_recon_loss = nn.CrossEntropyLoss(ignore_index=-100)(torch.flatten(decoded_text), torch.flatten(one_hot_input_text))

        kl_divergence = (0.5 * torch.sum(torch.square(mu) + torch.exp(logvar) - logvar - 1))
        
        one_hot_post_class = nn.functional.one_hot(post_class, num_classes=self.num_classes).float().to(self.device)
        binary_classifier_loss = nn.CrossEntropyLoss()(torch.flatten(predicted_class), torch.flatten(one_hot_post_class))

        # print(text_recon_loss)
        # print(kl_divergence)
        # print(binary_classifier_loss)

        loss = (
            text_recon_loss + kl_divergence + binary_classifier_loss
        ) / text.shape[0]
        
        # print(loss)

        return loss
