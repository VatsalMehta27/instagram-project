import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")


class ImageVAE(nn.Module):
    def __init__(self, device, hidden_dim, latent_size, img_embedding_size, num_classes):
        super(ImageVAE, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.img_embedding_size = img_embedding_size
        self.num_classes = num_classes

        # Generate Image Feature Representation

        # Takes the VGG-19 image embeddings and passes through 2 fully connected layers
        # to produce the final image feature representation
        self.gen_img_feature_rep = nn.Sequential(
            nn.Linear(self.img_embedding_size, self.img_embedding_size // 4),
            nn.Tanh(),
            nn.Linear(self.img_embedding_size // 4, self.hidden_dim),
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
            nn.Sigmoid(),
        )

        # Binary Classifier
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.Tanh(),
            nn.Linear(self.latent_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, img_input):
        # Encoder Layers

        img_feature_rep = self.gen_img_feature_rep(img_input)
        # print("Img Feature Rep:", img_feature_rep.shape)

        encoder_result = self.latent_fc(img_feature_rep)

        mu = self.mu_layer(encoder_result)
        logvar = self.logvar_layer(encoder_result)

        sampled_latent_vector = self.reparametrize(mu, logvar)

        # Classification
        classifier_result = self.binary_classifier(sampled_latent_vector)

        # Decode Layers

        decoded_img = self.recon_img(sampled_latent_vector)

        return decoded_img, mu, logvar, classifier_result

    def reparametrize(self, mu, logvar):
        std_gauss_sample = torch.randn(size=mu.shape).to(self.device)

        sampled_latent_vector = mu + (logvar * std_gauss_sample)

        return sampled_latent_vector

    def loss_function(self, img, post_class, decoded_img, mu, logvar, predicted_class):
        img_recon_loss = nn.MSELoss()(img, decoded_img)

        kl_divergence = (0.5 * torch.sum(torch.square(mu) + torch.exp(logvar) - logvar - 1))
        
        one_hot_post_class = nn.functional.one_hot(post_class, num_classes=self.num_classes).float().to(self.device)
        binary_classifier_loss = nn.CrossEntropyLoss()(torch.flatten(predicted_class), torch.flatten(one_hot_post_class))

        # print(img_recon_loss + kl_divergence)
        # print(binary_classifier_loss)

        loss = (
            img_recon_loss + kl_divergence + binary_classifier_loss
        ) / decoded_img.shape[0]

        return loss
