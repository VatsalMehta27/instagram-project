### **Motivation/Contribution**

Goal: Multimodal VAE trained on image and text content of social media posts to identify features for detecting fake news

Model is able to identify *correlations across modalities*, which previous papers are unable to do

### **Brief Literature review**

- The network proposed (described further later) had performance gains compared with other models used for the same problem

- Input data type to the model is similar to our dataset, so preprocessing steps and model architecture may be reusable

- Didn’t clearly show how their network finds correlations across modals in their results, which was one of their main aims

- The network proposed in the paper can be adapted for this project - the binary classifier can identify democrat/republican instead

### **Method Overview**

**Datasets:**

1. Twitter Dataset - Dataset of tweets released for Verifying Multimedia Use (MediaEval)
    1. Tweets with videos were removed
    2. Used standard text pre-processing
    3. All tweets were in English - other languages were translated
2. Weibo Dataset - Dataset of data from Xinhua News Agency and Weibo
    1. Used Stanford Word Segmenter to tokenize Chinese text into words

Word embeddings generated using Word2Vec representation for words

**MVAE Model:**

The model can be broken into 3 parts, the encoder, decoder, and fake news detector.

1. **Encoder** - Concatenate separately generated text and image feature representations and pass “shared representation” through fully connected layer to obtain mean (µ) and variance (σ) vectors. This is used to create sampled latent vector.
    1. *Text Feature Representation Generation*
        - ***Process***
            1. Sequence of word embedding vectors
            2. Stacked bi-directional LSTM
            3. Fully connected layer
            4. Textual Feature Representation
    2. *Image Feature Representation Generation*
        - *********************************Process*********************************
            1. Image
            2. VGG-19 CNN architecture
            3. Multiple fully connected layers
            4. Image Feature Representation
2. **Decoder** - Reverse encoder process to reconstruct text and image content. Reconstructed output is compared with input data to optimize VAE loss
    1. *Reconstructed Words Generation*
        - ***Process***
            1. Sampled latent vector
            2. Fully connected layer
            3. Stacked bi-directional LSTM
            4. Fully connected layer w/ softmax
            5. Reconstructed words
    2. *Reconstructed VVG Features Generation*
        - ***Process***
            1. Sampled latent vector
            2. Multiple fully connected layers
            3. Reconstructed VVG features
3. **Fake News Detector** - Separates news into real/fake using binary classifier with several activation layers using input as the sampled latent vector 

**Hyperparameters:**

- Batch Size = 128
- Epochs = 300
- Learning Rate = $1 \times 10^{-5}$
- Optimizer = Adam
- L2-regularizer penalty = 0.05 (encoder/decoder), 0.3 (fake news detector)