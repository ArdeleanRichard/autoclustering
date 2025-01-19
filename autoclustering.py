import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from datasets import create_2d4c, create_2d10c

# AutoClustering: A Feed-Forward Neural Network Based Clustering Algorithm
# Masaomi Kimura
# 2018 IEEE International Conference on Data Mining Workshops (ICDMW)
# https://ieeexplore.ieee.org/document/8637379

class AutoClustering(tf.keras.Model):
    def __init__(self, input_dim, encoder_layer_sizes, decoder_layer_sizes, num_clusters, alpha_final=500.0, gamma=4.0):
        """
        AutoClustering model with configurable encoder and decoder layers.

        Args:
            input_dim (int): Dimension of input data.
            encoder_layer_sizes (list of int): List specifying the number of neurons in each encoder layer.
            decoder_layer_sizes (list of int): List specifying the number of neurons in each decoder hidden layer.
            num_clusters (int): Number of clusters for the output of the encoder.
            alpha_final (float): Final value of alpha for the quasi-max function.
            gamma (float): Controls the curve of alpha growth.
        """
        super(AutoClustering, self).__init__()
        self.alpha_final = alpha_final
        self.gamma = gamma

        # Build the encoder layers
        self.encoder_layers = []
        for size in encoder_layer_sizes:
            self.encoder_layers.append(tf.keras.layers.Dense(size, activation='tanh'))

        # Final encoder layer for producing cluster assignments
        self.encoder_output_layer = tf.keras.layers.Dense(num_clusters, activation=None)

        # Build the decoder layers
        self.decoder_layers = []
        for size in decoder_layer_sizes:
            self.decoder_layers.append(tf.keras.layers.Dense(size, activation='tanh'))

        # Final decoder layer for reconstructing input
        self.decoder_output_layer = tf.keras.layers.Dense(input_dim, activation=None)

    def quasi_max(self, logits, alpha):
        """
        Quasi-max function approximated using scaled softmax.
        Args:
            logits (Tensor): The raw logits to be scaled.
            alpha (float): Scaling factor for softmax.

        Returns:
            Tensor: Quasi-max output.
        """
        scaled_logits = alpha * logits
        return tf.nn.softmax(scaled_logits)

    def call(self, inputs, alpha):
        """
        Forward pass through the AutoClustering model.

        Args:
            inputs (Tensor): Input data.
            alpha (float): Current alpha value for quasi-max function.

        Returns:
            tuple: (clusters, exemplars)
        """
        # Encoder: Input -> Cluster Assignments
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        logits = self.encoder_output_layer(x)
        clusters = self.quasi_max(logits, alpha)

        # Decoder: Cluster Assignments -> Exemplars
        x = clusters
        for layer in self.decoder_layers:
            x = layer(x)
        exemplars = self.decoder_output_layer(x)

        return clusters, exemplars

    def alpha_schedule(self, epoch, max_epochs):
        """Gradual scaling of alpha from softmax to hardmax."""
        return 1.0 + (self.alpha_final - 1.0) * ((epoch / max_epochs) ** self.gamma)


    def loss_function(self, inputs, exemplars):
        """Compute loss as squared Euclidean distance."""
        return tf.reduce_mean(tf.reduce_sum(tf.square(inputs - exemplars), axis=1))


    def train(self, X, learning_rate=0.001, max_epochs=100, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices(tf.cast(X, tf.float32)).shuffle(100).batch(batch_size)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        for epoch in range(max_epochs):
            alpha = self.alpha_schedule(epoch, max_epochs)
            epoch_loss = 0.0

            for batch in dataset:
                with tf.GradientTape() as tape:
                    clusters, exemplars = self.call(batch, alpha)
                    loss = self.loss_function(batch, exemplars)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += loss.numpy()

            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}, Alpha: {alpha:.2f}")

        # Extract final clusters
        final_clusters, exemplars = model(X, self.alpha_final)
        print(f"\nFinal clusters: \n{final_clusters}")
        unique_clusters = tf.unique(tf.argmax(final_clusters, axis=1)).y.numpy()
        print(f"Number of unique clusters: {len(unique_clusters)}")

        return model, exemplars


    def plot(self, X):
        # Perform PCA for 2D visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        # Predict final cluster labels
        final_clusters, exemplars = self.call(X, self.alpha_final)
        predicted_labels = tf.argmax(final_clusters, axis=1).numpy()

        # Plotting the ground truth and clustering results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Ground truth clusters
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=ground_truth, cmap='viridis', s=50, alpha=0.7)
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # Perform PCA for 2D visualization
        pca = PCA(n_components=2)
        exemplars = np.unique(exemplars, axis=0)
        print(exemplars)
        exemplars_2d = pca.fit_transform(exemplars)
        print(f"\nExemplars: \n{exemplars_2d}")

        # Predicted clusters
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.7)
        axes[1].scatter(exemplars_2d[:, 0], exemplars_2d[:, 1], c='red', s=100, alpha=0.7)
        axes[1].set_title("AutoClustering")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        plt.tight_layout()
        plt.show()

        print("F:", v_measure_score(ground_truth, predicted_labels))



if __name__ == "__main__":
    # LOAD IRIS
    # data = load_iris()
    # ground_truth = data.target
    # X = StandardScaler().fit_transform(data.data)

    # LOAD 2d4c
    # X, ground_truth = create_2d4c()
    # X = StandardScaler().fit_transform(X)

    # LOAD 2d10c
    X, ground_truth = create_2d10c()
    X = StandardScaler().fit_transform(X)

    print(X.shape)

    # Hyperparameters
    input_dim = X.shape[1]
    # Encoder and decoder configs
    encoder_layer_sizes = [16, 32, 16]
    decoder_layer_sizes = [16, 32, 16]
    num_clusters = len(np.unique(ground_truth))

    model = AutoClustering(input_dim, encoder_layer_sizes, decoder_layer_sizes, num_clusters)
    model.train(X)
    model.plot(X)
