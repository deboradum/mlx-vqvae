import mlx.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import Quantizer

class VQVAE(nn.Module):
    def __init__(self, encoder_h_dim, res_h_dim, num_res_layers, k, d, beta):
        super().__init__()

        self.encoder = Encoder(3, encoder_h_dim, res_h_dim, num_res_layers)
        self.pre_quantization_conv = nn.Conv2d(encoder_h_dim, d, kernel_size=1)
        self.quantizer = Quantizer(k, d, beta)
        self.decoder = Decoder(d, encoder_h_dim, res_h_dim, num_res_layers)

    def __call__(self, x):
        # The model takes an input x, that is passed through an encoder producing output z_{e}(x)
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        # The discrete latent variables z are then calculated by a nearest neighbour look-up using the shared embedding space e
        loss_term_1, loss_term_2, z_q, perplexity, _, closest_indices = self.quantizer(
            z_e
        )
        # The input to the decoder is the corresponding embedding vector
        x_hat = self.decoder(z_q)

        return x_hat, loss_term_1, loss_term_2, perplexity, closest_indices
