import torch
import torch.nn as nn
import torch.nn.functional as F


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

    # Multi-Head Attention Mechanism


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "The number of input dimensions must be divisible by the number of heads."

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Linear transformations
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose for multi-head attention
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product
        attention_weights = F.softmax(energy, dim=-1)

        # Attention output
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                                              -1)  # (batch_size, seq_length, input_dim)
        output = self.fc_out(attention_output)

        return output

    # Residual Block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.match_dimensions = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,
                                          padding=0) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)

        out = self.conv2(out)

        if self.match_dimensions is not None:
            identity = self.match_dimensions(identity)

        out += identity
        out = self.leaky_relu(out)
        return out

    # Comprehensive Neural Network


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NeuralNetwork, self).__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim)
        self.multihead_attention = MultiHeadAttention(latent_dim, 8)

        # 1D CNN Module (Customized Residual Block)
        self.res_block1 = ResidualBlock(1, 32, stride=2)
        self.res_block2 = ResidualBlock(32, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 128, stride=2)
        self.res_block4 = ResidualBlock(128, 256, stride=2)
        self.res_block5 = ResidualBlock(256, 512, stride=2)
        self.res_block6 = ResidualBlock(512, 1024, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final Regression Layer
        self.fc = nn.Linear(1024, 1)

        # Weight Initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Separate the last two columns from the input
        last_column = x[:, -1]  # Last column to be added later
        x = x[:, :-2]  # Take all but the last two columns as input to the network

        x = self.autoencoder(x)

        # Multi-Head Attention Module
        attention_input = x.unsqueeze(1)  # Add sequence length dimension
        x = self.multihead_attention(attention_input)
        x = x.squeeze(1) + attention_input.squeeze(1)  # Skip connection

        # 1D CNN Part
        cnn_input = x.unsqueeze(1)  # Add channel dimension

        x = self.res_block1(cnn_input)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)

        x = self.pool(x).squeeze(-1)  # Global Average Pooling

        # Final Regression Layer
        x = self.fc(x)

        # Integrate the last column into the output
        output = x.squeeze(1) + last_column - 0.5  # Add last column and subtract 0.5
        return output

# # Test the Model
# if __name__ == "__main__":
#     input_dim = 50000  # Input dimension
#     latent_dim = 512  # Dimensionality after reduction
#     batch_size = 64
#
#     # Create model
#     model = NeuralNetwork(input_dim-2, latent_dim).double()  # Use double precision
#     print(model)
#
#     # Create dummy data, ensuring the data is double precision and has 50000 features
#     dummy_data = torch.randn(batch_size, input_dim, dtype=torch.double)  # (batch_size, input_dim)
#     output = model(dummy_data)
#
#     # Output shape
#     print("Output shape:", output.shape)  # Should be (batch_size,)