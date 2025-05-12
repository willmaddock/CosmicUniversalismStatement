import torch
import torch.nn as nn

class CURecursiveTimeLayer(nn.Module):
    """
    Recursive Time Compression Layer that simulates recursive temporal dynamics
    in line with the Cosmic Universalism framework. This layer compresses time sequences
    through recursive operations, integrating time dynamics into the symbolic memory system.
    """

    def __init__(self, input_size, output_size, time_depth=10, compression_factor=0.5):
        """
        Initialize the Recursive Time Layer.

        Args:
            input_size (int): The input size of each time step's features.
            output_size (int): The output size of the compressed time embeddings.
            time_depth (int): The recursive depth of time compression, number of recursive steps to perform.
            compression_factor (float): Factor by which time data is compressed at each level.
        """
        super(CURecursiveTimeLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_depth = time_depth
        self.compression_factor = compression_factor

        # Define layers for recursive processing
        self.time_linear = nn.ModuleList([
            nn.Linear(input_size if i == 0 else output_size, output_size) for i in range(time_depth)
        ])

        # Activation function, to simulate temporal transitions
        self.activation = nn.Tanh()

        # Output compression layer after recursive encoding
        self.final_compression = nn.Linear(output_size, output_size)

    def forward(self, x):
        """
        Forward pass through the recursive time compression layer.

        Args:
            x (tensor): Input tensor with shape (batch_size, time_steps, input_size).

        Returns:
            tensor: Output tensor after recursive time compression with shape (batch_size, output_size).
        """
        batch_size, time_steps, _ = x.size()

        compressed_time = x
        for depth in range(self.time_depth):
            # Flatten batch and time steps for processing
            compressed_time = compressed_time.view(batch_size * time_steps, -1)

            # Apply linear transformation at each depth
            compressed_time = self.time_linear[depth](compressed_time)

            # Apply activation function
            compressed_time = self.activation(compressed_time)

            # Restore batch dimensions while gradually compressing time representation
            compressed_time = compressed_time.view(batch_size, time_steps, self.output_size)

            # Apply compression factor (reducing magnitude)
            compressed_time *= self.compression_factor

        # Extract the last time step for the final output
        final_output = self.final_compression(compressed_time[:, -1, :])
        return final_output


# Example usage
if __name__ == "__main__":
    # Define input and output sizes
    input_size = 128
    output_size = 64

    # Create random input tensor with shape (batch_size, time_steps, input_size)
    batch_size = 32
    time_steps = 10
    x = torch.randn(batch_size, time_steps, input_size)

    # Initialize and forward-pass through Recursive Time Compression Layer
    recursive_time_layer = CURecursiveTimeLayer(input_size, output_size)
    output = recursive_time_layer(x)

    print(f"Output shape: {output.shape}")  # Expected: (batch_size, output_size)
