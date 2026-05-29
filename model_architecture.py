import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM neural network model designed for day-ahead electricity load forecasting.
    Takes sequences of historical multivariable measurements and projects predictions
    across a 24-hour horizon.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 24):
        """
        Args:
            input_size (int): The number of features in the input sequence.
            hidden_size (int): Hidden dimension size of LSTM layers. Default is 64.
            num_layers (int): The number of stacked LSTM layers. Default is 2.
            output_size (int): Target length of predictions (hours). Default is 24.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer (batch_first=True expects shape: [batch, sequence, features])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully Connected Layer to project hidden state to final 24-hour forecast
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [batch_size, sequence_length, features].

        Returns:
            torch.Tensor: Projected forecasts of shape [batch_size, output_size].
        """
        # out: [batch_size, sequence_length, hidden_size]
        out, _ = self.lstm(x)

        # Extract only the final hidden state of the sequence to feed into the FC layer
        # out[:, -1, :] has shape [batch_size, hidden_size]
        out = self.fc(out[:, -1, :])

        return out
