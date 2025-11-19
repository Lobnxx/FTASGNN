"""Model factory implementation."""

from ..config import Config
from .graphsage import GraphSAGE
from .cnn1d import CNN1D
from .FTASGNN import FTASGNN
def create_model(model_num):
    """Factory function to create the selected model.
    
    Args:
        model_num (int): Model selection number
            0: CombinedModel
            1: GraphSAGE
            2: CNN1D 
    Returns:
        nn.Module: The selected model instance
    """
    # Ensure hidden_channels is a multiple of num_heads
    hidden_channels = (Config.HIDDEN_DIM // Config.heads) * Config.heads
    
    models = [
        GraphSAGE(
            in_channels=Config.d_session,
            hidden_channels=Config.HIDDEN_DIM,
            out_channels=Config.OUT_CHANNELS,
            aggr="mean"
        ),
        CNN1D(
            input_dim=Config.d_session,
            output_dim=Config.OUT_CHANNELS,
        ),
        FTASGNN(
        input_dim=Config.d_packet if hasattr(Config, 'd_packet') else 64,
        hidden_dim=Config.HIDDEN_DIM,
        out_channels=Config.OUT_CHANNELS,
        seq_len=Config.seq_len if hasattr(Config, 'seq_len') else 8,
        heads=Config.heads if hasattr(Config, 'head') else 2,
        dropout=Config.dropout if hasattr(Config, 'dropout') else 0.1,
        aggr="mean",
        latent_dim_scale=2
        )
    ]
    return models[model_num].to(Config.DEVICE)
