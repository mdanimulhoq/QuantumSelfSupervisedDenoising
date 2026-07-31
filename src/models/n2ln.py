"""N2LN-QEM model."""
import torch.nn as nn
from src.models.encoder import BitstringEncoder
from src.models.set_transformer import CountWeightedSetTransformer
from src.models.decoder import DualHeadDecoder

class N2LNQEM(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_isab=2, n_sab=1,
                 d_ff=256, m=16, decoder_hidden=128, dropout=0.1,
                 max_qubits=20, use_mlp_scorer=True, temperature_floor=0.3):
        super().__init__()
        self.encoder = BitstringEncoder(d_model, max_qubits)
        self.transformer = CountWeightedSetTransformer(
            d_model, n_heads, n_isab, n_sab, d_ff, m, dropout)
        self.decoder = DualHeadDecoder(
            d_model, max_qubits, decoder_hidden, 0.3, dropout,
            use_mlp_scorer, temperature_floor)
        self.decoder.set_bitstring_encoder(self.encoder)
        self.mode = 'unified'

    def forward(self, bitstrings, counts, mode=None):
        if mode is None:
            mode = self.mode
        embeddings, mask = self.encoder(bitstrings)
        z = self.transformer(embeddings, counts, mask=mask)
        sn_dist, hn_dist = self.decoder(z, bitstrings, mask=mask)
        if mode == 'sn_only':
            return sn_dist, None
        elif mode == 'hn_only':
            return None, hn_dist
        else:
            return sn_dist, hn_dist

    def set_mode(self, mode):
        assert mode in ['sn_only', 'hn_only', 'unified']
        self.mode = mode
