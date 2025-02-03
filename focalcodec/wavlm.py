# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""WavLM (see https://arxiv.org/abs/2110.13900) with support for jitting and PyTorch FlashAttention."""

# Adapted from:
# https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/wavlm/modeling_wavlm.py
# https://github.com/microsoft/unilm/blob/2d5b25f7cecc4ab4ad44f65de5899ef6aa8c53f5/wavlm/WavLM.py
# https://github.com/microsoft/unilm/blob/2d5b25f7cecc4ab4ad44f65de5899ef6aa8c53f5/wavlm/modules.py

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


__all__ = ["WavLM"]


class Fp32LayerNorm(nn.LayerNorm):
    """Layer normalization with `float32` precision. See documentation of `nn.LayerNorm`."""

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        output = nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvBlock(nn.Module):
    """Convolutional block with dropout, normalization, and activation.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the convolutional kernel.
    stride:
        Stride for the convolution.
    bias:
        Whether to include a bias term in the convolution.
    dropout:
        Dropout probability.

    """

    def __init__(
        self,
        input_dim: "int" = 512,
        output_dim: "int" = 512,
        kernel_size: "int" = 10,
        stride: "int" = 5,
        bias: "bool" = False,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.dropout_ = dropout

        # Modules
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = Fp32LayerNorm(output_dim, elementwise_affine=True)
        self.activation = nn.GELU()

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, input_dim, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, output_dim, seq_length // stride).

        """
        output = self.conv(input)
        output = self.dropout(output)
        output = self.norm(output.movedim(-1, -2)).movedim(-1, -2)
        output = self.activation(output)
        return output


class SamePad(nn.Module):
    """Applies padding to ensure the output has the same temporal size as the input.

    Parameters
    ----------
    kernel_size:
        Size of the convolutional kernel.
    causal:
        If True, applies causal padding.

    """

    def __init__(
        self,
        kernel_size: "int",
        causal: "bool" = False,
    ) -> "None":
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.remove = 1 if kernel_size % 2 == 0 else 0
        if causal:
            self.remove = kernel_size - 1

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., seq_length).

        Returns
        -------
            Output tensor of shape (..., seq_length - x), where x = (kernel_size - 1)
            if causal, x = 1 if (not causal and kernel_size % 2 == 0), x = 0 otherwise.

        """
        if self.remove > 0:
            input = input[..., : -self.remove]
        return input

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"kernel_size={self.kernel_size}, "
            f"causal={self.causal})"
        )


class PositionalConvEmbedding(nn.Module):
    """Positional convolutional embedding for encoding positional information.

    Parameters
    ----------
    dim:
        Number of input/output channels.
    kernel_size:
        Size of the convolutional kernel.
    groups:
        Number of convolutional groups.

    """

    def __init__(
        self,
        dim: "int" = 512,
        kernel_size: "int" = 128,
        groups: "int" = 16,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups

        # Modules
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.padding = SamePad(kernel_size)
        self.activation = nn.GELU()

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        output = input.movedim(-1, -2)
        output = self.conv(output)
        output = self.padding(output)
        output = self.activation(output)
        output = output.movedim(-1, -2)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional embeddings.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability for attention weights.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = dim // num_heads

        # Modules
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        # Parameters
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Forward pass.

        This method applies relative positional embeddings and multi-head attention
        and handles key-value caching for efficient block-wise streaming inference.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        position_bias:
            Precomputed relative positional embeddings for the current input sequence.
        mask:
            A float mask of the same type as query, key, value that is added to the attention scores,
            shape (batch_size, ..., seq_length, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        gated_input = input.reshape(input.shape[:-1] + (self.num_heads, -1))
        gated_input = gated_input.movedim(-3, -2)

        relative_position_proj = self.gru_rel_pos_linear(gated_input)
        relative_position_proj = relative_position_proj.reshape(
            gated_input.shape[:-1] + (2, 4)
        ).sum(dim=-1)

        gate_a, gate_b = relative_position_proj.sigmoid().chunk(2, dim=-1)
        gate_input = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        gated_position_bias = gate_input * position_bias

        if mask is not None:
            # `mask` must be a float tensor
            gated_position_bias = gated_position_bias + mask

        output = self._scaled_dot_product_attention(
            input,
            gated_position_bias,
        )

        return output

    def _scaled_dot_product_attention(
        self,
        input: "Tensor",
        mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        B, T, _ = input.shape

        qs = self.q_proj(input).reshape(B, T, self.num_heads, -1)
        ks = self.k_proj(input).reshape(B, T, self.num_heads, -1)
        vs = self.v_proj(input).reshape(B, T, self.num_heads, -1)

        # Reshape for scaled_dot_product_attention
        qs = qs.movedim(-3, -2)  # [B, num_heads, T, head_dim]
        ks = ks.movedim(-3, -2)  # [B, num_heads, T, head_dim]
        vs = vs.movedim(-3, -2)  # [B, num_heads, T, head_dim]

        output = nn.functional.scaled_dot_product_attention(
            qs,
            ks.type_as(qs),
            vs.type_as(qs),
            attn_mask=mask.type_as(qs) if mask is not None else mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, num_heads, T, head_dim]

        # [B, T, num_heads * head_dim]
        output = output.movedim(1, 2).contiguous().reshape(B, T, -1)
        output = self.out_proj(output)  # [B, T, dim]

        return output


class FeedForward(nn.Module):
    """Feed-forward neural network module.

    Parameters
    ----------
    dim:
        Input and output feature dimensions.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation layer.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.in_proj = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(ffn_dim, dim)

    def forward(self, input: "Tensor"):
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        output = self.in_proj(input)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    """Transformer layer module comprising self-attention, feed-forward
    and normalization layers.

    Parameters
    ----------
    dim:
        Dimension of the input and output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    num_heads:
        Number of attention heads in the self-attention mechanism.
    dropout:
        Dropout probability applied in the attention and feed-forward layers.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_ = dropout

        # Modules
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)

    def forward(
        self,
        input: "Tensor",
        position_bias: "Tensor",
        mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """See documentation of `MultiHeadAttention.forward`."""
        output = input
        residual = output

        output = self.attention_norm(output)
        output = self.attention(output, position_bias, mask)
        output = self.dropout(output)
        output = residual + output

        residual = output
        output = self.feed_forward_norm(output)
        output = self.feed_forward(output)
        output = residual + output

        return output


class TransformerEncoder(nn.Module):
    """Transformer encoder with relative positional embeddings and
    convolutional positional embeddings.

    Parameters
    ----------
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of the input and output embeddings in the transformer.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    dropout:
        Dropout probability applied throughout the model.
    conv_pos:
        Size of the convolutional positional embeddings.
    conv_pos_groups:
        Number of groups in the convolutional positional embeddings.

    """

    def __init__(
        self,
        num_layers: "int" = 6,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: int = 800,
        dropout: "float" = 0.0,
        conv_pos: "int" = 128,
        conv_pos_groups: "int" = 16,
    ) -> "None":
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.dropout = dropout
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups

        # Needed to compute position bias
        self._num_buckets = num_buckets // 2
        self._num_buckets_minus_one = self._num_buckets - 1
        self._max_exact = self._num_buckets // 2
        self._num_buckets_minus_max_exact = self._num_buckets - self._max_exact
        self._log_max_distance_over_max_exact = math.log(
            self.max_distance / self._max_exact
        )

        # Modules
        self.positional_embedding = PositionalConvEmbedding(
            dim,
            conv_pos,
            conv_pos_groups,
        )
        self.relative_embedding = nn.Embedding(num_buckets, num_heads)
        self.layers = nn.ModuleList(
            TransformerLayer(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        input: "Tensor",
        key_padding_mask: "Optional[Tensor]" = None,
        output_norm: "bool" = False,
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim), representing the input embeddings.
        key_padding_mask:
            A boolean mask where a value of True indicates that the element should take part in attention,
            shape (batch_size, seq_length).
        output_norm:
            True to normalize the output, False otherwise.

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        B, T, _ = input.shape

        output = input
        if key_padding_mask is not None:
            output = output.clone()
            output[~key_padding_mask] = 0
            float_mask = torch.full(
                (B, T), -float("inf"), dtype=input.dtype, device=input.device
            )
            float_mask[key_padding_mask] = 0.0
            key_padding_mask = float_mask[:, None, None]

        output = output + self.positional_embedding(output)
        output = self.dropout(output)

        position_bias = self._compute_bias(T, T)[:, -T:].type_as(output)
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                position_bias,
                key_padding_mask,
            )

        if output_norm:
            output = self.norm(output)
        return output

    def _compute_bias(self, query_length: "int", key_length: "int") -> "Tensor":
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(
            self.relative_embedding.weight.device
        )
        values = self.relative_embedding(relative_position_bucket)
        values = values.movedim(-1, 0)
        return values

    def _relative_positions_bucket(self, relative_positions: "Tensor") -> "Tensor":
        relative_buckets = (relative_positions > 0).to(torch.long) * self._num_buckets
        relative_positions = relative_positions.abs()
        is_small = relative_positions < self._max_exact
        relative_positions_if_large = (
            relative_positions.float() / self._max_exact
        ).log()
        relative_positions_if_large /= self._log_max_distance_over_max_exact
        relative_positions_if_large *= self._num_buckets_minus_max_exact
        relative_positions_if_large += self._max_exact
        relative_positions_if_large = relative_positions_if_large.to(torch.long)
        relative_positions_if_large = relative_positions_if_large.clamp(
            max=self._num_buckets_minus_one
        )
        relative_buckets += torch.where(
            is_small, relative_positions, relative_positions_if_large
        )
        return relative_buckets


class FeatureExtractor(nn.Module):
    """Feature extraction module that applies a series of convolutional layers.

    Parameters
    ----------
    input_dim:
        Number of input channels or features.
    hidden_dims:
        Number of output channels for each convolutional layer.
    kernel_sizes:
        Kernel size for each convolutional layer.
    strides:
        Stride for each convolutional layer.
    bias:
        Whether to include a bias term in the convolutional layers.
    dropout:
        Dropout probability applied after each convolutional block.

    """

    def __init__(
        self,
        input_dim: "int" = 1,
        hidden_dims: "Sequence[int]" = (512,) + (512,) * 512 + (512,) * 512,
        kernel_sizes: "Sequence[int]" = (10,) + (3,) * 4 + (2,) * 2,
        strides: "Sequence[int]" = (5,) + (2,) * 4 + (2,) * 2,
        bias: "bool" = False,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.bias = bias
        self.dropout = dropout

        # Modules
        self.layers = nn.ModuleList()
        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            layer = ConvBlock(input_dim, hidden_dim, kernel_size, stride, bias, dropout)
            self.layers.append(layer)
            input_dim = hidden_dim

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length / prod(strides), hidden_dims[-1]).

        """
        output = input.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output)
        output = output.movedim(-2, -1)
        return output


class WavLM(nn.Module):
    """WavLM model.

    Parameters
    ----------
    hidden_dims:
        Number of filters for each convolutional layer.
    kernel_sizes:
        Kernel sizes for each convolutional layer.
    strides:
        Strides for each convolutional layer.
    num_layers:
        Number of transformer layers in the encoder.
    dim:
        Dimension of the input and output embeddings in the transformer.
    ffn_dim:
        Dimension of the feed-forward layer within each transformer layer.
    num_heads:
        Number of attention heads in each transformer layer.
    num_buckets:
        Number of buckets for relative positional embeddings.
    max_distance:
        Maximum distance for relative positional embeddings.
    dropout:
        Dropout probability applied throughout the model.
    conv_pos:
        Size of the convolutional positional embeddings.
    conv_pos_groups:
        Number of groups in the convolutional positional embeddings.

    """

    def __init__(
        self,
        hidden_dims: "Sequence[int]" = (512,) + (512,) * 4 + (512,) * 2,
        kernel_sizes: "Sequence[int]" = (10,) + (3,) * 4 + (2,) * 2,
        strides: "Sequence[int]" = (5,) + (2,) * 4 + (2,) * 2,
        num_layers: "int" = 6,
        dim: "int" = 1024,
        ffn_dim: "int" = 4096,
        num_heads: "int" = 16,
        num_buckets: "int" = 320,
        max_distance: int = 800,
        dropout: "float" = 0.0,
        conv_pos: "int" = 128,
        conv_pos_groups: "int" = 16,
    ) -> "None":
        super().__init__()
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.dropout = dropout
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups

        # Modules
        self.feature_extractor = FeatureExtractor(
            input_dim=1,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dims[-1])
        self.feature_proj = nn.Linear(hidden_dims[-1], dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            dropout=dropout,
            conv_pos=conv_pos,
            conv_pos_groups=conv_pos_groups,
        )

    def forward(
        self,
        input: "Tensor",
        key_padding_mask: "Optional[Tensor]" = None,
        output_norm: "bool" = False,
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input audio of shape (batch_size, seq_length).
        key_padding_mask:
            A boolean mask where a value of True indicates that the element should take part in attention,
            shape (batch_size, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        # [B, T, 1]
        input = input[..., None]
        output = self.feature_extractor(input)
        output = self.norm(output)
        output = self.feature_proj(output)
        output = self.dropout(output)
        if key_padding_mask is not None:
            extra = key_padding_mask.shape[1] % output.shape[1]
            if extra > 0:
                key_padding_mask = key_padding_mask[:, :-extra]
            key_padding_mask = key_padding_mask.reshape(
                key_padding_mask.shape[0], output.shape[1], -1
            )
            key_padding_mask = key_padding_mask.any(dim=-1)
        output = self.encoder(
            output,
            key_padding_mask,
            output_norm,
        )
        return output


def test_model() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = WavLM().to(device)
    print(model)
    print(sum([x.numel() for x in model.state_dict().values()]) / 1e6)

    # Process 16000 timesteps
    input = torch.randn(B, 16000, device=device)
    output = model(input, output_norm=True)
    model_jit = torch.jit.script(model)
    output_jit = model_jit(input, output_norm=True)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    print(output.shape)
    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    # Key padding mask
    key_padding_mask = torch.ones(B, 8000, dtype=torch.bool, device=device)
    key_padding_mask[0, 4000:] = False
    input = torch.randn(B, 8000, device=device)
    model(input, key_padding_mask=key_padding_mask)


def test_batch_invariance() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    model = WavLM().to(device)
    print(model)

    input = torch.randn(B, 16000, device=device)
    batch_outputs = model(input)

    single_outputs = []
    for i in range(B):
        single_output = model(input[i][None])
        single_outputs.append(single_output)
    single_outputs = torch.cat(single_outputs)

    assert torch.allclose(batch_outputs, single_outputs, atol=1e-2), (
        ((batch_outputs - single_outputs) ** 2).mean().sqrt(),
    )


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
