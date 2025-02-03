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

"""Focal modulation networks (see https://arxiv.org/abs/2203.11926) with support for jitting."""

# Adapted from:
# https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/focalnet/modeling_focalnet.py
# https://github.com/microsoft/FocalNet/blob/v1.0.1/classification/focalnet.py

from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn


__all__ = ["FocalDecoder", "FocalEncoder"]


class FeedForward(nn.Module):
    """Feed-forward neural network module.

    Parameters
    ----------
    dim:
        Input and output feature dimensions.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the activation and output projection layers.

    """

    def __init__(
        self,
        dim: "int" = 256,
        ffn_dim: "int" = 1024,
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
        output = input
        output = self.in_proj(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class FocalModulation(nn.Module):
    """Focal modulation layer that combines local and global context for processing the input.

    Parameters
    ----------
    dim:
        Dimensionality of the input and output features.
    focal_window:
        Size of the initial focal window.
    focal_level:
        Number of focal levels for hierarchical context aggregation.
    focal_factor:
        Scaling factor for focal window sizes across levels.
    dropout:
        Dropout probability applied to the output.
    use_post_norm:
        If True, apply layer normalization after modulation.
    normalize_modulator:
        If True, normalize the modulator for stabilizing training.
    store_hidden:
        If True, store the hidden states (`self.gates` and `self.modulator`).
        Useful for inspecting the model (e.g. plotting the modulator).

    """

    def __init__(
        self,
        dim: "int" = 256,
        focal_window: "int" = 7,
        focal_level: "int" = 2,
        focal_factor: "int" = 2,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        normalize_modulator: "bool" = False,
        store_hidden: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.normalize_modulator = normalize_modulator
        self.store_hidden = store_hidden

        # Modules
        self.in_proj = nn.Linear(dim, 2 * dim + focal_level + 1)
        self.layers = nn.ModuleList()
        self.activation = nn.GELU()
        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        for k in range(focal_level):
            kernel_size = focal_factor * k + focal_window
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                )
            )

        if use_post_norm:
            self.norm = nn.LayerNorm(dim)
        else:
            # JIT compilable
            self.norm = nn.Identity()

        # JIT compilable
        self.gates = torch.as_tensor(float("nan"))
        self.modulator = torch.as_tensor(float("nan"))

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
        input = self.in_proj(input).movedim(-1, -2)
        query, context, gates = input.split(
            [self.dim, self.dim, self.focal_level + 1], 1
        )

        # Context aggregation
        context_all = 0.0
        for level, layer in enumerate(self.layers):
            context = layer(context)
            context_all += context * gates[:, level : level + 1]
        context_global = self.activation(context.mean(dim=-1, keepdim=True))
        context_global = context_global * gates[:, self.focal_level :]
        context_all += context_global

        # Normalize context
        if self.normalize_modulator:
            context_all /= self.focal_level + 1

        # Focal modulation
        modulator = self.context_proj(context_all)
        output = query * modulator
        output = output.movedim(-2, -1)
        if self.use_post_norm:
            output = self.norm(output)

        output = self.out_proj(output)
        output = self.dropout(output)

        if self.store_hidden:
            self.modulator = modulator
            self.gates = gates

        return output


class FocalBlock(nn.Module):
    """Focal block that integrates focal modulation and feed forward layers with
    optional layer scaling.

    Parameters
    ----------
    dim:
        Dimensionality of the input and output features.
    ffn_dim:
        Dimensionality of the feed-forward network.
    focal_window:
        Size of the initial focal window in the modulation layer.
    focal_level:
        Number of hierarchical focal levels in the modulation layer.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layer.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layer for stabilizing training.

    """

    def __init__(
        self,
        dim: "int" = 256,
        ffn_dim: "int" = 1024,
        focal_window: "int" = 7,
        focal_level: "int" = 2,
        focal_factor: "int" = 2,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        normalize_modulator: "bool" = False,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.normalize_modulator = normalize_modulator

        # Modules
        self.modulation_norm = nn.LayerNorm(dim)
        self.modulation = FocalModulation(
            dim=dim,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            dropout=dropout,
            use_post_norm=use_post_norm,
            normalize_modulator=normalize_modulator,
        )
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(
            dim,
            ffn_dim,
            dropout,
        )

        if use_layerscale:
            self.modulation_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
            self.feed_forward_gamma = nn.Parameter(torch.full((dim,), layerscale_init))
        else:
            # JIT compilable
            self.modulation_gamma = 1.0
            self.feed_forward_gamma = 1.0

        # JIT compilable
        self.gates = torch.as_tensor(float("nan"))
        self.modulator = torch.as_tensor(float("nan"))

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
        output = input
        if not self.use_post_norm:
            output = self.modulation_norm(output)
        output = self.modulation(output)
        if self.use_post_norm:
            output = self.modulation_norm(output)
        if self.use_layerscale:
            output *= self.modulation_gamma
        output += input

        shortcut = output
        if self.use_post_norm:
            output = self.feed_forward(output)
            output = self.feed_forward_norm(output)
        else:
            output = self.feed_forward_norm(output)
            output = self.feed_forward(output)
        if self.use_layerscale:
            output *= self.feed_forward_gamma
        output += shortcut

        self.gates = self.modulation.gates
        self.modulator = self.modulation.modulator

        return output


class Snake1d(nn.Module):
    """Snake activation function for 1D inputs, allowing for periodic inductive bias.

    See https://arxiv.org/abs/2006.08195.

    Parameters
    ----------
    dim:
        Dimensionality of the input and output features.

    """

    def __init__(self, dim: "int" = 256) -> "None":
        super().__init__()
        self.dim = dim

        # Parameters
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, dim, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, dim, seq_length).

        """
        gate = (self.alpha * input).sin() ** 2
        output = input + (self.alpha + 1e-9).reciprocal() * gate
        return output

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(dim={self.dim})"


class DownScale(nn.Module):
    """A module for downscaling 1D input features using convolution
    followed by a Snake activation.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the convolutional kernel.
    stride:
        Stride of the convolution.

    """

    def __init__(
        self,
        input_dim: "int" = 512,
        output_dim: "int" = 256,
        kernel_size: "int" = 1,
        stride: "int" = 1,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

        # Modules
        self.downscale = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
        )
        self.activation = Snake1d(output_dim)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length // stride, output_dim).

        """
        output = self._maybe_pad(input)
        output = output.movedim(-1, -2)
        output = self.downscale(output)
        output = self.activation(output)
        output = output.movedim(-2, -1)
        return output

    def _maybe_pad(self, input: "Tensor") -> "Tensor":
        # [B, T, C]
        pad = (input.shape[-2] - self.kernel_size) % self.stride
        if pad > 0:
            input = nn.functional.pad(input, [0, 0, 0, pad])
        return input


class UpScale(nn.Module):
    """A module for downscaling 1D input features using convolution
    followed by a Snake activation.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    output_dim:
        Number of output channels.
    kernel_size:
        Size of the convolutional kernel.
    stride:
        Stride of the convolution.

    """

    def __init__(
        self,
        input_dim: "int" = 256,
        output_dim: "int" = 512,
        kernel_size: "int" = 1,
        stride: "int" = 1,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

        # Modules
        self.activation = Snake1d(input_dim)
        self.upscale = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=self.kernel_size, stride=self.stride
        )

    def forward(
        self,
        input: "Tensor",
        output_shape: "Optional[List[int]]" = None,
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        output_shape:
            A tuple specifying the desired output shape as (target_seq_length, output_dim).
            If provided, the output tensor is padded to match this shape.

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * stride or target_seq_length, output_dim).

        """
        output = input.movedim(-1, -2)
        output = self.activation(output)
        output = self.upscale(output)
        output = output.movedim(-2, -1)
        output = self._maybe_pad_or_trim(output, output_shape)
        return output

    def _maybe_pad_or_trim(
        self,
        input: "Tensor",
        output_shape: "Optional[List[int]]" = None,
    ) -> "Tensor":
        # [B, T, C]
        if output_shape is None:
            return input
        pad = output_shape[-2] - input.shape[-2]
        if pad > 0:
            input = nn.functional.pad(input, [0, 0, 0, pad])
        elif pad < 0:
            input = input.narrow(-2, 0, output_shape[-2])
        return input


class FocalEncoder(nn.Module):
    """A focal encoder module that combines downscaling and focal modulation
    for hierarchical feature extraction.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    hidden_dims:
        Sequence of hidden dimensions for intermediate layers.
    downscale_factors:
        Sequence of downscaling factors for each layer.
    focal_window:
        Size of the initial focal window in the modulation layers.
    focal_level:
        Number of hierarchical focal levels in the modulation layers.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layers.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for stabilizing training.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        output_dim: "int" = 13,
        hidden_dims: "Sequence[int]" = (1024, 512, 256),
        downscale_factors: "Sequence[int]" = (1, 1, 1),
        focal_window: "int" = 7,
        focal_level: "int" = 2,
        focal_factor: "int" = 2,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        normalize_modulator: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.downscale_factors = downscale_factors
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.normalize_modulator = normalize_modulator

        # Modules
        self.layers = nn.ModuleList()
        for hidden_dim, downscale_factor in zip(hidden_dims, downscale_factors):
            layer = nn.Sequential(
                DownScale(input_dim, hidden_dim, downscale_factor, downscale_factor),
                FocalBlock(
                    hidden_dim,
                    ffn_dim=hidden_dim * 4,
                    focal_window=focal_window,
                    focal_level=focal_level,
                    focal_factor=focal_factor,
                    dropout=dropout,
                    use_post_norm=use_post_norm,
                    use_layerscale=use_layerscale,
                    layerscale_init=layerscale_init,
                    normalize_modulator=normalize_modulator,
                ),
            )
            self.layers.append(layer)
            input_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, output_dim)

        # JIT compilable
        self.gates = [torch.as_tensor(float("nan")) for _ in self.layers]
        self.modulators = [torch.as_tensor(float("nan")) for _ in self.layers]

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length // stride, output_dim).

        """
        output = input
        for i, layer in enumerate(self.layers):
            output = layer(output)
            self.gates[i] = layer[1].gates
            self.modulators[i] = layer[1].modulator
        output = self.dropout(output)
        output = self.out_proj(output)
        return output


class FocalDecoder(nn.Module):
    """A focal decoder module that combines upscaling and focal modulation
    for hierarchical feature reconstruction.

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    output_dim:
        Dimension of the output features.
    hidden_dims:
        Sequence of hidden dimensions for intermediate layers.
    upscale_factors:
        Sequence of upscaling factors for each layer.
    focal_window:
        Size of the initial focal window in the modulation layers.
    focal_level:
        Number of hierarchical focal levels in the modulation layers.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layers.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for stabilizing training.

    """

    def __init__(
        self,
        input_dim: "int" = 13,
        output_dim: "int" = 1024,
        hidden_dims: "Sequence[int]" = (256, 512, 1024),
        upscale_factors: "Sequence[int]" = (1, 1, 1),
        focal_window: "int" = 7,
        focal_level: "int" = 2,
        focal_factor: "int" = 2,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        normalize_modulator: "bool" = False,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.upscale_factors = upscale_factors
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.normalize_modulator = normalize_modulator

        # Modules
        hidden_dims = tuple(hidden_dims) + (output_dim,)
        output_dim = hidden_dims[0]
        hidden_dims = hidden_dims[1:]
        self.in_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for hidden_dim, upscale_factor in zip(hidden_dims, upscale_factors):
            layer = nn.Sequential(
                FocalBlock(
                    output_dim,
                    ffn_dim=output_dim * 4,
                    focal_window=focal_window,
                    focal_level=focal_level,
                    focal_factor=focal_factor,
                    dropout=dropout,
                    use_post_norm=use_post_norm,
                    use_layerscale=use_layerscale,
                    layerscale_init=layerscale_init,
                    normalize_modulator=normalize_modulator,
                ),
                UpScale(output_dim, hidden_dim, upscale_factor, upscale_factor),
            )
            self.layers.append(layer)
            output_dim = hidden_dim

        # JIT compilable
        self.gates = [torch.as_tensor(float("nan")) for _ in self.layers]
        self.modulators = [torch.as_tensor(float("nan")) for _ in self.layers]

    def forward(
        self,
        input: "Tensor",
        output_shape: "Optional[List[int]]" = None,
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_dim).
        output_shape:
            A tuple specifying the desired output shape as (target_seq_length, output_dim).
            If provided, the output tensor is padded to match this shape.

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * stride or target_seq_length, output_dim).

        """
        output = self.in_proj(input)
        output = self.dropout(output)
        for i, layer in enumerate(self.layers):
            output = layer(output)
            self.gates[i] = layer[0].gates
            self.modulators[i] = layer[0].modulator
        output = self._maybe_pad_or_trim(output, output_shape)
        return output

    def _maybe_pad_or_trim(
        self,
        input: "Tensor",
        output_shape: "Optional[List[int]]" = None,
    ) -> "Tensor":
        # [B, T, C]
        if output_shape is None:
            return input
        pad = output_shape[-2] - input.shape[-2]
        if pad > 0:
            input = nn.functional.pad(input, [0, 0, 0, pad])
        elif pad < 0:
            input = input.narrow(-2, 0, output_shape[-2])
        return input


def test_model() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    encoder = FocalEncoder().to(device)
    print(encoder)
    print(sum([x.numel() for x in encoder.state_dict().values()]) / 1e6)
    decoder = FocalDecoder().to(device)
    print(decoder)
    print(sum([x.numel() for x in decoder.state_dict().values()]) / 1e6)

    input = torch.randn(B, 50, 1024, device=device)
    output = encoder(input)
    output = decoder(output, input.shape)
    assert input.shape == output.shape
    encoder_jit = torch.jit.script(encoder)
    decoder_jit = torch.jit.script(decoder)
    output_jit = encoder_jit(input)
    output_jit = decoder_jit(output_jit, input.shape)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    output.sum().backward()
    for k, v in encoder.named_parameters():
        assert v.grad is not None, k
    for k, v in decoder.named_parameters():
        assert v.grad is not None, k


def test_batch_invariance() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    encoder = FocalEncoder().to(device)
    print(encoder)
    decoder = FocalDecoder().to(device)
    print(decoder)

    input = torch.randn(B, 50, 1024, device=device)
    batch_encoder_outputs = encoder(input)
    batch_decoder_outputs = decoder(batch_encoder_outputs)

    single_encoder_outputs = []
    single_decoder_outputs = []
    for i in range(B):
        single_encoder_output = encoder(input[i][None])
        single_decoder_output = decoder(single_encoder_output)
        single_encoder_outputs.append(single_encoder_output)
        single_decoder_outputs.append(single_decoder_output)
    single_encoder_outputs = torch.cat(single_encoder_outputs)
    single_decoder_outputs = torch.cat(single_decoder_outputs)

    assert torch.allclose(batch_encoder_outputs, single_encoder_outputs, atol=1e-3), (
        ((batch_encoder_outputs - single_encoder_outputs) ** 2).mean().sqrt(),
    )

    assert torch.allclose(batch_decoder_outputs, single_decoder_outputs, atol=1e-3), (
        ((batch_decoder_outputs - single_decoder_outputs) ** 2).mean().sqrt(),
    )


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
