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

"""Vocos with support for jitting."""

# Adapted from:
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/heads.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/models.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/modules.py
# https://github.com/gemelo-ai/vocos/tree/v0.1.0/vocos/spectral_ops.py

from typing import Optional

import torch
from torch import Tensor, nn


__all__ = ["Vocos"]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D audio signals adapted from https://github.com/facebookresearch/ConvNeXt.

    Parameters
    ----------
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    padding:
        Padding of the convolution.
    layerscale_init:
        Initial value for layer scaling parameter.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        padding: "int" = 3,
        layerscale_init: "Optional[float]" = None,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.layerscale_init = layerscale_init

        # Modules
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(ffn_dim, dim)

        # Parameters
        if layerscale_init is not None:
            self.gamma = nn.Parameter(
                torch.full((dim,), layerscale_init), requires_grad=True
            )
        else:
            self.gamma = None

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
        output = self.dwconv(input)
        output = output.movedim(-1, -2)
        output = self.norm(output)
        output = self.pwconv1(output)
        output = self.activation(output)
        output = self.pwconv2(output)
        if self.gamma is not None:
            output = self.gamma * output
        output = output.movedim(-1, -2)
        output = input + output
        return output


class ISTFT(nn.Module):
    """Custom implementation of inverse STFT with support for "same" padding.

    Parameters
    ----------
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.
    win_length:
        Size of window frame and STFT filter.

    """

    def __init__(
        self,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
        win_length: "int" = 1024,
    ) -> "None":
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad = (win_length - hop_length) // 2

        # Buffers
        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, n_fft // 2 + 1, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        # Inverse FFT
        ifft = torch.fft.irfft(input, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and add
        T = input.shape[-1]
        output_size = (T - 1) * self.hop_length + self.win_length
        output = nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, self.pad : -self.pad]

        # Window envelope
        window_sq = (self.window**2).expand(1, T, -1).movedim(-1, -2)
        window_envelope = nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[self.pad : -self.pad]

        # Normalize
        # assert (window_envelope > 1e-11).all()
        output /= window_envelope

        return output

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"win_length={self.win_length})"
        )


class VocosBackbone(nn.Module):
    """Vocos backbone.

    Parameters
    ----------
    input_channels:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of hidden channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    padding:
        Padding of the convolution.
    layerscale_init:
        Initial value for layer scaling parameter.

    """

    def __init__(
        self,
        input_channels: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        padding: "int" = 3,
        layerscale_init: "Optional[float]" = None,
    ) -> "None":
        super().__init__()
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.layerscale_init = (
            1 / num_layers if layerscale_init is None else layerscale_init
        )

        # Modules
        self.embedding = nn.Conv1d(input_channels, dim, kernel_size, padding=padding)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.layers = nn.ModuleList(
            ConvNeXtBlock(dim, ffn_dim, kernel_size, padding, self.layerscale_init)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)

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
        output = self.embedding(input)
        output = output.movedim(-1, -2)
        output = self.input_norm(output)
        output = output.movedim(-1, -2)
        for layer in self.layers:
            output = layer(output)
        output = output.movedim(-1, -2)
        output = self.output_norm(output)
        output = output.movedim(-1, -2)
        return output


class ISTFTHead(nn.Module):
    """Inverse STFT head.

    Parameters
    ----------
    dim:
        Number of input channels.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.

    """

    def __init__(
        self,
        dim: "int" = 512,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Modules
        self.proj = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
        )

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, dim, seq_length).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        input = input.movedim(-1, -2)
        output = self.proj(input)
        output = output.movedim(-1, -2)
        mag, phase = output.chunk(2, dim=-2)
        mag = mag.exp()
        # Safeguard to prevent excessively large magnitudes
        mag = mag.clamp(max=1e2)
        # Real and imaginary value
        # JIT compilable
        S = mag * torch.complex(phase.cos(), phase.sin())
        audio = self.istft(S)
        return audio


class Vocos(nn.Module):
    """Vocos generator for waveform synthesis.

    Parameters
    ----------
    input_channels:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of input channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the convolutional kernel.
    padding:
        Padding of the convolution.
    layerscale_init:
        Initial value for layer scaling parameter.
    n_fft:
        Size of Fourier transform.
    hop_length:
        Distance between neighboring sliding window frames.

    """

    def __init__(
        self,
        input_channels: "int" = 1024,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 1536,
        kernel_size: "int" = 7,
        padding: "int" = 3,
        layerscale_init: "Optional[float]" = None,
        n_fft: "int" = 1024,
        hop_length: "int" = 320,
    ) -> "None":
        super().__init__()
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.layerscale_init = layerscale_init or 1 / num_layers
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Modules
        self.backbone = VocosBackbone(
            input_channels, num_layers, dim, ffn_dim, layerscale_init=layerscale_init
        )
        self.head = ISTFTHead(dim, n_fft, hop_length)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, input_channels).

        Returns
        -------
            Output tensor of shape (batch_size, seq_length * hop_length).

        """
        output = input.movedim(-1, -2)
        output = self.backbone(output)
        output = self.head(output)
        return output


def test_model() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = Vocos().to(device)
    print(model)
    print(sum([x.numel() for x in model.state_dict().values()]) / 1e6)

    input = torch.randn(B, 50, 1024, device=device)
    output = model(input)
    model_jit = torch.jit.script(model)
    output_jit = model_jit(input)
    assert torch.allclose(output, output_jit, atol=1e-6), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k


def test_batch_invariance() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    model = Vocos().to(device)
    print(model)

    input = torch.randn(B, 50, 1024, device=device)
    batch_outputs = model(input)

    single_outputs = []
    for i in range(B):
        single_output = model(input[i][None])
        single_outputs.append(single_output)
    single_outputs = torch.cat(single_outputs)

    assert torch.allclose(batch_outputs, single_outputs, atol=1e-3), (
        ((batch_outputs - single_outputs) ** 2).mean().sqrt(),
    )


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
