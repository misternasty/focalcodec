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

"""Binary spherical quantization (see https://arxiv.org/abs/2406.07548) with support for jitting."""

# Adapted from:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/3e4ce165774d3f5944f12ffb5ccd02848bb38df6/vector_quantize_pytorch/lookup_free_quantization.py

import math
from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = ["BinarySphericalQuantizer"]


class BinarySphericalQuantizer(nn.Module):
    """Binary spherical quantizer that maps inputs to binary codes on the unit hypersphere.

    Parameters
    ----------
    codebook_size:
        Number of binary codes in the codebook.

    """

    def __init__(self, codebook_size: "int" = 8192) -> "None":
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = int(math.log2(codebook_size))
        self.codebook_value = 1 / math.sqrt(self.dim)

        # Buffers
        self.register_buffer("mask", 2 ** torch.arange(self.dim - 1, -1, -1))
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self._bits_to_codes(bits) * self.codebook_value
        self.register_buffer("codebook", codebook.float(), persistent=False)

    def forward(self, input: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input latents of shape (..., dim).

        Returns
        -------
            - Output tokens of shape (...);
            - output codes (i.e. quantized latents) of shape (..., dim).

        """
        toks = self.lats_to_toks(input)
        codes = self.toks_to_codes(toks)
        return toks, codes

    @torch.jit.export
    def lats_to_codes(self, input: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        codebook_value = torch.full_like(input, self.codebook_value)
        output = torch.where(input > 0, codebook_value, -codebook_value)
        return output

    @torch.jit.export
    def lats_to_toks(self, input: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (...).

        """
        toks = self.codes_to_toks(self.lats_to_codes(input))
        return toks

    @torch.jit.export
    def codes_to_toks(self, input: "Tensor") -> "Tensor":
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (...).

        """
        output = (input > 0).int() * self.mask.int()
        output = output.sum(dim=-1)
        return output

    @torch.jit.export
    def toks_to_codes(self, input: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        input:
            Input tensor of shape (...).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        bits = ((input[..., None].int() & self.mask) != 0).to(self.codebook.dtype)
        output = self._bits_to_codes(bits)
        output *= self.codebook_value
        return output

    def _bits_to_codes(self, input: "Tensor") -> "Tensor":
        return input * 2 - 1

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(codebook_size={self.codebook_size})"


def test_model() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = BinarySphericalQuantizer().to(device)
    print(model)
    print(sum([x.numel() for x in model.state_dict().values()]) / 1e6)

    input = torch.randn(B, 50, 13, device=device)
    toks, codes = model(input)
    codes2 = model.lats_to_codes(input)
    toks2 = model.lats_to_toks(input)
    toks3 = model.codes_to_toks(codes)
    assert (toks == toks2).all()
    assert (toks == toks3).all()
    assert (codes == codes2).all()
    model_jit = torch.jit.script(model)
    toks_jit, codes_jit = model_jit(input)
    assert (toks == toks_jit).all()
    assert (codes == codes_jit).all()


def test_batch_invariance() -> "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    model = BinarySphericalQuantizer().to(device)
    print(model)

    input = torch.randn(B, 50, 13, device=device)
    batch_toks, batch_codes = model(input)

    all_single_toks, all_single_codes = [], []
    for i in range(B):
        single_toks, single_codes = model(input[i][None])
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)
    all_single_toks = torch.cat(all_single_toks)
    all_single_codes = torch.cat(all_single_codes)

    assert (batch_toks == all_single_toks).all()
    assert (batch_codes == all_single_codes).all()


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
