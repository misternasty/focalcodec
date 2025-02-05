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

"""PyTorch Hub entry point."""

from typing import Any

from focalcodec import FocalCodec


# Make sure it is consistent with requirements.txt and README.md
dependencies = [
    "huggingface_hub",
    "safetensors",
    "torch",
]


def focalcodec(
    config: "str" = "lucadellalib/focalcodec/LibriTTS960_50Hz",
    pretrained: "bool" = True,
    **kwargs: "Any",
) -> "FocalCodec":
    """Load FocalCodec.

    Parameters
    ----------
    config:
        Path to the configuration file. This can be either:
        - A local JSON file;
        - a JSON file hosted on Hugging Face Hub (e.g. "username/repo_name/config.json").
        `.json` is automatically appended if the given path does not end with `.json`.
    pretrained:
        Whether to load the corresponding pretrained checkpoint. If True, the method will look for a
        checkpoint file with the same path/URL as the configuration file but with a `.safetensors` or
        `.pt` extension.

    """
    codec = FocalCodec.from_config(config, pretrained, **kwargs)
    return codec


if __name__ == "__main__":
    model = focalcodec()
    print(model)
    print(
        f"Total number of parameters/buffers: "
        f"{sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f} M"
    )
