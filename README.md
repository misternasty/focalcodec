# FocalCodec

A low-bitrate 16 kHz speech codec based on [focal modulation](https://arxiv.org/abs/2203.11926).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org). Open a terminal and run:

```
pip install huggingface-hub safetensors torch torchaudio
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

We use `torch.hub` to make loading the model easy (no need to clone the repository):

```python
import torch
import torchaudio

# Load FocalCodec model
config = "lucadellalib/focalcodec/LibriTTS960_50Hz"
codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec", config=config)
codec.eval().requires_grad_(False)

# Load and preprocess the input audio
sig, sample_rate = torchaudio.load("<path-to-audio-file>")
sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate)

# Encode and decode the audio
toks = codec.sig_to_toks(sig)
rec_sig = codec.toks_to_sig(toks)

# Save the reconstructed audio
torchaudio.save("reconstruction.wav", rec_sig, codec.sample_rate)
```

For more details and example usage, see `demo.py`.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
