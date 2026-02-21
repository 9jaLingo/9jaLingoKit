<p align="center">
  <img src="assets/logo.png" alt="9jaLingo Logo" width="400"/>
</p>

<div align="center">

# 9jaLingo TTS-2

**Text-to-Speech for Nigerian Languages with Voice Cloning**

[![PyPI version](https://img.shields.io/pypi/v/naijalingo-tts-2.svg)](https://pypi.org/project/naijalingo-tts-2/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

</div>

---

9jaLingo TTS-2 is a neural text-to-speech engine built for **Nigerian languages**. It uses causal language models with advanced architectural innovations to generate natural-sounding speech with voice cloning capabilities.

## Supported Languages

| Language | Tag |
|---|---|
| 🇳🇬 Nigerian Accented English | `en_NG` |
| 🇳🇬 Hausa | `ha` |
| 🇳🇬 Igbo | `ig` |
| 🇳🇬 Yoruba | `yo` |
| 🇳🇬 Pidgin | `pcm` |

## Features

- **5 Nigerian Languages** — English (Nigerian Accent), Hausa, Igbo, Yoruba, and Pidgin
- **Voice Cloning** — Clone any voice from a short reference audio sample
- **Speaker Embeddings** — True voice control through learned speaker representations
- **Learnable RoPE Theta** — Per-layer frequency scaling for better position encoding
- **Frame-Level Position Encoding** — Precise temporal control with configurable audio frame positioning
- **Language Tag Support** — Multi-language support through language identifiers
- **Extended Generation** — Up to 40 seconds of continuous high-quality audio
- **Flexible Sampling** — Temperature, top-p, and repetition penalty at generation time

## Installation

```bash
pip install naijalingo-tts-2
pip install -U "transformers==4.56.0"
```

## Quick Start

```python
from naijalingo_tts_2 import NaijaLingoTTS

# Initialize model
model = NaijaLingoTTS('9jalingo/your-model-name')

# Generate speech
audio, text = model("Bawo ni, kilode?", language_tag="yo")

# Save to file
model.save_audio(audio, "output.wav")
```

Three lines for high-quality Nigerian language TTS! 🎉

## Voice Cloning

9jaLingo TTS-2 supports **voice cloning** — extract a speaker's voice from a reference audio and generate speech in that voice.

```python
from naijalingo_tts_2 import NaijaLingoTTS, SpeakerEmbedder

# Initialize
model = NaijaLingoTTS('9jalingo/your-model-name')
embedder = SpeakerEmbedder()

# Extract speaker embedding from reference audio
speaker_embedding = embedder.embed_audio_file("reference_voice.wav")  # [1, 128]

# Generate speech with that voice
audio, text = model(
    "Na so the matter be, my broda.",
    language_tag="pcm",
    speaker_emb=speaker_embedding
)
model.save_audio(audio, "cloned_voice.wav")
```

### How Speaker Embeddings Work

The speaker embedder uses a **WavLM-based model** trained to extract speaker characteristics:

1. **Input**: Audio at any sample rate (3-30 seconds recommended, automatically resampled to 16kHz)
2. **Processing**: MVN normalization → WavLM encoder → Stats pooling → Projection → L2 normalization
3. **Output**: 128-dim L2-normalized speaker embedding ready for TTS

```python
from naijalingo_tts_2 import SpeakerEmbedder
import torch

embedder = SpeakerEmbedder()

# From audio file
embedding = embedder.embed_audio_file("voice.wav")

# From numpy array
import numpy as np
audio_array = np.random.randn(16000 * 5)
embedding = embedder.embed_audio(audio_array, sample_rate=16000)

# Save for later
torch.save(embedding, "my_voice.pt")

# Use saved embedding
audio, text = model("Hello!", speaker_emb="my_voice.pt")
```

> **Pro tip**: Use 10-20 seconds of clean reference audio for best results. Audio at any sample rate is supported (automatic resampling).

## Language Tag Support

```python
from naijalingo_tts_2 import NaijaLingoTTS

model = NaijaLingoTTS('9jalingo/your-multilingual-model')

# Check available tags
print(f"Status: {model.status}")
model.show_language_tags()

# Generate with specific language
audio, text = model("Sannu da zuwa!", language_tag="ha")       # Hausa
audio, text = model("Kedu ka imere?", language_tag="ig")       # Igbo
audio, text = model("Bawo ni o?", language_tag="yo")           # Yoruba
audio, text = model("How far, my guy?", language_tag="pcm")    # Pidgin
audio, text = model("Good morning everyone.", language_tag="en_NG")  # Nigerian English
```

## Controlling Generation

```python
model = NaijaLingoTTS(
    '9jalingo/your-model-name',
    max_new_tokens=3000,
    suppress_logs=True,
    show_info=True,
)

audio, text = model(
    "Your text here",
    temperature=0.7,           # Lower = more deterministic
    top_p=0.9,                 # Nucleus sampling threshold
    repetition_penalty=1.2,    # Penalize repetition
    speaker_emb=speaker_emb,   # Optional: voice cloning
    language_tag="yo"          # Optional: language tag
)
```

## Model Info Banner

When initialized, the model displays helpful information:

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║                    9 j a L i n g o  TTS-2                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

     🗣️  Nigerian Language Text-to-Speech Engine

──────────────────────────────────────────────────────────────
  Model: 9jalingo/your-model-name
  Device: GPU (CUDA)
  Mode: Available language tags (5 language tags)
  Tags: en_NG, ha, ig, yo, pcm

  Configuration:
    • Sample Rate: 22050 Hz
    • Max Tokens: 3000
    • Speaker Embedding Dim: 128
    • Learnable RoPE: Enabled
──────────────────────────────────────────────────────────────

  Supported: English (Nigerian) | Hausa | Igbo | Yoruba | Pidgin
  Voice Cloning: Enabled 🎙️

  Ready to generate speech! 🎵
```

## Playing Audio in Jupyter

```python
from naijalingo_tts_2 import NaijaLingoTTS
from IPython.display import Audio as aplay

model = NaijaLingoTTS('9jalingo/your-model-name')
audio, text = model("E kaabo!", language_tag="yo")

aplay(audio, rate=model.sample_rate)
```

## API Reference

### `NaijaLingoTTS(model_name, **kwargs)`

Main TTS interface.

**Parameters:**
- `model_name` (str): HuggingFace model ID or local path
- `max_new_tokens` (int): Max generation length (default: 3000)
- `device_map` (str): Device mapping (default: "auto")
- `suppress_logs` (bool): Suppress library logs (default: True)
- `show_info` (bool): Display model info banner (default: True)

**Methods:**
- `model(text, language_tag=None, speaker_emb=None, temperature=1.0, top_p=0.95, repetition_penalty=1.1)` → `(audio, text)`
- `model.generate(...)` → Same as `__call__`
- `model.save_audio(audio, path)` → Save audio to file
- `model.show_model_info()` → Display model banner
- `model.show_language_tags()` → Display available language tags
- `model.load_speaker_embedding(path)` → Load speaker embedding from `.pt` file

### `SpeakerEmbedder(model_name, device, max_duration_sec)`

Extract speaker embeddings from audio.

**Parameters:**
- `model_name` (str): HuggingFace model ID (default: `"nineninesix/speaker-emb-tbr"`)
- `device` (str): `"cuda"` or `"cpu"` (default: auto-detect)
- `max_duration_sec` (float): Max audio length in seconds (default: 30.0)

**Methods:**
- `embedder.embed_audio(audio, sample_rate=16000)` → `[1, 128]` tensor
- `embedder.embed_audio_file(path)` → `[1, 128]` tensor

### Convenience Function

```python
from naijalingo_tts_2 import compute_speaker_embedding

embedding = compute_speaker_embedding("speaker.wav")
```

## Complete Example: Voice Cloning Pipeline

```python
from naijalingo_tts_2 import NaijaLingoTTS, SpeakerEmbedder
import torch

# 1. Initialize
tts = NaijaLingoTTS('9jalingo/your-model-name')
embedder = SpeakerEmbedder()

# 2. Extract speaker embedding
speaker_emb = embedder.embed_audio_file("reference_speaker.wav")
torch.save(speaker_emb, "my_voice.pt")

# 3. Generate in multiple languages with cloned voice
languages = {
    "en_NG": "Good morning, how are you doing today?",
    "ha":    "Ina kwana, yaya dai?",
    "ig":    "Ụtụtụ ọma, kedu ka ị mere?",
    "yo":    "E kaaro, bawo ni o se wa?",
    "pcm":   "Good morning o, how body?",
}

for lang, text in languages.items():
    audio, _ = tts(text, language_tag=lang, speaker_emb=speaker_emb)
    tts.save_audio(audio, f"output_{lang}.wav")
    print(f"✅ Generated {lang}: {text}")
```

## Voice Cloning Best Practices

**Reference Audio Quality:**
- ✅ Clean recordings without background noise
- ✅ Proper audio levels (not too quiet, not clipping)
- ✅ 10-20 seconds of clear speech
- ✅ Any sample rate (automatic resampling to 16kHz)
- ❌ Avoid noisy, compressed, or low-quality recordings

**Better Speaker Representation:**

```python
# Average multiple samples for more robust embedding
embeddings = [embedder.embed_audio_file(f) for f in sample_files]
averaged_embedding = torch.stack(embeddings).mean(dim=0)
```

## Architecture

**Two-Stage Pipeline:**
1. **Text → Audio Tokens**: Modified LFM2 causal LM generates discrete audio tokens
2. **Audio Tokens → Waveform**: NVIDIA NeMo NanoCodec decodes tokens to 22kHz audio

**Key Innovations:**
- **Learnable RoPE** — Per-layer frequency scaling for better positional encoding
- **Frame-Level Positions** — Audio tokens grouped in frames of 4 with shared positions
- **Speaker Embeddings** — 128-dim continuous representations for zero-shot voice cloning
- **Language Tags** — Accent and language control via prefix identifiers

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- Transformers 4.56.0+
- NeMo Toolkit

## Performance

| Setup | ~10s of audio |
|---|---|
| GPU (CUDA) | 2-5 seconds |
| CPU | 20-60 seconds |
| VRAM | 4-8 GB (bfloat16) |

## Responsible Use

**Prohibited activities include:**
- Generating false or misleading information
- Impersonating individuals without consent
- Hate speech, harassment, or incitement of violence
- Malicious activities such as spamming, phishing, or fraud

By using this package, you agree to comply with all applicable laws.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

```bibtex
@software{naijalingo_tts_2,
  author = {9jaLingo},
  title = {9jaLingo TTS-2: Text-to-Speech for Nigerian Languages},
  year = {2026},
  publisher = {PyPI},
  howpublished = {\url{https://pypi.org/project/naijalingo-tts-2/}},
  note = {Supports English (Nigerian), Hausa, Igbo, Yoruba, and Pidgin with voice cloning}
}
```

---

**Made with ❤️ by 9jaLingo for Nigeria 🇳🇬**
