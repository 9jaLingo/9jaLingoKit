"""
9jaLingo TTS-2: Text-to-Speech for Nigerian Languages

A simple interface for generating speech from text using a causal language model
and NVIDIA NeMo audio codec.

Supported Languages:
    - Nigerian Accented English
    - Hausa
    - Igbo
    - Yoruba
    - Pidgin

Features:
    - Voice cloning via speaker embeddings
    - Multi-language support with language tags
    - High-quality neural TTS
"""

from .api import NaijaLingoTTS, suppress_all_logs
from .core import TTSConfig
from .speaker_embedder import SpeakerEmbedder, compute_speaker_embedding

__version__ = "0.1.0"
__all__ = ["NaijaLingoTTS", "TTSConfig", "suppress_all_logs", "SpeakerEmbedder", "compute_speaker_embedding"]
