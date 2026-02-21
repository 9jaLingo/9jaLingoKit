"""
Speaker Embedder Module for 9jaLingo TTS-2
==========================================

Lightweight module for generating speaker embeddings from audio using WavLM model.
Model: Orange/Speaker-wavLM-tbr (16kHz input, 128-dim L2-normalized output)

Based on spk_embeddings.py from Orange SA (CC-BY-SA-3.0)
https://huggingface.co/Orange/Speaker-wavLM-tbr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from transformers.models.wavlm.modeling_wavlm import WavLMPreTrainedModel, WavLMModel


class TopLayers(nn.Module):
    """
    Projection layers on top of WavLM for speaker embedding extraction.

    Architecture:
        - Conv1d: 2048 -> 512
        - BatchNorm + ReLU
        - Conv1d: 512 -> embd_size (default 128)
        - BatchNorm + ReLU
        - L2 normalization
    """

    def __init__(self, embd_size: int = 250, top_interm_size: int = 512):
        super(TopLayers, self).__init__()
        self.affine1 = nn.Conv1d(in_channels=2048, out_channels=top_interm_size, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=top_interm_size, affine=False, eps=1e-03)
        self.affine2 = nn.Conv1d(in_channels=top_interm_size, out_channels=embd_size, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=embd_size, affine=False, eps=1e-03)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Stats pooling output [batch, 2048, 1]

        Returns:
            L2-normalized embeddings [batch, embd_size]
        """
        out = self.batchnorm1(self.activation(self.affine1(x)))
        out = self.batchnorm2(self.activation(self.affine2(out)))
        return F.normalize(out[:, :, 0])  # L2 normalization


class EmbeddingsModel(WavLMPreTrainedModel):
    """
    Complete WavLM-based speaker embedding model.

    Architecture:
        1. MVN normalization on input audio
        2. WavLM encoder
        3. Stats pooling (mean + std)
        4. Top projection layers
        5. L2 normalization
    """

    def __init__(self, config):
        super().__init__(config)
        self.wavlm = WavLMModel(config)
        self.top_layers = TopLayers(config.embd_size, config.top_interm_size)

    def forward(self, input_values):
        """
        Args:
            input_values: Audio waveform [batch, time_samples]

        Returns:
            Speaker embeddings [batch, embd_size]
        """
        # MVN normalization (mean-variance normalization)
        x_norm = (input_values - input_values.mean(dim=1, keepdim=True)) / (
            input_values.std(dim=1, keepdim=True) + 1e-10
        )

        # WavLM forward pass
        base_out = self.wavlm(input_values=x_norm, output_hidden_states=False).last_hidden_state

        # Stats pooling: concatenate mean and std
        mean = base_out.mean(dim=1)
        var = base_out.var(dim=1).clamp(min=1e-10)
        std = var.pow(0.5)
        x_stats = torch.cat((mean, std), dim=1).unsqueeze(dim=2)  # [batch, 2048, 1]

        # Top layers forward + L2 normalization
        return self.top_layers(x_stats)


class SpeakerEmbedder:
    """
    Simple speaker embedder for single audio -> embedding generation.

    Features:
        - Loads WavLM model once
        - Generates 128-dim L2-normalized speaker embeddings
        - Expects 16kHz audio input
        - Handles variable-length audio (max 20 seconds recommended)
        - Returns PyTorch tensors ready for TTS model

    Usage:
        embedder = SpeakerEmbedder()

        # From numpy array (16kHz)
        audio = np.random.randn(16000 * 5)  # 5 seconds
        embedding = embedder.embed_audio(audio)  # [1, 128]

        # From torch tensor
        audio_tensor = torch.randn(1, 16000 * 5)
        embedding = embedder.embed_audio(audio_tensor)
    """

    def __init__(
        self,
        model_name: str = "nineninesix/speaker-emb-tbr",
        device: Optional[str] = None,
        max_duration_sec: float = 30.0,
    ):
        """
        Initialize speaker embedder.

        Args:
            model_name: HuggingFace model ID
            device: Target device ('cuda', 'cpu', or None for auto-detect)
            max_duration_sec: Maximum audio duration in seconds (longer audio will be truncated)
        """
        self.model_name = model_name
        self.target_sr = 16000  # WavLM requires 16kHz
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * self.target_sr)

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        print(f"🔊 Loading WavLM speaker embedder from {model_name}...")
        self.model = EmbeddingsModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Speaker embedder ready on {self.device}")

    def _prepare_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Prepare audio for embedding extraction: resample if needed, convert to mono, truncate.

        Args:
            audio: Audio tensor (1D or 2D)
            sample_rate: Current sample rate of audio

        Returns:
            Prepared audio tensor [time_samples] at target_sr
        """
        # Handle multi-channel audio (convert to mono)
        if audio.dim() == 2:
            # If shape is [channels, time] where channels < time, average channels
            if audio.shape[0] < audio.shape[1]:
                audio = audio.mean(dim=0)
            # If shape is [batch, time], take first sample
            else:
                audio = audio[0]

        # Ensure 1D
        if audio.dim() != 1:
            raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")

        # Resample if needed
        if sample_rate != self.target_sr:
            try:
                import torchaudio.transforms as T
            except ImportError:
                raise ImportError(
                    "torchaudio is required for resampling. Install with: pip install torchaudio"
                )

            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            audio = resampler(audio)

        # Check audio length
        if audio.shape[0] == 0:
            raise ValueError("Audio is empty")

        # Truncate if too long
        if audio.shape[0] > self.max_samples:
            print(f"⚠️  Audio is {audio.shape[0] / sample_rate:.2f}s, truncating to {self.max_duration_sec}s")
            audio = audio[:self.max_samples]

        return audio

    def embed_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate speaker embedding from audio.

        Args:
            audio: Audio waveform as numpy array or torch tensor
                   - If 1D: shape [time_samples]
                   - If 2D: shape [batch, time_samples] or [channels, time_samples]
            sample_rate: Sample rate of input audio (if None, assumes 16kHz)

        Returns:
            Speaker embedding tensor [1, 128] (L2-normalized)

        Raises:
            ValueError: If audio is empty
        """
        # Convert to torch tensor if numpy
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio.float()

        # Default to target_sr if not specified
        if sample_rate is None:
            sample_rate = self.target_sr

        # Prepare audio: resample, convert to mono, truncate
        audio = self._prepare_audio(audio, sample_rate)

        # Add batch dimension [1, time_samples]
        audio_batch = audio.unsqueeze(0).to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model(audio_batch)  # [1, 128]

        return embedding

    def embed_audio_file(self, audio_path: str) -> torch.Tensor:
        """
        Generate speaker embedding from audio file.

        Args:
            audio_path: Path to audio file (supports any sample rate, will be resampled)

        Returns:
            Speaker embedding tensor [1, 128]

        Raises:
            ImportError: If torchaudio is not installed
            ValueError: If audio file cannot be loaded
        """
        try:
            import torchaudio
        except ImportError:
            raise ImportError("torchaudio is required for loading audio files. Install with: pip install torchaudio")

        # Load audio file
        audio, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio[0]

        return self.embed_audio(audio, sample_rate=sr)


# Convenience function for quick embedding generation
def compute_speaker_embedding(
    audio: Union[np.ndarray, torch.Tensor, str],
    sample_rate: int = 16000,
    model_name: str = "nineninesix/speaker-emb-tbr",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience function to generate speaker embedding in one line.

    Args:
        audio: Audio as numpy array, torch tensor, or file path
        sample_rate: Sample rate of audio (ignored if audio is file path)
        model_name: HuggingFace model ID
        device: Target device

    Returns:
        Speaker embedding [1, 128]

    Example:
        # From numpy array
        audio_np = np.random.randn(16000 * 5)
        emb = compute_speaker_embedding(audio_np)

        # From file
        emb = compute_speaker_embedding("speaker.wav")
    """
    embedder = SpeakerEmbedder(model_name=model_name, device=device)

    if isinstance(audio, str):
        return embedder.embed_audio_file(audio)
    else:
        return embedder.embed_audio(audio, sample_rate=sample_rate)
