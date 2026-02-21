"""Simple API for 9jaLingo TTS-2."""
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
import logging
import warnings
import torch
from .core import TTSConfig, NemoAudioPlayer, NaijaLingoModel


def suppress_all_logs():
    """
    Suppress all logging output from transformers, NeMo, PyTorch, and other libraries.
    Only print() statements from user code will be visible.
    """
    # Suppress Python warnings
    warnings.filterwarnings('ignore')

    # Suppress transformers logs
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass

    # Suppress NeMo logs
    logging.getLogger('nemo').setLevel(logging.ERROR)
    logging.getLogger('nemo_logger').setLevel(logging.ERROR)

    # Suppress PyTorch logs
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('pytorch').setLevel(logging.ERROR)

    # Suppress other common loggers
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    # Set root logger to ERROR level
    logging.getLogger().setLevel(logging.ERROR)


class NaijaLingoTTS:
    """
    Simple interface for 9jaLingo text-to-speech model.

    Supports 5 Nigerian languages + voice cloning:
        - Nigerian Accented English
        - Hausa
        - Igbo
        - Yoruba
        - Pidgin

    Example:
        >>> model = NaijaLingoTTS('9jalingo/your-model-name')
        >>> audio, text = model("Hello, world!")
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        max_new_tokens: int = 3000,
        tokeniser_length: int = 64400,
        suppress_logs: bool = True,
        show_info: bool = True,
        text_vocab_size: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        audio_step: Optional[float] = None,
        use_learnable_rope: Optional[bool] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        speaker_emb_dim: Optional[int] = None,
    ):
        """
        Initialize 9jaLingo TTS-2 model.

        Args:
            model_name: Hugging Face model ID or path to local model
            device_map: Device mapping for model (default: "auto")
            max_new_tokens: Maximum number of tokens to generate (default: 3000)
            tokeniser_length: Length of text tokenizer vocabulary (default: 64400)
            suppress_logs: Whether to suppress library logs (default: True)
            show_info: Whether to display model info on initialization (default: True)
            text_vocab_size: Text vocabulary size for position encoding.
                           If None, reads from model config. (default: None)
            tokens_per_frame: Number of audio tokens per frame.
                            If None, reads from model config. (default: None)
            audio_step: Position step size per audio frame.
                       If None, reads from model config. (default: None)
            use_learnable_rope: Enable learnable RoPE with per-layer alpha.
                              If None, reads from model config. (default: None)
            alpha_min: Minimum alpha value for learnable RoPE.
                      If None, reads from model config. (default: None)
            alpha_max: Maximum alpha value for learnable RoPE.
                      If None, reads from model config. (default: None)
            speaker_emb_dim: Dimension of speaker embeddings.
                           If None, reads from model config. (default: None)
        """
        if suppress_logs:
            suppress_all_logs()

        self.config = TTSConfig(
            device_map=device_map,
            tokeniser_length=tokeniser_length,
            max_new_tokens=max_new_tokens,
            text_vocab_size=text_vocab_size,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim,
        )
        self.model_name = model_name

        self.player = NemoAudioPlayer(self.config)
        self.model = NaijaLingoModel(self.config, model_name, self.player)
        self.status = self.model.status
        self.language_tags_list = self.model.language_tags_list
        self.sample_rate = self.config.sample_rate

        # Update config with actual values from loaded model (for display purposes)
        self._sync_config_from_model()

        if show_info:
            self.show_model_info()

    def _sync_config_from_model(self):
        """
        Synchronize config with actual values from loaded model.
        This ensures display shows the correct values that were loaded.
        """
        # Read actual values from the loaded model
        loaded_model = self.model.model

        if self.config.text_vocab_size is None:
            self.config.text_vocab_size = getattr(loaded_model, 'text_vocab_size', None)

        if self.config.tokens_per_frame is None:
            self.config.tokens_per_frame = getattr(loaded_model, 'tokens_per_frame', None)

        if self.config.audio_step is None:
            self.config.audio_step = getattr(loaded_model, 'audio_step', None)

        if self.config.use_learnable_rope is None:
            self.config.use_learnable_rope = getattr(loaded_model, 'use_learnable_rope', None)

        if self.config.alpha_min is None:
            self.config.alpha_min = getattr(loaded_model, 'alpha_min', None)

        if self.config.alpha_max is None:
            self.config.alpha_max = getattr(loaded_model, 'alpha_max', None)

        if self.config.speaker_emb_dim is None:
            self.config.speaker_emb_dim = getattr(loaded_model, 'speaker_emb_dim', None)

    def __call__(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb: Optional[Union[torch.Tensor, str, Path]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            language_tag: Optional language tag if model supports different languages or accents
            speaker_emb: Optional speaker embedding. Can be:
                - torch.Tensor: [1, speaker_emb_dim] or [speaker_emb_dim]
                - str/Path: Path to .pt file containing speaker embedding
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)

        Returns:
            Tuple of (audio_waveform, text) where audio_waveform is a numpy array
            containing the audio samples and text is the input text.
        """
        return self.generate(text, language_tag, speaker_emb, temperature, top_p, repetition_penalty)

    def generate(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb: Optional[Union[torch.Tensor, str, Path]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            language_tag: Optional language tag if model supports different languages or accents
            speaker_emb: Optional speaker embedding. Can be:
                - torch.Tensor: [1, speaker_emb_dim] or [speaker_emb_dim]
                - str/Path: Path to .pt file containing speaker embedding
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)

        Returns:
            Tuple of (audio_waveform, text) where audio_waveform is a numpy array
            containing the audio samples and text is the input text.
        """
        # Load speaker embedding if path is provided
        if speaker_emb is not None and not isinstance(speaker_emb, torch.Tensor):
            speaker_emb = self.load_speaker_embedding(speaker_emb)

        # Ensure speaker_emb has batch dimension
        if speaker_emb is not None and speaker_emb.ndim == 1:
            speaker_emb = speaker_emb.unsqueeze(0)

        return self.model.run_model(text, language_tag, speaker_emb, temperature, top_p, repetition_penalty)

    def load_speaker_embedding(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load speaker embedding from a .pt file.

        Args:
            path: Path to .pt file containing speaker embedding

        Returns:
            Speaker embedding tensor [speaker_emb_dim] or [1, speaker_emb_dim]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Speaker embedding file not found: {path}")

        if path.suffix != '.pt':
            raise ValueError(f"Speaker embedding must be a .pt file, got: {path.suffix}")

        speaker_emb = torch.load(path)

        # Validate shape
        if speaker_emb.ndim == 1:
            expected_dim = self.config.speaker_emb_dim
            if speaker_emb.shape[0] != expected_dim:
                raise ValueError(
                    f"Speaker embedding has wrong dimension: expected {expected_dim}, "
                    f"got {speaker_emb.shape[0]}"
                )
        elif speaker_emb.ndim == 2:
            if speaker_emb.shape[1] != self.config.speaker_emb_dim:
                raise ValueError(
                    f"Speaker embedding has wrong dimension: expected [..., {self.config.speaker_emb_dim}], "
                    f"got {speaker_emb.shape}"
                )
        else:
            raise ValueError(f"Speaker embedding must be 1D or 2D, got shape: {speaker_emb.shape}")

        return speaker_emb

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio waveform to file.

        Args:
            audio: Audio waveform as numpy array
            output_path: Path to save audio file (e.g., "output.wav")
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio. "
                "Install it with: pip install soundfile"
            )

    def show_model_info(self):
        """
        Display model information banner.
        """
        print()
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                                                            ║")
        print("║                    9 j a L i n g o  TTS-2                  ║")
        print("║                                                            ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("     🗣️  Nigerian Language Text-to-Speech Engine")
        print()
        print("─" * 62)

        # Model name
        model_display = self.model_name
        if len(model_display) > 50:
            model_display = "..." + model_display[-47:]
        print(f"  Model: {model_display}")

        # Device info
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"  Device: {device}")

        # Language tags info
        if self.status == 'available_language_tags':
            print(f"  Mode: Available language tags ({len(self.language_tags_list)} language tags)")
            if self.language_tags_list and len(self.language_tags_list) <= 6:
                lang_str = ", ".join(self.language_tags_list)
                print(f"  Tags: {lang_str}")
            elif self.language_tags_list:
                print(f"  Tags: {self.language_tags_list[0]}, {self.language_tags_list[1]}, ... (use .show_language_tags() to see all)")
        else:
            print(f"  Mode: No language tags")

        print()
        print("  Configuration:")
        print(f"    • Sample Rate: {self.sample_rate} Hz")
        print(f"    • Max Tokens: {self.config.max_new_tokens}")
        print(f"    • Speaker Embedding Dim: {self.config.speaker_emb_dim or 'Unknown'}")
        print(f"    • Text Vocab Size: {self.config.text_vocab_size or 'Unknown'}")
        print(f"    • Tokens per Frame: {self.config.tokens_per_frame or 'Unknown'}")
        print(f"    • Audio Step: {self.config.audio_step or 'Unknown'}")
        if self.config.use_learnable_rope:
            print(f"    • Learnable RoPE: Enabled (per-layer frequency scaling)")
            print(f"    • Alpha Range: [{self.config.alpha_min or 'Unknown'}, {self.config.alpha_max or 'Unknown'}]")
        else:
            print(f"    • Learnable RoPE: Disabled (standard RoPE)")

        print("─" * 62)
        print()
        print("  Supported: English (Nigerian) | Hausa | Igbo | Yoruba | Pidgin")
        print("  Voice Cloning: Enabled 🎙️")
        print()
        print("  Ready to generate speech! 🎵")
        print()

    def show_language_tags(self)->None:

        print("=" * 50)
        if self.status == 'available_language_tags':
            print("Available language tags:")
            print("-" * 50)
            if self.language_tags_list:
                for i, tag in enumerate(self.language_tags_list, 1):
                    print(f"  {i}. {tag}")
            else:
                print("  No tags configured")
        else:
            print("This model does not support language tag selection.")
        print("=" * 50)
