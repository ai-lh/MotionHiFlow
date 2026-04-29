"""
Frozen Text Encoder Module

This module provides a unified interface for encoding text using various pretrained
language models (CLIP, SigLIP2, T5, DistilBERT). Models are frozen (non-trainable)
and cached for efficient reuse.

Supported Models:
    - CLIP ViT-B/32
    - SigLIP2 Base
    - Flan-T5 Base
    - DistilBERT Base Uncased
"""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable, cast, Any
from collections import OrderedDict

import torch
import torch.nn as nn
from pathlib import Path
from os.path import join as pjoin

from transformers import AutoModel, AutoTokenizer, Siglip2TextModel, PreTrainedTokenizer

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry and Caching Utilities
# =============================================================================

class FrozenModels:
    """
    A singleton-like model registry that caches pretrained models and tokenizers.
    
    This class prevents redundant model loading by maintaining a shared cache
    of models and their corresponding tokenizers. Once a model is loaded,
    subsequent requests for the same model return the cached instance.
    
    Attributes:
        models (Dict[str, nn.Module]): Class-level cache for loaded models.
        tokenizers (Dict[str, AutoTokenizer]): Class-level cache for tokenizers.
        dir (str): Default directory path for model checkpoints.
    
    Example:
        >>> models = FrozenModels('./deps')
        >>> clip_model, clip_tokenizer = models('clip-vit-base-patch32')
        >>> # Second call returns cached instance
        >>> same_model, same_tokenizer = models('clip-vit-base-patch32')
    """
    
    # Class-level caches shared across all instances
    models: Dict[str, nn.Module] = {}
    tokenizers: Dict[str, PreTrainedTokenizer] = {}
    
    # Mapping of model names to their loader classes
    MODEL_LOADERS: Dict[str, type] = {
        'siglip2-base-patch16-512': Siglip2TextModel,
    }
    
    def __init__(self, dir: str = ''):
        """
        Initialize the FrozenModels registry.
        
        Args:
            dir (str): Default directory path where model checkpoints are stored.
        """
        self.dir = dir
    
    def __call__(
        self, 
        model_name: str, 
        dir: Optional[str] = None
    ) -> Tuple[nn.Module, PreTrainedTokenizer]:
        """
        Load and return a model-tokenizer pair, using cache if available.
        
        Args:
            model_name (str): Name of the model to load (must match folder name).
            dir (Optional[str]): Override directory path. Uses default if None.
        
        Returns:
            Tuple[nn.Module, PreTrainedTokenizer]: The loaded model and its tokenizer.
        
        Raises:
            OSError: If the model checkpoint directory does not exist.
        """
        if dir is None:
            dir = self.dir
        
        # Return cached model if available
        if model_name not in self.models:
            model_path = pjoin(dir, model_name)
            logger.info(f"Loading model '{model_name}' from {model_path}")
            
            # Select appropriate loader based on model type
            if model_name in self.MODEL_LOADERS:
                loader_class = self.MODEL_LOADERS[model_name]
                self.models[model_name] = loader_class.from_pretrained(model_path)
            else:
                self.models[model_name] = AutoModel.from_pretrained(model_path)
            
            # Load corresponding tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Successfully loaded model '{model_name}'")
        
        return self.models[model_name], self.tokenizers[model_name]
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear all cached models and tokenizers to free memory.
        
        This is useful when switching between different model configurations
        or when memory needs to be reclaimed.
        """
        cls.models.clear()
        cls.tokenizers.clear()
        logger.info("Model cache cleared")
    
    @classmethod
    def list_cached_models(cls) -> List[str]:
        """
        Return a list of currently cached model names.
        
        Returns:
            List[str]: Names of models currently in cache.
        """
        return list(cls.models.keys())


class EmbeddingCache:
    """
    A cache for storing computed text embeddings to avoid redundant computation.
    
    This cache stores the pooled embeddings of previously processed texts,
    significantly speeding up repeated encoding of the same text strings.
    
    Attributes:
        cached_pooled (Dict[str, torch.Tensor]): Mapping from text to embedding.
        max_size (int): Maximum number of entries to cache (0 = unlimited).
    
    Example:
        >>> cache = EmbeddingCache(max_size=1000)
        >>> embeddings = cache(["hello", "world"], generate_fn, device='cuda')
    """
    
    def __init__(self, max_size: int = 0):
        """
        Initialize the embedding cache.
        
        Args:
            max_size (int): Maximum cache size. 0 means unlimited.
        """
        self.cached_pooled: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.max_size = max_size
    
    def __call__(
        self, 
        texts: List[str], 
        generate: Optional[Callable[[List[str]], torch.Tensor]], 
        device: Union[str, torch.device] = 'cpu'
    ) -> torch.Tensor:
        """
        Retrieve embeddings from cache or generate them if not cached.
        
        Args:
            texts (List[str]): List of text strings to get embeddings for.
            generate (Optional[Callable]): Function to generate embeddings for
                texts not in cache. Should accept List[str] and return Tensor.
            device (Union[str, torch.device]): Target device for output tensor.
        
        Returns:
            torch.Tensor: Stacked embeddings of shape (len(texts), embed_dim).
        
        Raises:
            KeyError: If text is not cached and generate is None.
        """
        # Identify texts that need to be generated
        not_cached = [text for text in texts if text not in self.cached_pooled]
        
        # Generate embeddings for uncached texts
        if generate is not None and len(not_cached) > 0:
            logger.debug(f"Generating embeddings for {len(not_cached)} uncached texts")
            results = generate(not_cached)
            
            for text, result in zip(not_cached, results):
                # Handle cache size limit with LRU eviction
                if self.max_size > 0 and len(self.cached_pooled) >= self.max_size:
                    self._evict_oldest()
                
                # Store embedding on CPU to save GPU memory
                self.cached_pooled[text] = result.cpu()
        
        # Retrieve all embeddings from cache and update LRU position
        cached = []
        for text in texts:
            self.cached_pooled.move_to_end(text)
            cached.append(self.cached_pooled[text])
        
        return torch.stack(cached, dim=0).to(device=device)
    
    def _evict_oldest(self) -> None:
        """Remove the least recently added/accessed item from cache (O(1) LRU)."""
        if self.cached_pooled:
            self.cached_pooled.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cached_pooled.clear()
        logger.debug("Embedding cache cleared")
    
    def __len__(self) -> int:
        """Return the number of cached embeddings."""
        return len(self.cached_pooled)
    
    @property
    def memory_usage_mb(self) -> float:
        """
        Estimate the memory usage of cached embeddings in megabytes.
        
        Returns:
            float: Estimated memory usage in MB.
        """
        total_bytes = sum(
            tensor.numel() * tensor.element_size() 
            for tensor in self.cached_pooled.values()
        )
        return total_bytes / (1024 * 1024)


# =============================================================================
# Model Configuration Constants
# =============================================================================

# Model name to checkpoint directory mapping
MODEL_CHECKPOINTS = {
    "ViT-B/32": "clip-vit-base-patch32",
    "Siglip2-base": "siglip2-base-patch16-512",
    "t5-base": "flan-t5-base",
    "distilbert-base-uncased": "distilbert-base-uncased",
}

# Model output dimensions
MODEL_DIMENSIONS = {
    "ViT-B/32": 512,
    "Siglip2-base": 768,
    "t5-base": 768,
    "distilbert-base-uncased": 768,
}

# Maximum sequence lengths for each model
MODEL_MAX_LENGTHS = {
    "ViT-B/32": 77,
    "Siglip2-base": 64,
    "t5-base": 512,
    "distilbert-base-uncased": 512,
}


# =============================================================================
# Main Text Encoder Class
# =============================================================================

class FrozenTextEncoder(nn.Module):
    """
    A frozen (non-trainable) text encoder that supports multiple pretrained models.
    
    This encoder provides a unified interface for encoding text using various
    pretrained models including CLIP, SigLIP2, T5, and DistilBERT. It supports:
    
    - Single encoder mode: Uses one model for both word embeddings and pooled output
    - Dual encoder mode: Uses different models for word embeddings and pooled output
    - Optional secondary encoder: Concatenates embeddings from two text encoders
    
    The encoder is frozen by default, meaning all parameters have requires_grad=False.
    This makes it suitable for use in pipelines where only downstream components
    should be trained.
    
    Attributes:
        text_dim (int): Dimension of the primary text encoder output.
        pooled_dim (int): Dimension of the pooled encoder output.
        text_model (nn.Module): Primary text encoder model.
        pooled_model (nn.Module): Model used for pooled representations.
        text_model2 (Optional[nn.Module]): Optional secondary text encoder.
        cache (EmbeddingCache): Cache for pooled embeddings.
    
    Example:
        >>> encoder = FrozenTextEncoder(
        ...     text_encoder="ViT-B/32",
        ...     pooled_encoder="ViT-B/32",
        ...     checkpoints_dir="./models"
        ... )
        >>> word_emb, attn_mask, pooled = encoder.encode_text(["Hello world!"])
        >>> print(word_emb.shape)  # (1, seq_len, 768)
        >>> print(pooled.shape)    # (1, 512)
    """
    
    # Supported encoder types for validation
    SUPPORTED_TEXT_ENCODERS = {"ViT-B/32", "Siglip2-base", "t5-base"}
    SUPPORTED_POOLED_ENCODERS = {"ViT-B/32", "Siglip2-base"}
    SUPPORTED_SECONDARY_ENCODERS = {"distilbert-base-uncased", "t5-base", "Siglip2-base"}
    
    def __init__(
        self,
        text_encoder: str = "ViT-B/32",
        pooled_encoder: str = "ViT-B/32",
        checkpoints_dir: str = './deps',
        text_encoder2: Optional[str] = None,
        cache_max_size: int = 10000,
    ):
        """
        Initialize the frozen text encoder.
        
        Args:
            text_encoder (str): Primary text encoder type. 
                Options: "ViT-B/32", "Siglip2-base", "t5-base"
            pooled_encoder (str): Encoder for pooled representations.
                Options: "ViT-B/32", "Siglip2-base"
            checkpoints_dir (str): Directory containing model checkpoints.
            text_encoder2 (Optional[str]): Optional secondary text encoder for
                concatenated embeddings.
                Options: "distilbert-base-uncased", "t5-base", "Siglip2-base"
            cache_max_size (int): Maximum number of pooled embeddings to cache.
                Set to 0 for unlimited cache.
        
        Raises:
            ValueError: If an invalid encoder type is specified.
        """
        super().__init__()
        
        # Store configuration for later reference
        self.opt = {
            'text_encoder': text_encoder,
            'pooled_encoder': pooled_encoder,
            'checkpoints_dir': checkpoints_dir,
            'text_encoder2': text_encoder2,
        }
        
        # Disable tokenizer parallelism to avoid warnings in multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Validate encoder choices
        self._validate_encoder_choices(text_encoder, pooled_encoder, text_encoder2)
        
        # Set output dimensions based on encoder types
        self.text_dim = MODEL_DIMENSIONS.get(text_encoder, 768)
        self.pooled_dim = MODEL_DIMENSIONS.get(pooled_encoder, 768)
        
        # Initialize embedding cache
        self.cache = EmbeddingCache(max_size=cache_max_size)
        
        # Load models using the shared model registry
        models = FrozenModels(checkpoints_dir)
        
        # Initialize primary text encoder
        self.max_length = MODEL_MAX_LENGTHS.get(text_encoder, 77)
        self.text_model, self.text_tokenizer = self._load_text_encoder(
            models, text_encoder
        )
        
        # Initialize pooled encoder (may share with text encoder)
        self.pooled_model, self.pooled_tokenizer = self._load_pooled_encoder(
            models, pooled_encoder
        )
        
        # Initialize optional secondary text encoder
        self.max_length2 = 77
        self.text_dim2 = None
        if text_encoder2 is not None:
            self.text_model2, self.text_tokenizer2, self.text_dim2, self.max_length2 = \
                self._load_secondary_encoder(models, text_encoder2)
        else:
            self.text_model2 = None
            self.text_tokenizer2 = None
        
        # Initialize cached outputs (for avoiding redundant computation)
        self._reset_cache()
        
        # Freeze all parameters
        self.freeze()
        
        # Log initialization summary
        logger.info(
            f"Initialized FrozenTextEncoder: "
            f"text_encoder={text_encoder} (dim={self.text_dim}), "
            f"pooled_encoder={pooled_encoder} (dim={self.pooled_dim})"
            + (f", text_encoder2={text_encoder2}" if text_encoder2 else "")
        )
    
    def _validate_encoder_choices(
        self, 
        text_encoder: str, 
        pooled_encoder: str, 
        text_encoder2: Optional[str]
    ) -> None:
        """
        Validate that the specified encoder types are supported.
        
        Args:
            text_encoder: Primary text encoder type.
            pooled_encoder: Pooled encoder type.
            text_encoder2: Optional secondary encoder type.
        
        Raises:
            ValueError: If any encoder type is not supported.
        """
        if text_encoder not in self.SUPPORTED_TEXT_ENCODERS:
            raise ValueError(
                f"Invalid text encoder: {text_encoder}. "
                f"Supported: {self.SUPPORTED_TEXT_ENCODERS}"
            )
        
        if pooled_encoder not in self.SUPPORTED_POOLED_ENCODERS:
            raise ValueError(
                f"Invalid pooled encoder: {pooled_encoder}. "
                f"Supported: {self.SUPPORTED_POOLED_ENCODERS}"
            )
        
        if text_encoder2 is not None and text_encoder2 not in self.SUPPORTED_SECONDARY_ENCODERS:
            raise ValueError(
                f"Invalid secondary text encoder: {text_encoder2}. "
                f"Supported: {self.SUPPORTED_SECONDARY_ENCODERS}"
            )
    
    def _load_text_encoder(
        self, 
        models: FrozenModels, 
        encoder_type: str
    ) -> Tuple[nn.Module, PreTrainedTokenizer]:
        """
        Load the primary text encoder model and tokenizer.
        
        Args:
            models: FrozenModels registry instance.
            encoder_type: Type of encoder to load.
        
        Returns:
            Tuple of (model, tokenizer).
        """
        checkpoint_name = MODEL_CHECKPOINTS[encoder_type]
        model, tokenizer = models(checkpoint_name)
        
        # Extract the text encoder component based on model architecture
        if encoder_type in ["ViT-B/32"]:
            # CLIP models have a text_model attribute; cast to nn.Module for typing
            return cast(nn.Module, model.text_model), tokenizer
        elif encoder_type == "Siglip2-base":
            # SigLIP2 text model is loaded directly
            return model, tokenizer
        elif encoder_type == "t5-base":
            # T5 encoder is the encoder component
            return cast(nn.Module, model.encoder), tokenizer
        else:
            return model, tokenizer
    
    def _load_pooled_encoder(
        self, 
        models: FrozenModels, 
        encoder_type: str
    ) -> Tuple[nn.Module, PreTrainedTokenizer]:
        """
        Load the pooled encoder model and tokenizer.
        
        Args:
            models: FrozenModels registry instance.
            encoder_type: Type of encoder to load.
        
        Returns:
            Tuple of (model, tokenizer).
        """
        checkpoint_name = MODEL_CHECKPOINTS[encoder_type]
        model, tokenizer = models(checkpoint_name)
        
        if encoder_type in ["ViT-B/32"]:
            # text_model can be typed as Tensor | Module; cast to Module
            return cast(nn.Module, model.text_model), tokenizer
        elif encoder_type == "Siglip2-base":
            return model, tokenizer
        else:
            return model, tokenizer
    
    def _load_secondary_encoder(
        self, 
        models: FrozenModels, 
        encoder_type: str
    ) -> Tuple[nn.Module, PreTrainedTokenizer, int, int]:
        """
        Load the optional secondary text encoder.
        
        Args:
            models: FrozenModels registry instance.
            encoder_type: Type of encoder to load.
        
        Returns:
            Tuple of (model, tokenizer, embedding_dim, max_length).
        """
        checkpoint_name = MODEL_CHECKPOINTS[encoder_type]
        model, tokenizer = models(checkpoint_name)
        dim = MODEL_DIMENSIONS[encoder_type]
        max_len = MODEL_MAX_LENGTHS.get(encoder_type, 77)
        
        if encoder_type == "t5-base":
            return cast(nn.Module, model.encoder), tokenizer, dim, max_len
        elif encoder_type == "Siglip2-base":
            # Ensure static typing: cast text_model to nn.Module
            return cast(nn.Module, model.text_model), tokenizer, dim, 64
        else:
            return model, tokenizer, dim, max_len
    
    def _reset_cache(self) -> None:
        """Reset the internal result cache."""
        self.word_emb: Optional[torch.Tensor] = None
        self.text_attn_mask: Optional[torch.Tensor] = None
        self.pooled_output: Optional[torch.Tensor] = None
        self.text: List[str] = []
    
    @property
    def device(self) -> torch.device:
        """
        Get the device where the model parameters are located.
        
        Returns:
            torch.device: The device of the model.
        """
        return next(self.text_model.parameters()).device
    
    def freeze(self) -> None:
        """
        Freeze all model parameters to prevent gradient computation.
        
        This sets all models to eval mode and disables gradient tracking
        for all parameters.
        """
        self.text_model.eval()
        self.pooled_model.eval()
        
        if self.text_model2 is not None:
            self.text_model2.eval()
        
        for param in self.parameters():
            param.requires_grad = False
        
        logger.debug("All encoder parameters frozen")
    
    @torch.no_grad()
    def encode_text(
        self, 
        text: List[str], 
        drop_text: float = 0.0,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a list of text strings into embeddings.
        
        This method produces three outputs:
        1. Word embeddings: Token-level representations for each input text
        2. Attention mask: Boolean mask indicating valid tokens
        3. Pooled output: Single vector representation per text
        
        Args:
            text (List[str]): List of text strings to encode.
            drop_text (float): Probability of dropping (zeroing) each text input.
                Used for classifier-free guidance training. Range: [0.0, 1.0].
            use_cache (bool): Whether to use cached results for identical inputs.
        
        Returns:
            Tuple containing:
                - word_emb (torch.Tensor): Shape (batch, seq_len, embed_dim)
                - text_attn_mask (torch.Tensor): Shape (batch, seq_len), boolean
                - pooled_output (torch.Tensor): Shape (batch, pooled_dim)
        
        Note:
            When drop_text > 0 and a secondary encoder is used, the effective
            drop probability for each encoder is sqrt(drop_text), ensuring
            the combined drop probability equals drop_text.
        """
        # Return cached result if input hasn't changed
        if use_cache and text == self.text and drop_text == 0.:
            assert self.word_emb is not None and self.text_attn_mask is not None and self.pooled_output is not None, "Cache is incomplete"
            return self.word_emb, self.text_attn_mask, self.pooled_output
        
        self.text = text
        
        # Calculate per-encoder drop probability
        # For dual encoder: sqrt(p) * sqrt(p) = p
        prob = math.sqrt(drop_text) if self.text_model2 is not None else drop_text
        
        # Apply text dropout (classifier-free guidance)
        drop_mask = torch.rand(len(text)) < prob
        texts = ["" if drop_mask[i] else text[i] for i in range(len(text))]
        
        # Tokenize input texts
        tokens = self.text_tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        text_input_ids = tokens.input_ids.to(self.device)
        text_attn_mask = tokens.attention_mask.to(self.device).bool()
        
        # Forward pass through primary text encoder
        output = self.text_model(text_input_ids, attention_mask=text_attn_mask)
        word_emb = output.last_hidden_state
        
        # Get pooled output (from same model or separate pooled encoder)
        if self.opt['pooled_encoder'] == self.opt['text_encoder']:
            pooled_output = output.pooler_output
        else:
            # Use cached pooled embeddings if available
            def generate_pooled(uncached_texts: List[str]) -> torch.Tensor:
                if self.pooled_tokenizer is self.text_tokenizer:
                    indices = [texts.index(t) for t in uncached_texts]
                    input_ids = text_input_ids[indices]
                    attn_mask = text_attn_mask[indices]
                    return self.get_pooled_output(uncached_texts, input_ids=input_ids, attention_mask=attn_mask)
                else:
                    return self.get_pooled_output(uncached_texts)

            pooled_output = self.cache(texts, generate_pooled, self.device)
        
        # Process secondary encoder if configured
        if self.text_model2 is not None and self.text_tokenizer2 is not None:
            word_emb, text_attn_mask = self._encode_with_secondary(
                text, prob, word_emb, text_attn_mask
            )
        
        # Cache results for potential reuse
        self.word_emb = word_emb
        self.text_attn_mask = text_attn_mask
        self.pooled_output = pooled_output
        
        return word_emb, text_attn_mask, pooled_output
    
    def _encode_with_secondary(
        self,
        text: List[str],
        drop_prob: float,
        word_emb: torch.Tensor,
        text_attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text with the secondary encoder and concatenate with primary.
        
        Args:
            text: Original text inputs.
            drop_prob: Probability of text dropout.
            word_emb: Embeddings from primary encoder.
            text_attn_mask: Attention mask from primary encoder.
        
        Returns:
            Tuple of concatenated (word_emb, text_attn_mask).
        """
        # Apply independent dropout for secondary encoder
        drop_mask = torch.rand(len(text)) < drop_prob
        texts = ["" if drop_mask[i] else text[i] for i in range(len(text))]
        
        # Tokenize for secondary encoder
        assert self.text_tokenizer2 is not None, "Secondary tokenizer is not initialized"
        tokens2 = self.text_tokenizer2(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length2,
            return_tensors="pt",
            return_attention_mask=True
        )
        text_input_ids2 = tokens2.input_ids.to(self.device)
        text_attn_mask2 = tokens2.attention_mask.to(self.device).bool()
        
        # Forward pass through secondary encoder
        assert self.text_model2 is not None, "Secondary text model is not initialized"
        output2 = self.text_model2(text_input_ids2, attention_mask=text_attn_mask2)
        word_emb2 = output2.last_hidden_state
        
        # Align embedding dimensions through zero-padding
        if word_emb.shape[-1] < word_emb2.shape[-1]:
            word_emb = torch.nn.functional.pad(
                word_emb, (0, word_emb2.shape[-1] - word_emb.shape[-1])
            )
        elif word_emb.shape[-1] > word_emb2.shape[-1]:
            word_emb2 = torch.nn.functional.pad(
                word_emb2, (0, word_emb.shape[-1] - word_emb2.shape[-1])
            )
        
        # Concatenate along sequence dimension
        word_emb = torch.cat([word_emb, word_emb2], dim=1)
        text_attn_mask = torch.cat([text_attn_mask, text_attn_mask2], dim=1)
        
        return word_emb, text_attn_mask
    
    @torch.no_grad()
    def get_pooled_output(
        self, 
        texts: List[str],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pooled text representations using the pooled encoder.
        
        This method is used when the pooled encoder differs from the
        primary text encoder, providing a separate pooled representation.
        
        Args:
            texts (List[str]): List of text strings to encode.
        
        Returns:
            torch.Tensor: Pooled embeddings of shape (len(texts), pooled_dim).
        """
        if input_ids is not None and attention_mask is not None:
            text_input_ids = input_ids
            text_attn_mask = attention_mask
        else:
            tokens = self.pooled_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            text_input_ids = tokens.input_ids.to(self.device)
            text_attn_mask = tokens.attention_mask.to(self.device).bool()
        
        pooled_output = self.pooled_model(
            text_input_ids, attention_mask=text_attn_mask
        ).pooler_output
        
        return pooled_output
    
    def clear_caches(self) -> None:
        """
        Clear all internal caches.
        
        This includes:
        - The embedding cache for pooled outputs
        - The internal result cache for repeated encode_text calls
        """
        self.cache.clear()
        self._reset_cache()
        logger.debug("All caches cleared")
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """
        Get the output dimensions for all encoder components.
        
        Returns:
            Dict with keys 'text_dim', 'pooled_dim', and optionally 'text_dim2'.
        """
        dims = {
            'text_dim': self.text_dim,
            'pooled_dim': self.pooled_dim,
        }
        if self.text_dim2 is not None:
            dims['text_dim2'] = self.text_dim2
        return dims
    
    def to(self, device: Union[str, torch.device]) -> 'FrozenTextEncoder':
        """
        Move the encoder to the specified device.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).
        
        Returns:
            self: The encoder instance (for method chaining).
        """
        super().to(device)
        # Clear cache since embeddings are stored on CPU
        self._reset_cache()
        return self
    
    def __repr__(self) -> str:
        """Return a string representation of the encoder configuration."""
        return (
            f"FrozenTextEncoder(\n"
            f"  text_encoder={self.opt['text_encoder']},\n"
            f"  pooled_encoder={self.opt['pooled_encoder']},\n"
            f"  text_encoder2={self.opt.get('text_encoder2')},\n"
            f"  text_dim={self.text_dim},\n"
            f"  pooled_dim={self.pooled_dim},\n"
            f"  device={self.device}\n"
            f")"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_text_encoder(
    preset: str = "default",
    checkpoints_dir: str = "./deps"
) -> FrozenTextEncoder:
    """
    Factory function to create text encoders with common configurations.
    
    Presets:
        - "default": ViT-B/32 for both text and pooled
        - "siglip": Siglip2-base for both text and pooled
        - "dual": ViT-B/32 + t5-base secondary
    
    Args:
        preset (str): Configuration preset name.
        checkpoints_dir (str): Directory containing model checkpoints.
    
    Returns:
        FrozenTextEncoder: Configured encoder instance.
    
    Raises:
        ValueError: If preset is not recognized.
    
    Example:
        >>> encoder = create_text_encoder("large", "./models")
    """
    presets = {
        "default": {
            "text_encoder": "ViT-B/32",
            "pooled_encoder": "ViT-B/32",
        },
        "siglip": {
            "text_encoder": "Siglip2-base",
            "pooled_encoder": "Siglip2-base",
        },
        "dual": {
            "text_encoder": "ViT-B/32",
            "pooled_encoder": "ViT-B/32",
            "text_encoder2": "t5-base",
        },
    }
    
    if preset not in presets:
        raise ValueError(
            f"Unknown preset: {preset}. Available: {list(presets.keys())}"
        )
    
    config = cast(Dict[str, Any], presets[preset])
    return FrozenTextEncoder(checkpoints_dir=checkpoints_dir, **config)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    """
    Simple test to verify the module works correctly.
    Run with: python text_encoder.py
    """
    print("=" * 60)
    print("Testing FrozenTextEncoder")
    print("=" * 60)
    
    # Note: This requires the model checkpoints to be available
    try:
        encoder = FrozenTextEncoder(
            text_encoder="ViT-B/32",
            pooled_encoder="ViT-B/32",
            checkpoints_dir="./deps"
        )
        
        test_texts = [
            "A photo of a cat",
            "A beautiful sunset over the ocean",
            "A person walking in the park"
        ]
        
        word_emb, attn_mask, pooled = encoder.encode_text(test_texts)
        
        print(f"\nInput texts: {test_texts}")
        print(f"Word embeddings shape: {word_emb.shape}")
        print(f"Attention mask shape: {attn_mask.shape}")
        print(f"Pooled output shape: {pooled.shape}")
        print(f"\nEncoder info:\n{encoder}")

        word_emb2, attn_mask2, pooled2 = encoder.encode_text(test_texts)
        assert torch.allclose(word_emb, word_emb2), "Cached word embeddings do not match"
        assert torch.allclose(attn_mask.float(), attn_mask2.float()), "Cached attention masks do not match"
        assert torch.allclose(pooled, pooled2), "Cached pooled outputs do not match"
        print("\nCache test passed: repeated calls return identical results.")
        print("\nTest passed!")
        
    except Exception as e:
        print(f"Test failed (this may be expected if checkpoints are not available): {e}")
        import pdb; pdb.set_trace()