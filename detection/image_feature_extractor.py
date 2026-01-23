"""
Image Feature Extractor for Plagiarism Detection
=================================================
Extracts visual embeddings and features from images using multiple methods:
- CNN features (CLIP or ResNet)
- Perceptual hashes (pHash, dHash, aHash)
- OCR text extraction
- Orientation-invariant features

This is the main class used by run_system.py for embedding images.
"""

import os
import io
import logging
import hashlib
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class ImageFeatures:
    """Container for all extracted image features"""
    # CNN embedding vector
    cnn_vector: Optional[np.ndarray] = None
    
    # Perceptual hashes for different orientations
    orientation_hashes: Dict[str, str] = field(default_factory=dict)
    
    # Primary perceptual hash
    phash: Optional[str] = None
    dhash: Optional[str] = None
    ahash: Optional[str] = None
    
    # OCR data
    ocr_regions: Optional[List[Dict]] = None
    text_hash: Optional[str] = None
    full_text: Optional[str] = None
    
    # Metadata
    width: int = 0
    height: int = 0
    paper_id: str = ""
    image_id: str = ""


class ImageFeatureExtractor:
    """
    Extract visual features from images using deep learning models.
    
    Also known as: MultiMethodFeatureExtractor (legacy name)
    
    Features extracted:
    - CNN embeddings (CLIP or ResNet)
    - Perceptual hashes (rotation invariant)
    - OCR text regions
    """
    
    def __init__(
        self,
        model_name: str = "resnet",  # "clip" or "resnet"
        device: str = None,
        enable_ocr: bool = True,
        use_gpu: bool = True  # Added for backward compatibility
    ):
        """
        Initialize the image feature extractor.
        
        Args:
            model_name: Model to use ("clip" or "resnet")
            device: Device to use ("cuda" or "cpu")
            enable_ocr: Whether to enable OCR extraction
            use_gpu: Whether to use GPU (alternative to device parameter)
        """
        self.model_name = model_name
        
        # Handle device parameter - use_gpu takes precedence if device not specified
        if device:
            self.device = device
        elif use_gpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        self.model = None
        self.processor = None
        self.transform = None
        self.embedding_dim = 512  # Default for CLIP
        self.enable_ocr = enable_ocr
        self.ocr_reader = None
        
        self._load_model()
        
        if enable_ocr:
            self._init_ocr()
    
    def _load_model(self):
        """Load the feature extraction model."""
        try:
            if self.model_name == "clip":
                self._load_clip()
            else:
                self._load_resnet()
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name} model: {e}, trying ResNet")
            if self.model_name == "clip":
                self._load_resnet()
            else:
                raise
    
    def _load_clip(self):
        """Load CLIP model for image embeddings."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_id = "openai/clip-vit-base-patch32"
            logger.info(f"Loading CLIP model: {model_id}")
            
            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            
            self.embedding_dim = 512
            logger.info(f"CLIP model loaded (device: {self.device})")
            
        except ImportError:
            logger.warning("transformers not installed, falling back to ResNet")
            self._load_resnet()
    
    def _load_resnet(self):
        """Load ResNet model for image embeddings."""
        from torchvision import models, transforms
        
        logger.info("Loading ResNet50 model")
        
        # Load pretrained ResNet50, remove final classification layer
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.embedding_dim = 2048
        self.model_name = "resnet"
        logger.info(f"ResNet50 model loaded (device: {self.device})")
    
    def _init_ocr(self):
        """Initialize OCR reader."""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=self.device == 'cuda')
            logger.info("EasyOCR initialized")
        except ImportError:
            logger.warning("easyocr not installed, OCR disabled")
            self.enable_ocr = False
        except Exception as e:
            logger.warning(f"Failed to init OCR: {e}")
            self.enable_ocr = False
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Load and prepare image for processing."""
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            return img
        except Exception as e:
            logger.error(f"Failed to prepare image: {e}")
            return None
    
    def extract_features(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract CNN feature vector from an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature vector as numpy array, or None if failed
        """
        img = self._prepare_image(image)
        if img is None:
            return None
        
        try:
            with torch.no_grad():
                if self.model_name == "clip" and self.processor:
                    # CLIP processing
                    inputs = self.processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    features = self.model.get_image_features(**inputs)
                    embedding = features.cpu().numpy().flatten()
                else:
                    # ResNet processing
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    features = self.model(img_tensor)
                    embedding = features.cpu().numpy().flatten()
            
            # Normalize
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract CNN features: {e}")
            return None
    
    def extract_all_features(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        paper_id: str = "",
        image_id: str = ""
    ) -> ImageFeatures:
        """
        Extract ALL features from an image (CNN + hashes + OCR).
        
        This is the main method used by run_system.py for embedding.
        
        Args:
            image: Image to process
            paper_id: Paper DOI/ID for metadata
            image_id: Image identifier (e.g., "fig_0", "table_1")
            
        Returns:
            ImageFeatures with all extracted data
        """
        features = ImageFeatures(paper_id=paper_id, image_id=image_id)
        
        img = self._prepare_image(image)
        if img is None:
            return features
        
        features.width, features.height = img.size
        
        # 1. CNN embedding
        features.cnn_vector = self.extract_features(img, normalize=True)
        
        # 2. Perceptual hashes (rotation invariant)
        features.orientation_hashes = self._compute_orientation_hashes(img)
        if features.orientation_hashes:
            features.phash = features.orientation_hashes.get('0', '')
        
        # 3. OCR text extraction
        if self.enable_ocr and self.ocr_reader:
            ocr_data = self._extract_ocr(img)
            features.ocr_regions = ocr_data.get('regions', [])
            features.full_text = ocr_data.get('full_text', '')
            features.text_hash = ocr_data.get('text_hash', '')
        
        return features
    
    def _compute_perceptual_hash(self, img: Image.Image, hash_size: int = 16) -> str:
        """Compute perceptual hash of image."""
        try:
            # Resize to hash_size x hash_size
            img_small = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img_small)
            
            # Compute difference hash (dHash)
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Convert to hex string
            hash_int = sum([2**i for i, v in enumerate(diff.flatten()) if v])
            return format(hash_int, f'0{hash_size * hash_size // 4}x')
        except Exception as e:
            logger.debug(f"Failed to compute phash: {e}")
            return ""
    
    def _compute_orientation_hashes(self, img: Image.Image) -> Dict[str, str]:
        """
        Compute perceptual hashes for all orientations.
        
        Returns dict with keys: '0', '90', '180', '270', 'flip_h', 'flip_v'
        """
        hashes = {}
        
        try:
            # Original
            hashes['0'] = self._compute_perceptual_hash(img)
            
            # Rotations
            for angle in [90, 180, 270]:
                rotated = img.rotate(angle, expand=True)
                hashes[str(angle)] = self._compute_perceptual_hash(rotated)
            
            # Flips
            hashes['flip_h'] = self._compute_perceptual_hash(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
            hashes['flip_v'] = self._compute_perceptual_hash(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
            
        except Exception as e:
            logger.debug(f"Failed to compute orientation hashes: {e}")
        
        return hashes
    
    def _extract_ocr(self, img: Image.Image) -> Dict[str, Any]:
        """Extract OCR text from image."""
        result = {
            'regions': [],
            'full_text': '',
            'text_hash': ''
        }
        
        if not self.ocr_reader:
            return result
        
        try:
            # Convert to numpy for EasyOCR
            img_array = np.array(img)
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(img_array)
            
            texts = []
            for bbox, text, confidence in ocr_results:
                if confidence > 0.3:  # Filter low confidence
                    region = {
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': [list(map(float, point)) for point in bbox]
                    }
                    result['regions'].append(region)
                    texts.append(text)
            
            # Full text
            result['full_text'] = ' '.join(texts)
            
            # Text hash for quick comparison
            if result['full_text']:
                result['text_hash'] = hashlib.md5(
                    result['full_text'].lower().encode()
                ).hexdigest()
            
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
        
        return result
    
    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> List[Optional[np.ndarray]]:
        """
        Extract CNN features from multiple images.
        
        Args:
            images: List of image paths or PIL Images
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of feature vectors (None for failed images)
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img in batch:
                embedding = self.extract_features(img, normalize=normalize)
                results.append(embedding)
        
        return results
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize if needed
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        return float(np.dot(embedding1, embedding2))


# Alias for backward compatibility
MultiMethodFeatureExtractor = ImageFeatureExtractor


# Convenience function
def get_image_extractor(model_name: str = "resnet") -> ImageFeatureExtractor:
    """Get an image feature extractor instance."""
    return ImageFeatureExtractor(model_name=model_name)


if __name__ == "__main__":
    # Test
    print("Testing ImageFeatureExtractor...")
    extractor = ImageFeatureExtractor(model_name="resnet", enable_ocr=False)
    print(f"Model: {extractor.model_name}")
    print(f"Embedding dim: {extractor.embedding_dim}")
    print(f"Device: {extractor.device}")
    print("✓ ImageFeatureExtractor initialized successfully")
