"""
Embed ALL extracted content into Qdrant
=======================================

Works with extracted JSONs from Wasabi:
  extracted/biorxiv/April_2019/xxx.json
  extracted/medrxiv/June_2019/yyy.json

Features:
- Checks if already embedded (skip existing in Qdrant)
- Batch processing for efficiency
- Progress bar
- GPU acceleration if available

Usage:
    python embed_all.py --workers 10 --batch-size 100
    python embed_all.py --workers 10 --max 1000
    python embed_all.py --text-only  # Skip image embeddings
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

import boto3
from botocore.config import Config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    json_key: str
    success: bool
    doi: Optional[str] = None
    text_chunks: int = 0
    image_embeddings: int = 0
    error: Optional[str] = None


class BulkEmbedder:
    """Embed all extracted content into Qdrant"""
    
    def __init__(self, max_workers: int = 10, batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Wasabi client
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi()
        
        # Qdrant client
        self.qdrant = None
        self._init_qdrant()
        
        # Embedding model (lazy load)
        self.text_model = None
        self.image_model = None
        
        if not self.wasabi:
            logger.error("Wasabi not configured!")
            sys.exit(1)
    
    def _init_wasabi(self):
        """Initialize Wasabi client"""
        try:
            from storage.wasabi_client import WasabiClient
            wc = WasabiClient()
            
            if wc.access_key and wc.secret_key:
                endpoint = getattr(wc, 'endpoint', None) or getattr(wc, 'endpoint_url', None) or f"https://s3.{wc.region}.wasabisys.com"
                region = getattr(wc, 'region', 'ap-southeast-1')
                
                config = Config(max_pool_connections=50, retries={'max_attempts': 3})
                
                self.wasabi = boto3.client(
                    's3',
                    endpoint_url=endpoint,
                    aws_access_key_id=wc.access_key,
                    aws_secret_access_key=wc.secret_key,
                    region_name=region,
                    config=config
                )
                self.wasabi_bucket = wc.bucket
                logger.info(f"✓ Wasabi connected ({wc.bucket})")
        except Exception as e:
            logger.error(f"Wasabi init failed: {e}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client"""
        try:
            from storage.qdrant_client import QdrantManager
            self.qdrant = QdrantManager()
            logger.info(f"✓ Qdrant connected")
        except Exception as e:
            logger.warning(f"Qdrant init failed: {e}")
    
    def _load_text_model(self):
        """Load text embedding model"""
        if self.text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use BGE or PubMedBERT
                model_name = os.environ.get('TEXT_EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')
                self.text_model = SentenceTransformer(model_name)
                logger.info(f"✓ Text model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
        return self.text_model
    
    def list_extracted_folders(self) -> List[tuple]:
        """List all server/month folders with extracted JSONs"""
        folders = []
        paginator = self.wasabi.get_paginator('list_objects_v2')
        
        for server in ['biorxiv', 'medrxiv']:
            prefix = f"extracted/{server}/"
            
            try:
                for page in paginator.paginate(
                    Bucket=self.wasabi_bucket,
                    Prefix=prefix,
                    Delimiter='/'
                ):
                    for p in page.get('CommonPrefixes', []):
                        folder = p['Prefix'].replace(prefix, '').rstrip('/')
                        folders.append((server, folder))
            except Exception as e:
                logger.warning(f"Error listing {server}: {e}")
        
        return folders
    
    def list_jsons_in_folder(self, server: str, month: str, max_files: int = None) -> List[dict]:
        """List extracted JSONs in a folder"""
        files = []
        prefix = f"extracted/{server}/{month}/"
        paginator = self.wasabi.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.wasabi_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.json'):
                    files.append({
                        'key': key,
                        'server': server,
                        'month': month,
                        'filename': key.split('/')[-1]
                    })
                    if max_files and len(files) >= max_files:
                        return files
        
        return files
    
    def check_already_embedded(self, doi: str) -> bool:
        """Check if DOI is already embedded in Qdrant"""
        if not self.qdrant or not doi:
            return False
        
        try:
            # Search for this DOI in Qdrant
            results = self.qdrant.client.scroll(
                collection_name="text_embeddings",
                scroll_filter={
                    "must": [{"key": "doi", "match": {"value": doi}}]
                },
                limit=1
            )
            return len(results[0]) > 0
        except:
            return False
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def embed_single(self, json_info: dict, text_only: bool = False) -> EmbedResult:
        """Embed content from a single extracted JSON"""
        json_key = json_info['key']
        
        try:
            # Download JSON
            response = self.wasabi.get_object(Bucket=self.wasabi_bucket, Key=json_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            doi = data.get('doi')
            
            # Check if already embedded
            if doi and self.check_already_embedded(doi):
                return EmbedResult(json_key=json_key, success=True, doi=doi, error="skipped")
            
            # Get text model
            model = self._load_text_model()
            if not model:
                return EmbedResult(json_key=json_key, success=False, error="No model")
            
            text = data.get('text', '')
            chunks = self.chunk_text(text)
            
            if not chunks:
                return EmbedResult(json_key=json_key, success=False, error="No text")
            
            # Generate embeddings for all chunks
            embeddings = model.encode(chunks, show_progress_bar=False)
            
            # Store in Qdrant
            text_chunks_stored = 0
            if self.qdrant:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    try:
                        self.qdrant.client.upsert(
                            collection_name="text_embeddings",
                            points=[{
                                "id": hash(f"{doi}_{i}") % (2**63),
                                "vector": embedding.tolist(),
                                "payload": {
                                    "doi": doi,
                                    "server": data.get('server'),
                                    "month": data.get('month'),
                                    "chunk_index": i,
                                    "text": chunk[:500]  # Store first 500 chars
                                }
                            }]
                        )
                        text_chunks_stored += 1
                    except Exception as e:
                        logger.debug(f"Chunk upsert error: {e}")
            
            # TODO: Image embeddings (if not text_only)
            image_embeddings = 0
            
            return EmbedResult(
                json_key=json_key,
                success=True,
                doi=doi,
                text_chunks=text_chunks_stored,
                image_embeddings=image_embeddings
            )
            
        except Exception as e:
            return EmbedResult(json_key=json_key, success=False, error=str(e)[:50])
    
    def embed_folder(self, server: str, month: str, max_files: int = None, text_only: bool = False) -> dict:
        """Embed all JSONs in a folder"""
        stats = {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0, 'chunks': 0}
        
        files = self.list_jsons_in_folder(server, month, max_files)
        stats['total'] = len(files)
        
        if not files:
            return stats
        
        # Process sequentially for now (model is not thread-safe)
        if HAS_TQDM:
            pbar = tqdm(files, desc=f"    {month[:15]}", unit="doc", ncols=100)
        else:
            pbar = files
        
        for f in pbar:
            result = self.embed_single(f, text_only)
            
            if result.success:
                if result.error == "skipped":
                    stats['skipped'] += 1
                else:
                    stats['success'] += 1
                    stats['chunks'] += result.text_chunks
            else:
                stats['failed'] += 1
            
            if HAS_TQDM:
                pbar.set_postfix({
                    '✓': stats['success'],
                    '⏭': stats['skipped'],
                    '✗': stats['failed']
                })
        
        return stats
    
    def embed_all(self, max_per_folder: int = None, server_filter: str = None, 
                  month_filter: str = None, text_only: bool = False):
        """Embed from all folders"""
        
        print("="*70)
        print("EMBED ALL EXTRACTED CONTENT INTO QDRANT")
        print("="*70)
        print(f"Source: Wasabi s3://{self.wasabi_bucket}/extracted/")
        print(f"Destination: Qdrant")
        print(f"Text only: {text_only}")
        print("="*70)
        print()
        
        # Get all folders
        folders = self.list_extracted_folders()
        
        # Apply filters
        if server_filter:
            folders = [(s, m) for s, m in folders if s == server_filter]
        if month_filter:
            folders = [(s, m) for s, m in folders if m == month_filter]
        
        logger.info(f"Found {len(folders)} folders to process")
        print()
        
        total_stats = {'folders': 0, 'total': 0, 'success': 0, 'skipped': 0, 'failed': 0, 'chunks': 0}
        start_time = datetime.now()
        
        for i, (server, month) in enumerate(folders, 1):
            display = f"{server}/{month}"
            print(f"[{i:3d}/{len(folders)}] {display:<45}", end="", flush=True)
            
            stats = self.embed_folder(server, month, max_per_folder, text_only)
            
            total_stats['folders'] += 1
            total_stats['total'] += stats['total']
            total_stats['success'] += stats['success']
            total_stats['skipped'] += stats['skipped']
            total_stats['failed'] += stats['failed']
            total_stats['chunks'] += stats['chunks']
            
            print(f"\r[{i:3d}/{len(folders)}] {display:<45} ✓ {stats['success']:4d} new | {stats['skipped']:4d} skip | {stats['failed']:3d} fail | {stats['chunks']} chunks")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print()
        print("="*70)
        print("EMBEDDING COMPLETE")
        print("="*70)
        print(f"  Folders processed:  {total_stats['folders']}")
        print(f"  Total documents:    {total_stats['total']}")
        print(f"  Newly embedded:     {total_stats['success']}")
        print(f"  Already existed:    {total_stats['skipped']}")
        print(f"  Failed:             {total_stats['failed']}")
        print(f"  Total chunks:       {total_stats['chunks']}")
        print(f"  Duration:           {duration/60:.1f} minutes")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed all extracted content into Qdrant")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Parallel workers (for loading)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding")
    parser.add_argument("--max-per-folder", type=int, help="Max docs per folder")
    parser.add_argument("--server", type=str, choices=['biorxiv', 'medrxiv'], help="Filter by server")
    parser.add_argument("--month", type=str, help="Filter by month")
    parser.add_argument("--text-only", action="store_true", help="Skip image embeddings")
    
    args = parser.parse_args()
    
    embedder = BulkEmbedder(max_workers=args.workers, batch_size=args.batch_size)
    embedder.embed_all(
        max_per_folder=args.max_per_folder,
        server_filter=args.server,
        month_filter=args.month,
        text_only=args.text_only
    )


if __name__ == "__main__":
    main()
