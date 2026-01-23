"""
Wasabi S3 Storage Client.
Handles image storage and retrieval from Wasabi S3-compatible storage.
"""

import os
from typing import Optional, List, Dict, BinaryIO
from io import BytesIO
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class WasabiClient:
    """Client for Wasabi S3-compatible storage"""
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize Wasabi client.
        
        Args:
            access_key: Wasabi access key (or from env WASABI_ACCESS_KEY)
            secret_key: Wasabi secret key (or from env WASABI_SECRET_KEY)
            bucket: Bucket name (or from env WASABI_BUCKET)
            region: Wasabi region (or from env WASABI_REGION)
            endpoint_url: Wasabi endpoint URL (auto-generated from region)
        """
        self.access_key = access_key or os.environ.get('WASABI_ACCESS_KEY')
        self.secret_key = secret_key or os.environ.get('WASABI_SECRET_KEY')
        self.bucket = bucket or os.environ.get('WASABI_BUCKET', 'uyar-plagiarism')
        self.region = region or os.environ.get('WASABI_REGION', 'us-east-1')
        
        # Build endpoint URL from region if not provided
        if endpoint_url:
            self.endpoint_url = endpoint_url
        else:
            # Wasabi endpoint - us-east-1 can use either format
            # https://docs.wasabi.com/docs/what-are-the-service-urls-for-wasabi-s3-storage
            if self.region == 'us-east-1':
                # Try without region first (original Wasabi format)
                self.endpoint_url = "https://s3.wasabisys.com"
            else:
                self.endpoint_url = f"https://s3.{self.region}.wasabisys.com"
        
        self._client = None
        self._resource = None
    
    def get_endpoint_info(self) -> str:
        """Get connection info for debugging"""
        return (f"Endpoint: {self.endpoint_url}\n"
                f"Bucket: {self.bucket}\n"
                f"Region: {self.region}\n"
                f"Access Key: {self.access_key[:8]}..." if self.access_key else "Access Key: None")
    
    @property
    def client(self):
        """Lazy load boto3 client"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def resource(self):
        """Lazy load boto3 resource"""
        if self._resource is None:
            self._resource = self._create_resource()
        return self._resource
    
    def _create_client(self):
        """Create boto3 S3 client"""
        try:
            import boto3
            
            return boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
        except ImportError:
            print("Warning: boto3 not installed")
            return None
    
    def _create_resource(self):
        """Create boto3 S3 resource"""
        try:
            import boto3
            
            return boto3.resource(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
        except ImportError:
            return None
    
    # =========================================================================
    # PATH UTILITIES
    # =========================================================================
    
    def build_figure_path(
        self,
        server: str,
        year: str,
        month: str,
        day: str,
        manuscript_id: str,
        figure_name: str
    ) -> str:
        """
        Build S3 path for a figure.
        
        Args:
            server: biorxiv or medrxiv
            year: Publication year
            month: Publication month
            day: Publication day
            manuscript_id: Manuscript ID
            figure_name: Figure filename (e.g., fig_p5_1.png)
        
        Returns:
            S3 key path
        """
        return f"{server}/{year}/{month}/{day}/{manuscript_id}/figures/{figure_name}"
    
    def build_figure_path_from_doi(self, doi: str, server: str, page: int, index: int) -> str:
        """
        Build S3 path from DOI.
        
        Args:
            doi: Paper DOI
            server: Server name
            page: Page number
            index: Figure index
        
        Returns:
            S3 key path
        """
        doi_suffix = doi.replace("10.1101/", "")
        parts = doi_suffix.split(".")
        
        if len(parts) >= 4 and parts[0].isdigit():
            year, month, day, ms_id = parts[0], parts[1], parts[2], parts[3]
            return self.build_figure_path(
                server, year, month, day, ms_id, f"fig_p{page}_{index}.png"
            )
        else:
            return f"{server}/legacy/{doi_suffix}/figures/fig_p{page}_{index}.png"
    
    def get_s3_url(self, key: str) -> str:
        """Get full S3 URL for a key"""
        return f"s3://{self.bucket}/{key}"
    
    def get_https_url(self, key: str) -> str:
        """Get HTTPS URL for direct access"""
        return f"https://s3.wasabisys.com/{self.bucket}/{key}"
    
    # =========================================================================
    # UPLOAD OPERATIONS
    # =========================================================================
    
    def upload_image(
        self,
        image: Image.Image,
        key: str,
        format: str = 'PNG',
        quality: int = 95
    ) -> bool:
        """
        Upload PIL Image to Wasabi.
        
        Args:
            image: PIL Image object
            key: S3 key (path)
            format: Image format (PNG, JPEG)
            quality: JPEG quality (if applicable)
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            buffer = BytesIO()
            
            if format.upper() == 'JPEG':
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(buffer, format='JPEG', quality=quality)
            else:
                image.save(buffer, format='PNG')
            
            buffer.seek(0)
            
            content_type = 'image/jpeg' if format.upper() == 'JPEG' else 'image/png'
            
            self.client.upload_fileobj(
                buffer,
                self.bucket,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            
            return True
        
        except Exception as e:
            print(f"Error uploading image: {e}")
            return False
    
    def upload_file(self, file_path: str, key: str) -> bool:
        """
        Upload file from local path.
        
        Args:
            file_path: Local file path
            key: S3 key (path)
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            self.client.upload_file(file_path, self.bucket, key)
            return True
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False
    
    def upload_bytes(self, data: bytes, key: str, content_type: str = None) -> bool:
        """
        Upload raw bytes.
        
        Args:
            data: Byte data
            key: S3 key
            content_type: MIME type
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                **extra_args
            )
            return True
        
        except Exception as e:
            print(f"Error uploading bytes: {e}")
            return False
    
    # =========================================================================
    # DOWNLOAD OPERATIONS
    # =========================================================================
    
    def download_image(self, key: str) -> Optional[Image.Image]:
        """
        Download image from Wasabi.
        
        Args:
            key: S3 key (path)
        
        Returns:
            PIL Image or None
        """
        if self.client is None:
            return None
        
        try:
            buffer = BytesIO()
            self.client.download_fileobj(self.bucket, key, buffer)
            buffer.seek(0)
            return Image.open(buffer)
        
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def download_file(self, key: str, local_path: str) -> bool:
        """
        Download file to local path.
        
        Args:
            key: S3 key
            local_path: Local file path
        
        Returns:
            True if successful
        """
        if self.client is None:
            return False
        
        try:
            self.client.download_file(self.bucket, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    
    def download_bytes(self, key: str) -> Optional[bytes]:
        """
        Download raw bytes.
        
        Args:
            key: S3 key
        
        Returns:
            Byte data or None
        """
        if self.client is None:
            return None
        
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            print(f"Error downloading bytes: {e}")
            return None
    
    # =========================================================================
    # MANAGEMENT OPERATIONS
    # =========================================================================
    
    def exists(self, key: str) -> bool:
        """Check if object exists"""
        if self.client is None:
            return False
        
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete object"""
        if self.client is None:
            return False
        
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            print(f"Error deleting object: {e}")
            return False
    
    def list_objects(self, prefix: str, max_keys: int = 1000) -> List[Dict]:
        """
        List objects with prefix.
        
        Args:
            prefix: Key prefix to filter
            max_keys: Maximum objects to return
        
        Returns:
            List of object metadata dicts
        """
        if self.client is None:
            return []
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
            
            return objects
        
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []
    
    def list_paper_figures(self, server: str, doi: str) -> List[str]:
        """
        List all figures for a paper.
        
        Args:
            server: Server name
            doi: Paper DOI
        
        Returns:
            List of figure keys
        """
        doi_suffix = doi.replace("10.1101/", "")
        parts = doi_suffix.split(".")
        
        if len(parts) >= 4 and parts[0].isdigit():
            prefix = f"{server}/{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}/figures/"
        else:
            prefix = f"{server}/legacy/{doi_suffix}/figures/"
        
        objects = self.list_objects(prefix)
        return [obj['key'] for obj in objects]
    
    def get_storage_stats(self, prefix: str = "") -> Dict:
        """
        Get storage statistics.
        
        Args:
            prefix: Optional prefix to filter
        
        Returns:
            Statistics dict
        """
        if self.client is None:
            return {}
        
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            
            total_size = 0
            total_count = 0
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    total_size += obj['Size']
                    total_count += 1
            
            return {
                'total_objects': total_count,
                'total_size_bytes': total_size,
                'total_size_gb': total_size / (1024**3),
                'prefix': prefix
            }
        
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def upload_paper_figures(
        self,
        figures: List[Dict],
        server: str,
        doi: str
    ) -> Dict[str, str]:
        """
        Upload all figures for a paper.
        
        Args:
            figures: List of dicts with 'image' (PIL), 'page', 'index'
            server: Server name
            doi: Paper DOI
        
        Returns:
            Dict mapping figure_id to S3 URL
        """
        uploaded = {}
        
        for fig in figures:
            key = self.build_figure_path_from_doi(
                doi, server, fig['page'], fig['index']
            )
            
            if self.upload_image(fig['image'], key):
                figure_id = f"fig_p{fig['page']}_{fig['index']}"
                uploaded[figure_id] = self.get_s3_url(key)
        
        return uploaded
    
    def delete_paper_figures(self, server: str, doi: str) -> int:
        """
        Delete all figures for a paper.
        
        Args:
            server: Server name
            doi: Paper DOI
        
        Returns:
            Number of deleted objects
        """
        keys = self.list_paper_figures(server, doi)
        deleted = 0
        
        for key in keys:
            if self.delete(key):
                deleted += 1
        
        return deleted


class MockWasabiClient(WasabiClient):
    """Mock client for testing without Wasabi"""
    
    def __init__(self, **kwargs):
        self.storage = {}
        self._client = self
        self._resource = self
    
    def upload_image(self, image: Image.Image, key: str, **kwargs) -> bool:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        self.storage[key] = buffer.getvalue()
        return True
    
    def download_image(self, key: str) -> Optional[Image.Image]:
        if key in self.storage:
            buffer = BytesIO(self.storage[key])
            return Image.open(buffer)
        return None
    
    def exists(self, key: str) -> bool:
        return key in self.storage
    
    def delete(self, key: str) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False
    
    def list_objects(self, prefix: str, max_keys: int = 1000) -> List[Dict]:
        return [
            {'key': k, 'size': len(v), 'last_modified': '2024-01-01'}
            for k, v in self.storage.items()
            if k.startswith(prefix)
        ][:max_keys]


if __name__ == "__main__":
    # Test with mock client
    client = MockWasabiClient()
    
    # Create test image
    test_image = Image.new('RGB', (256, 256), color='blue')
    
    # Upload
    key = "biorxiv/2024/01/15/575685/figures/fig_p1_1.png"
    success = client.upload_image(test_image, key)
    print(f"Upload successful: {success}")
    
    # Check exists
    print(f"Exists: {client.exists(key)}")
    
    # Download
    downloaded = client.download_image(key)
    print(f"Downloaded: {downloaded is not None}")
    
    # List
    objects = client.list_objects("biorxiv/2024/")
    print(f"Listed objects: {len(objects)}")
    
    # Get URLs
    print(f"S3 URL: {client.get_s3_url(key)}")
    print(f"HTTPS URL: {client.get_https_url(key)}")
