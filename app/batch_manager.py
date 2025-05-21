# mypy: ignore-errors
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import glob
from loguru import logger

class BatchJob:
    """Class representing a batch job with status and results"""
    
    def __init__(self, batch_id: str, status: str = "unknown", created_at: str = None, model: str = None):
        self.id = batch_id
        self.status = status
        self.created_at = created_at or datetime.now().isoformat()
        self.model = model
        self.results = []
        self.errors = []
        self.request_count = 0
        self.last_updated = datetime.now().isoformat()
        self.metadata = {}
    
    def update_status(self, status: str) -> None:
        """Update batch job status"""
        self.status = status
        self.last_updated = datetime.now().isoformat()
    
    def add_results(self, results: List[Dict]) -> None:
        """Add results to the batch job"""
        self.results = results
        self.last_updated = datetime.now().isoformat()
    
    def add_errors(self, errors: List[Dict]) -> None:
        """Add errors to the batch job"""
        self.errors = errors
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "model": self.model,
            "results": self.results,
            "errors": self.errors,
            "request_count": self.request_count,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchJob':
        """Create a BatchJob instance from dictionary"""
        batch_job = cls(
            batch_id=data.get("id"),
            status=data.get("status", "unknown"),
            created_at=data.get("created_at"),
            model=data.get("model")
        )
        batch_job.results = data.get("results", [])
        batch_job.errors = data.get("errors", [])
        batch_job.request_count = data.get("request_count", 0)
        batch_job.last_updated = data.get("last_updated", batch_job.created_at)
        batch_job.metadata = data.get("metadata", {})
        return batch_job

class BatchManager:
    """Manager for batch jobs with persistence and tracking"""

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "batches")
        self.storage_dir = storage_dir
        self.batches = {}  # In-memory cache of batch jobs
        self.ensure_storage_dir()
        self.scan_batch_folder()  # Load existing batches on initialization
    
    def ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created batch storage directory: {self.storage_dir}")
    

    def add_batch(self, batch_id: str, batch_info: Dict) -> None:
        """Add a new batch job or update an existing one.

        ``batch_info`` can be either a dictionary or a :class:`BatchJob` instance.
        """
        if isinstance(batch_info, dict):

    def add_batch(self, batch_id: str, batch_info: Union[Dict, BatchJob]) -> None:
        """Add a new batch job or update an existing one.

        `batch_info` may be a dictionary of batch data or an existing
        :class:`BatchJob` object.
        """
        if isinstance(batch_info, BatchJob):
            # Use the provided BatchJob instance directly
            batch_job = batch_info
            # Ensure the ID matches the key used in the manager
            if batch_job.id != batch_id:
                batch_job.id = batch_id
        elif isinstance(batch_info, dict):

            # Create BatchJob from dictionary
            if "id" not in batch_info:
                batch_info["id"] = batch_id
            batch_job = BatchJob.from_dict(batch_info)
        elif isinstance(batch_info, BatchJob):
            # Use the provided BatchJob instance directly
            batch_job = batch_info
            batch_job.id = batch_id
        else:

            # Create new BatchJob from an object with similar attributes

            # Fallback: treat as a simple object with attributes

            batch_job = BatchJob(
                batch_id=batch_id,
                status=getattr(batch_info, "status", "unknown"),
                created_at=getattr(batch_info, "created_at", None),
                model=getattr(batch_info, "model", None)
            )
        
        self.batches[batch_id] = batch_job
        self.save_batch(batch_id)
        logger.info(f"Added batch job {batch_id} to manager")
    
    def get_batch(self, batch_id: str) -> Optional[Dict]:
        """Get a batch job by ID"""
        # First check in-memory cache
        if batch_id in self.batches:
            return self.batches[batch_id].to_dict()
        
        # If not in memory, try to load from disk
        batch_file = os.path.join(self.storage_dir, f"{batch_id}.json")
        if os.path.exists(batch_file):
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    batch_job = BatchJob.from_dict(batch_data)
                    self.batches[batch_id] = batch_job
                    return batch_job.to_dict()
            except Exception as e:
                logger.error(f"Error loading batch {batch_id}: {str(e)}")
        
        return None
    
    def update_batch(self, batch_id: str, status: str = None, 
                     results: List[Dict] = None, errors: List[Dict] = None) -> None:
        """Update batch job with new status, results, or errors"""
        batch_job = self.get_batch_object(batch_id)
        if not batch_job:
            logger.error(f"Cannot update batch {batch_id}: not found")
            return
        
        if status:
            batch_job.update_status(status)
        
        if results is not None:
            batch_job.add_results(results)
        
        if errors is not None:
            batch_job.add_errors(errors)
        
        self.save_batch(batch_id)
        logger.info(f"Updated batch {batch_id} with status={status}")
    
    def get_batch_object(self, batch_id: str) -> Optional[BatchJob]:
        """Get the BatchJob object for a batch ID"""
        if batch_id in self.batches:
            return self.batches[batch_id]
        
        batch_data = self.get_batch(batch_id)
        if batch_data:
            return BatchJob.from_dict(batch_data)
        
        return None
    
    def save_batch(self, batch_id: str) -> None:
        """Save a batch job to disk"""
        if batch_id not in self.batches:
            logger.error(f"Cannot save batch {batch_id}: not found in memory")
            return
        
        batch_file = os.path.join(self.storage_dir, f"{batch_id}.json")
        try:
            with open(batch_file, 'w') as f:
                json.dump(self.batches[batch_id].to_dict(), f, indent=2)
            logger.debug(f"Saved batch {batch_id} to {batch_file}")
        except Exception as e:
            logger.error(f"Error saving batch {batch_id}: {str(e)}")
    
    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch job"""
        batch_file = os.path.join(self.storage_dir, f"{batch_id}.json")
        if os.path.exists(batch_file):
            try:
                os.remove(batch_file)
                if batch_id in self.batches:
                    del self.batches[batch_id]
                logger.info(f"Deleted batch {batch_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting batch {batch_id}: {str(e)}")
        return False
    
    def get_all_batch_ids(self) -> List[str]:
        """Get all batch job IDs"""
        # Combine in-memory batches with any on disk
        all_ids = set(self.batches.keys())
        
        # Find all JSON files in the storage directory
        batch_files = glob.glob(os.path.join(self.storage_dir, "*.json"))
        for file_path in batch_files:
            batch_id = os.path.basename(file_path).replace(".json", "")
            all_ids.add(batch_id)
        
        return sorted(list(all_ids))
    
    def scan_batch_folder(self) -> None:
        """
        Scan the batch folder and load all batch jobs.
        Also scan the batches directory to auto-detect batch input files.
        """
        # First, scan for batch JSON files in the storage directory
        batch_files = glob.glob(os.path.join(self.storage_dir, "*.json"))
        logger.info(f"Scanning batch folder, found {len(batch_files)} batch JSON files")
        
        for file_path in batch_files:
            batch_id = os.path.basename(file_path).replace(".json", "")
            try:
                with open(file_path, 'r') as f:
                    batch_data = json.load(f)
                    batch_job = BatchJob.from_dict(batch_data)
                    self.batches[batch_id] = batch_job
            except Exception as e:
                logger.error(f"Error loading batch file {file_path}: {str(e)}")
        
        # Next, scan for batch input JSONL files that might not be loaded yet
        # Check if batches directory exists
        batches_dir = os.path.join(os.path.dirname(__file__), "batches")
        if os.path.exists(batches_dir) and os.path.isdir(batches_dir):
            # Look for files that match the batch ID format (batch_*)
            batch_input_files = glob.glob(os.path.join(batches_dir, "batch_*.jsonl"))
            logger.info(
                f"Found {len(batch_input_files)} batch input files in {batches_dir}"
            )
            
            for input_file in batch_input_files:
                batch_id = os.path.basename(input_file).replace(".jsonl", "")
                
                # Check if this batch is already loaded
                if batch_id in self.batches:
                    continue
                    
                # This is a batch we haven't loaded yet, create a placeholder
                logger.info(f"Auto-detecting batch ID: {batch_id}")
                
                # Try to extract creation time from the filename or use file creation time
                try:
                    # Format is typically batch_timestamp_randomid
                    parts = batch_id.split('_')
                    if len(parts) >= 2 and parts[1].isdigit():
                        created_at = datetime.fromtimestamp(int(parts[1])).isoformat()
                    else:
                        created_at = datetime.fromtimestamp(os.path.getctime(input_file)).isoformat()
                except:
                    created_at = datetime.now().isoformat()
                
                # Create a new batch job with detected info
                batch_job = BatchJob(
                    batch_id=batch_id,
                    status="detected",  # Special status for auto-detected batches
                    created_at=created_at
                )
                
                # Set metadata with file info
                batch_job.metadata = {
                    "input_file": input_file,
                    "auto_detected": True,
                    "file_size": os.path.getsize(input_file)
                }
                
                # Add to our in-memory cache
                self.batches[batch_id] = batch_job
                
                # Save this batch info
                self.save_batch(batch_id)
    
    def prune_old_batches(self, days: int = 30) -> int:
        """Remove batches older than specified days"""
        current_time = time.time()
        seconds_threshold = days * 24 * 60 * 60
        pruned_count = 0
        
        for batch_id in list(self.batches.keys()):
            batch_job = self.batches[batch_id]
            
            # Convert ISO timestamp to epoch time
            try:
                created_time = datetime.fromisoformat(batch_job.created_at).timestamp()
                age_seconds = current_time - created_time
                
                if age_seconds > seconds_threshold:
                    if self.delete_batch(batch_id):
                        pruned_count += 1
            except Exception as e:
                logger.error(f"Error pruning batch {batch_id}: {str(e)}")
        
        logger.info(f"Pruned {pruned_count} old batches")
        return pruned_count
    
    def get_stats(self) -> Dict:
        """Get statistics about managed batches"""
        total_batches = len(self.get_all_batch_ids())
        in_progress = sum(1 for bid in self.get_all_batch_ids() 
                         if self.get_batch(bid) and self.get_batch(bid)["status"] == "in_progress")
        completed = sum(1 for bid in self.get_all_batch_ids() 
                       if self.get_batch(bid) and self.get_batch(bid)["status"] == "completed")
        failed = sum(1 for bid in self.get_all_batch_ids() 
                    if self.get_batch(bid) and self.get_batch(bid)["status"] == "failed")
        
        return {
            "total_batches": total_batches,
            "in_progress": in_progress,
            "completed": completed,
            "failed": failed,
            "other_status": total_batches - in_progress - completed - failed
        }

# Create a singleton instance
batch_manager = BatchManager() 
