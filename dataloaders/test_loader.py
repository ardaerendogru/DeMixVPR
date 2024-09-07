"""
Module for loading test datasets.
"""

import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TestDataset(Dataset):
    """Dataset for testing purposes."""
    
    def __init__(self, path='./data/sf_xs/val', transform=default_transform, multi_model=False):
        self.path = path
        self.transform = transform
        self.multi_model = multi_model
        self.db_path = os.path.join(path, "database")
        self.query_path = os.path.join(path, "queries")
        
        self.query_image_paths = self._get_image_paths(self.query_path)
        self.db_image_paths = self._get_image_paths(self.db_path)
        self.all_image_paths = self.query_image_paths + self.db_image_paths
        
        self.db_utm = self._extract_utm(self.db_image_paths)
        self.query_utm = self._extract_utm(self.query_image_paths)
        
        self.db_size = self._len_db()
        self.query_size = self._len_query()
        
        self.close_indices = self._get_close_indices()
    def __getitem__(self, index):
        image_path = self.all_image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.multi_model:
            image_depth = np.load(image_path.replace('.jpg', '_depth.npy'))
            image_depth = Image.fromarray(image_depth).resize((224, 224))
            image_depth = np.stack([image_depth] * 3, axis=2)
            return self.transform(image), self.transform(image_depth), index
        else:
            return self.transform(image), index
    def get_image(self, index):
        image_path = self.all_image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return image
    
    def __len__(self):
        return len(self.all_image_paths)
    
    def _len_db(self):
        return len(self.db_image_paths)
    
    def _len_query(self):
        return len(self.query_image_paths)
    
    def _get_image_paths(self, directory):
        """Retrieve image paths from a directory."""
        return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.jpg')]

    def _extract_utm(self, image_paths):
        """Extract UTM coordinates from image filenames."""
        return [(float(image.split('@')[1]), float(image.split('@')[2])) for image in image_paths]

    def _calculate_distances(self):
        """Calculate pairwise distances between database and query UTM coordinates."""
        db_utm_array = np.array(self.db_utm, dtype=float)
        query_utm_array = np.array(self.query_utm, dtype=float)
        return cdist(query_utm_array, db_utm_array, metric='euclidean')

    def _get_close_indices(self, threshold=25):
        """
        For each query index, find database indices within the threshold distance.
        
        Args:
            threshold (float): Maximum distance to consider (default: 25)
        
        Returns:
            dict: A dictionary where keys are query indices and values are lists of close database indices
        """
        distances = self._calculate_distances()
        close_indices = {}
        for query_idx, distances in enumerate(distances):
            close_mask = distances <= threshold
            close_db_indices = np.where(close_mask)[0].tolist()
            close_indices[query_idx] = close_db_indices
        
        return close_indices