import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np


default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GSVCitiesDataset(Dataset):
    """Dataset for GSV Cities."""

    def __init__(self,
                 path=Path("./data/gsv_xs"),
                 img_per_place=4,
                 min_img_per_place=4,
                 transform=default_transform,
                 generated_data_prob=None,
                 sample_size=None,
                 multi_model=False):
        super().__init__()
        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than or equal to {min_img_per_place}"
        
        self.path = path
        self.sample_size = sample_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.transform = transform
        self.generated_data_prob = generated_data_prob
        self.multi_model = multi_model
        self.df = self._get_dataframe()
        self.unique_ids = self.df['ID'].unique().tolist()
        self.id_to_images = self._map_id_to_images()
        
    def __getitem__(self, index):
        
        id = self.unique_ids[index]

        
        selected_images = np.random.choice(self.id_to_images[id], self.img_per_place, replace=False)

        
        if self.multi_model:
            img_list = []
            img_depth_list = []
            for img_path in selected_images:
                img, img_depth = self._load_and_transform_image(img_path)
                img_list.append(img)
                img_depth_list.append(img_depth)
        else:
            img_list = [self._load_and_transform_image(img_path) for img_path in selected_images]

        
        if self.multi_model:
            imgs = torch.stack(img_list)
            img_depths = torch.stack(img_depth_list)
            return imgs, img_depths, torch.tensor(index).repeat(self.img_per_place)
        else:
            imgs = torch.stack(img_list)
            return imgs, torch.tensor(index).repeat(self.img_per_place)

    def __len__(self):
        return len(self.unique_ids)

    def _load_and_transform_image(self, img_path):
        if self.generated_data_prob and torch.rand(1).item() < self.generated_data_prob:
            img_path = img_path.replace('.jpg', '_augmented.jpg')

        img = Image.open(img_path).convert('RGB')
        
        if self.multi_model:
            img_depth = np.load(img_path.replace('.jpg', '_depth.npy'))
            img_depth = Image.fromarray(img_depth).resize((224, 224))
            img_depth = np.stack([img_depth] * 3, axis=2)
            
            
        if self.transform:
            img = self.transform(img)
            if self.multi_model:
                img_depth = self.transform(img_depth)
                return img, img_depth

        return img

    def _get_dataframe(self):
        train_dir = os.path.join(self.path, "train")
        data = []

        for city in os.listdir(train_dir):
            city_path = os.path.join(train_dir, city)
            if os.path.isdir(city_path):
                data.extend([
                    [image_name.split('@')[-2], os.path.join(city_path, image_name)]
                    for image_name in os.listdir(city_path)
                    if image_name.endswith('.jpg') and 'augmented' not in image_name
                ])

        df = pd.DataFrame(data, columns=['ID', 'ImageName'])
        if self.sample_size:
            df, _ = train_test_split(df, train_size=self.sample_size, stratify=df['ID'])  

        df = df.groupby('ID').filter(lambda x: len(x) >= self.min_img_per_place).reset_index(drop=True)

        return df

    def _map_id_to_images(self):
        id_to_images = defaultdict(list)
        grouped = self.df.groupby('ID')['ImageName'].apply(list).items()
        for unique_id, images in grouped:
            id_to_images[unique_id] = images
        return id_to_images


