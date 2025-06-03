import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image 
import os


class MedicalImageDataset(Dataset):
    def __init__(self, image_paths_or_data, labels=None, transform=None, is_unlabeled=False):
        """
        Кастомный датасет для медицинских изображений (PyTorch).

        Args:
            image_paths_or_data: Список путей к изображениям или уже загруженные данные (np.array).
            labels: Метки для изображений (если есть).
            transform: Трансформации PyTorch для применения к изображениям.
            is_unlabeled: Флаг, указывающий, является ли этот датасет неразмеченным.
        """
        self.image_data = image_paths_or_data
        self.labels = labels
        self.transform = transform
        self.is_unlabeled = is_unlabeled

        if isinstance(image_paths_or_data, list) and len(image_paths_or_data) > 0 and isinstance(image_paths_or_data[0],
                                                                                                 str):
            self.load_from_paths = True
        else:
            self.load_from_paths = False  # Данные уже переданы как np.array

        if not is_unlabeled and labels is None:
            raise ValueError("Labels must be provided for a labeled dataset.")
        if not is_unlabeled and len(image_paths_or_data) != len(labels):
            raise ValueError("Number of images and labels must match for a labeled dataset.")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        if self.load_from_paths:
            # Заглушка: в реальном проекте здесь была бы загрузка Image.open(self.image_data[idx])
            # и конвертация в np.array. Для симуляции используем случайные данные.
            img_sample = (np.random.rand(64, 64) * 255).astype(np.uint8)  # Пример HxW
            # print(f"Simulating load from path: {self.image_data[idx]}")
        else:
            img_sample = self.image_data[idx]  # Предполагаем, что это уже np.array (признаки или HxW)

        # Убедимся, что img_sample имеет правильную форму для трансформаций
        # Например, если это 1D признаки, трансформации могут быть не нужны или другими
        # Если это 2D изображение (HxW), ToPILImage ожидает HxW или HxWx1 или HxWx3
        if img_sample.ndim == 1:  # Если это вектор признаков, а не изображение
            if self.transform:
                # print("Warning: Applying image transforms to feature vector. This might be unintended.")
                # Для простоты примера, пропустим трансформацию для векторов признаков
                # или нужно создать отдельные трансформации для них.
                # Здесь мы просто вернем тензор.
                img_tensor = torch.tensor(img_sample, dtype=torch.float32)
            else:
                img_tensor = torch.tensor(img_sample, dtype=torch.float32)

        elif img_sample.ndim == 2:  # Предполагаем (H, W) одноканальное изображение
            if self.transform:
                img_tensor = self.transform(img_sample)
            else:  # Если трансформаций нет, просто конвертируем в тензор
                img_tensor = torch.from_numpy(img_sample.astype(np.float32)).unsqueeze(0)  # Добавляем канал
        else:  # HxWxC или другая размерность
            if self.transform:
                img_tensor = self.transform(img_sample)
            else:
                img_tensor = torch.from_numpy(img_sample.astype(np.float32))

        if self.is_unlabeled:
            return img_tensor, idx  # Возвращаем индекс для отслеживания неразмеченных образцов
        else:
            label = self.labels[idx]
            return img_tensor, torch.tensor(label, dtype=torch.long)  # Метки для классификации


if __name__ == '__main__':
    from transforms import get_medical_image_transforms

    # Пример для размеченного датасета (симуляция с путями)
    dummy_image_paths = [f"data/initial_labeled_images/img_00{i}.png" for i in range(5)]
    dummy_labels = np.random.randint(0, 2, size=5)
    img_transform = get_medical_image_transforms()

    labeled_dataset = MedicalImageDataset(dummy_image_paths, labels=dummy_labels, transform=img_transform)
    print(f"Labeled dataset size: {len(labeled_dataset)}")
    img_tensor, label_tensor = labeled_dataset[0]
    print(f"Sample from labeled dataset: img_shape={img_tensor.shape}, label={label_tensor}")

    # Пример для неразмеченного датасета (симуляция с данными np.array)
    dummy_unlabeled_data_np = np.random.rand(10, 64, 64).astype(np.uint8)  # 10 изображений 64x64
    unlabeled_dataset = MedicalImageDataset(dummy_unlabeled_data_np, transform=img_transform, is_unlabeled=True)
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    img_tensor_u, original_idx = unlabeled_dataset[0]
    print(f"Sample from unlabeled dataset: img_shape={img_tensor_u.shape}, original_idx={original_idx}")

    # Пример с 1D признаками
    dummy_features = np.random.rand(5, 100)  # 5 образцов, 100 признаков
    dummy_feature_labels = np.random.randint(0, 2, 5)
    feature_dataset = MedicalImageDataset(dummy_features, labels=dummy_feature_labels,
                                          transform=None)  # Без трансформаций
    feat_tensor, feat_label = feature_dataset[0]
    print(f"Sample from feature dataset: feat_shape={feat_tensor.shape}, label={feat_label}")