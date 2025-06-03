import torchvision.transforms as transforms
import torch


def get_medical_image_transforms(image_size=(128, 128), augment=False):
    """
    Возвращает набор трансформаций для медицинских изображений.
    """
    transform_list = [
        transforms.ToPILImage(),  # Если входные данные - numpy массивы
        transforms.Resize(image_size),
        # transforms.Grayscale(num_output_channels=1), # Если нужно, но обычно мед. снимки уже 1-канальные
    ]

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Добавить другие аугментации, если нужно
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # Преобразует PIL Image или numpy.ndarray (H x W x C) в torch.FloatTensor (C x H x W) и нормализует к [0.0, 1.0]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для 1-канальных изображений
    ])

    return transforms.Compose(transform_list)


if __name__ == '__main__':
    # Пример использования
    # import numpy as np
    # dummy_image_np = (np.random.rand(256, 256) * 255).astype(np.uint8) # HxW

    # # Преобразуем в HxWx1 для ToPILImage, если оно ожидает 3 канала для цветного
    # # или если ToTensor будет работать с HxW для одноканального
    # # Для простоты, если это уже np.uint8, ToTensor может справиться

    # transform_no_aug = get_medical_image_transforms(augment=False)
    # transform_aug = get_medical_image_transforms(augment=True)

    # # tensor_no_aug = transform_no_aug(dummy_image_np)
    # # tensor_aug = transform_aug(dummy_image_np)

    # # print("Tensor shape (no augmentation):", tensor_no_aug.shape) # Ожидаем [1, 128, 128]
    # # print("Tensor shape (augmentation):", tensor_aug.shape)   # Ожидаем [1, 128, 128]
    print("Transforms defined. Test with dummy data in a notebook or main script.")