from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"BaseModel initialized with config: {self.config.get('model_name', 'Unknown')}")

    @abstractmethod
    def forward(self, x):
        pass

    # Можно добавить общие методы, например, для сохранения/загрузки весов,
    # или для переключения режима train/eval, если они одинаковы для всех моделей.
    def get_device(self):
        # Простой способ определить устройство, на котором находится модель
        return next(self.parameters()).device