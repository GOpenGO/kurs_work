from abc import ABC, abstractmethod

class BaseQueryStrategy(ABC):
    def __init__(self, config, **kwargs):
        self.config = config    
        self.strategy_name = self.config.get('active_learning_strategy', 'BaseStrategy')
        print(f"BaseQueryStrategy initialized for: {self.strategy_name}")

    @abstractmethod
    def query(self, model, unlabeled_dataloader, n_instances_to_query, device):
        """
        Выбирает экземпляры из неразмеченного пула.

        Args:
            model: Обученная модель PyTorch.
            unlabeled_dataloader: DataLoader для неразмеченных данных.
            n_instances_to_query: Количество экземпляров для выбора.
            device: Устройство (cpu/cuda), на котором выполняются вычисления.

        Returns:
            queried_indices: Список оригинальных индексов выбранных экземпляров.
        """
        pass