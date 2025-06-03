import torch
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseQueryStrategy


class UncertaintySampling(BaseQueryStrategy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.method = self.config.get('query_params', {}).get('uncertainty_method', 'least_confident')
        print(f"UncertaintySampling strategy initialized with method: {self.method}")

    @torch.no_grad()  # Важно для экономии памяти и ускорения на этапе инференса
    def query(self, model, unlabeled_dataloader, n_instances_to_query, device):
        model.eval()  # Переводим модель в режим оценки

        all_uncertainties = []
        all_original_indices = []

        for batch_data, batch_indices in unlabeled_dataloader:
            batch_data = batch_data.to(device)

            logits = model(batch_data)
            probabilities = F.softmax(logits, dim=1)  # (batch_size, num_classes)

            if self.method == "least_confident":
                # Уверенность - это максимальная вероятность. Неуверенность = 1 - уверенность.
                # Выбираем те, у которых max_prob минимальна (или 1 - max_prob максимальна)
                max_probs, _ = torch.max(probabilities, dim=1)
                uncertainties = 1.0 - max_probs
            elif self.method == "margin":
                # Разница между двумя наибольшими вероятностями
                # Сортируем вероятности для каждого образца
                sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                uncertainties = 1.0 - margin  # Меньший margin = большая неуверенность
            elif self.method == "entropy":
                epsilon = 1e-9
                # probabilities_clipped = torch.clamp(probabilities, epsilon, 1.0 - epsilon) # Заменяем clip на clamp
                probabilities_clipped = torch.clamp(probabilities, min=epsilon)  # Просто чтобы избежать log(0)
                log_probs = torch.log2(probabilities_clipped)
                uncertainties = -torch.sum(probabilities * log_probs, dim=1)  # Большая энтропия = большая неуверенность
            else:
                raise ValueError(f"Unknown uncertainty method: {self.method}")

            all_uncertainties.extend(uncertainties.cpu().tolist())
            all_original_indices.extend(batch_indices.cpu().tolist())

        if not all_original_indices:
            return np.array([], dtype=int)

        all_uncertainties_np = np.array(all_uncertainties)
        all_original_indices_np = np.array(all_original_indices)

        # Сортируем по убыванию неопределенности (для least_confident и margin - это 1-score, для entropy - сам score)
        # Поэтому всегда выбираем те, у кого uncertainty_score больше
        # (для least_confident и margin мы сделали 1.0 - score, поэтому тоже ищем максимум)
        num_available = len(all_uncertainties_np)
        n_to_select = min(n_instances_to_query, num_available)

        # argsort возвращает индексы, которые бы отсортировали массив.
        # Берем последние n_to_select индексов для наибольших значений неопределенности.
        sorted_query_indices_in_list = np.argsort(all_uncertainties_np)[-n_to_select:]

        queried_original_indices = all_original_indices_np[sorted_query_indices_in_list]

        print(f"Queried {len(queried_original_indices)} instances using UncertaintySampling ({self.method}).")
        return queried_original_indices