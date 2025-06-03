import numpy as np

class ActiveLearner:
    def __init__(self, strategy="uncertainty_sampling"):
        """
        Инициализация активного ученика с выбранной стратегией.
        """
        self.strategy = strategy
        print(f"ActiveLearner initialized with strategy: {self.strategy}")

    def query(self, model_wrapper, X_unlabeled_pool, n_instances=1):
        """
        Выбор экземпляров из неразмеченного пула для запроса метки.
        
        Args:
            model_wrapper: Обертка над моделью, имеющая метод predict_proba().
            X_unlabeled_pool: Пул неразмеченных данных (N_samples, N_features).
            n_instances: Количество экземпляров для выбора.

        Returns:
            query_indices: Индексы выбранных экземпляров в X_unlabeled_pool.
            query_instances: Сами выбранные экземпляры.
        """
        if X_unlabeled_pool.shape[0] == 0:
            print("No unlabeled instances left to query.")
            return np.array([], dtype=int), np.array([])
        
        if n_instances > X_unlabeled_pool.shape[0]:
            n_instances = X_unlabeled_pool.shape[0]
            print(f"Warning: Requested {n_instances} but only {X_unlabeled_pool.shape[0]} unlabeled samples available. Querying all.")


        if self.strategy == "uncertainty_sampling":
            # Получаем вероятности от модели
            probas = model_wrapper.predict_proba(X_unlabeled_pool)
            
            # Стратегия "наименьшей уверенности" (ближайшая к 0.5 для бинарной)
            # Для многоклассовой можно использовать entropies = -np.sum(probas * np.log2(probas + 1e-9), axis=1)
            # и выбирать np.argsort(entropies)[-n_instances:] (максимальная энтропия)
            
            if probas.shape[1] == 2: # Бинарная классификация
                # Выбираем те, где P(class_1) ближе всего к 0.5
                uncertainty_scores = np.abs(probas[:, 1] - 0.5)
                # argsort сортирует по возрастанию, нам нужны наименьшие значения (ближе к 0.5)
                sorted_indices = np.argsort(uncertainty_scores) 
            elif probas.shape[1] > 2: # Многоклассовая
                # Используем энтропию как меру неопределенности
                # Добавляем epsilon для предотвращения log(0)
                epsilon = 1e-9
                probas_clipped = np.clip(probas, epsilon, 1 - epsilon)
                entropy_scores = -np.sum(probas_clipped * np.log2(probas_clipped), axis=1)
                # argsort сортирует по возрастанию, нам нужны наибольшие значения энтропии
                sorted_indices = np.argsort(entropy_scores)[::-1] # переворачиваем для убывания
            else: # Один класс? Не должно быть, но на всякий случай
                print("Warning: Probabilities shape suggests only one class. Selecting randomly.")
                query_indices = np.random.choice(X_unlabeled_pool.shape[0], size=n_instances, replace=False)
                return query_indices, X_unlabeled_pool[query_indices]

            query_indices = sorted_indices[:n_instances]

        elif self.strategy == "random_sampling":
            query_indices = np.random.choice(X_unlabeled_pool.shape[0], size=n_instances, replace=False)
        else:
            raise ValueError(f"Unknown query strategy: {self.strategy}")
            
        query_instances = X_unlabeled_pool[query_indices]
        print(f"Queried {len(query_indices)} instances using {self.strategy}.")
        return query_indices, query_instances

if __name__ == '__main__':
    # Простой тест ActiveLearner
    from model_trainer import SimpleModel

    # Создаем dummy модель и данные
    dummy_model = SimpleModel()
    X_dummy_train = np.random.rand(5, 10) # Мало данных для начального обучения
    y_dummy_train = np.random.randint(0, 2, size=5)
    dummy_model.fit(X_dummy_train, y_dummy_train)

    X_dummy_unlabeled = np.random.rand(20, 10)

    # Тест uncertainty sampling
    learner_uncertainty = ActiveLearner(strategy="uncertainty_sampling")
    q_idx_unc, q_inst_unc = learner_uncertainty.query(dummy_model, X_dummy_unlabeled, n_instances=3)
    print("Uncertainty Query Indices:", q_idx_unc)
    # print("Uncertainty Query Instances:\n", q_inst_unc)

    # Тест random sampling
    learner_random = ActiveLearner(strategy="random_sampling")
    q_idx_rand, q_inst_rand = learner_random.query(dummy_model, X_dummy_unlabeled, n_instances=3)
    print("Random Query Indices:", q_idx_rand)