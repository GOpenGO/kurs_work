import numpy as np

class SimulatedOracle:
    def __init__(self, X_full_reference, y_full_reference):
        """
        Инициализация симулированного оракула.
        Оракул "знает" все истинные метки для всего датасета.
        """
        self.X_ref = X_full_reference
        self.y_ref = y_full_reference
        print("SimulatedOracle initialized.")

    def query_labels(self, query_instances):
        """
        Возвращает истинные метки для запрошенных экземпляров.
        В реальном проекте здесь был бы интерфейс для ручной разметки.
        """
        labels = []
        for instance in query_instances:
            # Находим соответствующий экземпляр в полном наборе данных
            # Это упрощенный поиск, предполагающий точное совпадение.
            # В реальности может потребоваться более сложный поиск по ID или хэшу.
            try:
                # np.where((self.X_ref == instance).all(axis=1)) вернет кортеж, берем первый элемент (массив индексов)
                # и из него первый индекс
                idx = np.where((self.X_ref == instance).all(axis=1))[0][0]
                labels.append(self.y_ref[idx])
            except IndexError:
                # Если экземпляр не найден (не должно случиться в этой симуляции, если все правильно)
                print(f"Warning: Instance not found in oracle's reference data. Returning random label.")
                labels.append(np.random.randint(0, np.max(self.y_ref) + 1)) # Случайная метка из возможных
        
        print(f"Oracle provided {len(labels)} labels.")
        return np.array(labels)

if __name__ == '__main__':
    # Простой тест Oracle
    X_total_dummy = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
    y_total_dummy = np.array([0, 1, 0, 1, 0])

    oracle_sim = SimulatedOracle(X_total_dummy, y_total_dummy)

    # Симулируем запрос нескольких экземпляров
    queried_X = np.array([[2,2], [5,5]])
    retrieved_labels = oracle_sim.query_labels(queried_X)
    print("Queried Instances:\n", queried_X)
    print("Retrieved Labels:", retrieved_labels) # Ожидаем [1, 0]

    queried_X_unknown = np.array([[6,6]]) # Неизвестный экземпляр
    retrieved_labels_unknown = oracle_sim.query_labels(queried_X_unknown)
    print("Unknown Queried Instances:\n", queried_X_unknown)
    print("Retrieved Labels for Unknown:", retrieved_labels_unknown)