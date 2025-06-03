import numpy as np
from data_loader import load_initial_labeled_data, load_unlabeled_pool
from model_trainer import SimpleModel
from active_learner import ActiveLearner
from oracle import SimulatedOracle
# from utils import plot_accuracy_history # Предполагается, что такая функция есть в utils.py

def run_active_learning_simulation(num_queries=10, batch_size_per_query=5)
    
    Запускает симуляцию цикла активного обучения.
    
    print(--- Starting Active Learning Simulation ---)

    # 1. Загрузка данных
    # Для симуляции оракула нам нужны все данные и все истинные метки сразу
    # В data_loader.py мы симулируем это
    X_initial_labeled, y_initial_labeled = load_initial_labeled_data()
    
    # Загружаем весь набор неразмеченных данных для пула
    # и истинные метки для симуляции оракула
    # Это немного искусственно, но нужно для симуляции.
    # В реальности y_unlabeled_true не был бы доступен сразу.
    X_pool_unlabeled_full, y_pool_unlabeled_true_full = load_unlabeled_pool_with_true_labels_for_simulation()
    # (Эта функция должна быть добавлена в data_loader.py для этого main.py)

    X_current_labeled = np.copy(X_initial_labeled)
    y_current_labeled = np.copy(y_initial_labeled)
    X_current_unlabeled = np.copy(X_pool_unlabeled_full)
    y_current_unlabeled_true = np.copy(y_pool_unlabeled_true_full) # Для оракула

    # Инициализация компонентов
    model = SimpleModel()
    active_learner = ActiveLearner(strategy=uncertainty_sampling) # или random_sampling
    
    # Для симуляции оракула ему нужно знать все истинные метки
    # Объединим начальные размеченные и истинные неразмеченные для справки оракула
    X_total_for_oracle = np.vstack((X_initial_labeled, X_pool_unlabeled_full))
    y_total_for_oracle = np.concatenate((y_initial_labeled, y_pool_unlabeled_true_full))
    oracle = SimulatedOracle(X_total_for_oracle, y_total_for_oracle)

    accuracies = [] # Будем хранить точность на каждой итерации

    # Начальное обучение, если есть данные
    if X_current_labeled.shape[0]  0
        model.fit(X_current_labeled, y_current_labeled)
        # Оцениваем точность (заглушка, в реальности нужна тестовая выборка)
        # Здесь для примера будем оценивать на всем доступном истинном наборе
        if X_total_for_oracle.shape[0]  0 
            initial_preds = model.predict(X_total_for_oracle)
            initial_acc = np.mean(initial_preds == y_total_for_oracle)
            accuracies.append(initial_acc)
            print(fInitial accuracy {initial_acc.4f} on {X_current_labeled.shape[0]} labeled samples.)

    # Цикл активного обучения
    for i in range(num_queries)
        print(fn--- Query Cycle {i+1}{num_queries} ---)
        if X_current_unlabeled.shape[0] == 0
            print(No more unlabeled data to query.)
            break

        # 2. Выбор экземпляров для запроса
        query_indices_in_current_unlabeled, queried_instances = active_learner.query(
            model, X_current_unlabeled, n_instances=batch_size_per_query
        )

        if len(query_indices_in_current_unlabeled) == 0
            print(Active learner did not select any instances.)
            break
            
        # 3. Запрос меток у оракула
        new_labels = oracle.query_labels(queried_instances)

        # 4. Добавление новых размеченных данных
        X_current_labeled = np.vstack((X_current_labeled, queried_instances))
        y_current_labeled = np.concatenate((y_current_labeled, new_labels))

        # Удаление из неразмеченного пула
        X_current_unlabeled = np.delete(X_current_unlabeled, query_indices_in_current_unlabeled, axis=0)
        # Также обновляем y_current_unlabeled_true для консистентности, хотя он прямо не используется в цикле, кроме как оракулом
        y_current_unlabeled_true = np.delete(y_current_unlabeled_true, query_indices_in_current_unlabeled, axis=0)


        # 5. Переобучение модели
        model.fit(X_current_labeled, y_current_labeled)

        # Оценка точности (заглушка)
        current_preds = model.predict(X_total_for_oracle) # Оцениваем на всем
        current_acc = np.mean(current_preds == y_total_for_oracle)
        accuracies.append(current_acc)
        print(fCycle {i+1} Labeled={X_current_labeled.shape[0]}, Unlabeled={X_current_unlabeled.shape[0]}, Accuracy={current_acc.4f})

    print(n--- Active Learning Simulation Finished ---)
    print(Accuracy progression, accuracies)
    
    # Здесь можно было бы вызвать функцию для построения графика точности
    # plot_accuracy_history(accuracies, num_queries)


# Вспомогательная функция для data_loader.py, чтобы main.py работал
def load_unlabeled_pool_with_true_labels_for_simulation(data_path=dataunlabeled_pool_images)
    
    Заглушка для загрузки большого пула неразмеченных данных ВМЕСТЕ с их истинными метками
    (ТОЛЬКО для симуляции работы оракула в main.py).
    
    print(fLoading unlabeled data pool with true labels (for simulation) from {data_path}...)
    X_unlabeled = np.random.rand(100, 10)
    y_unlabeled_true = np.random.randint(0, 2, size=100) # Истинные метки
    print(fLoaded {X_unlabeled.shape[0]} unlabeled samples with their true labels.)
    return X_unlabeled, y_unlabeled_true


if __name__ == '__main__'
    # Добавляем вызов функции в data_loader, чтобы main не падал при импорте
    # Это просто для примера, чтобы показать, что функция должна существовать.
    # В идеале, эта функция должна быть в data_loader.py
    import data_loader
    data_loader.load_unlabeled_pool_with_true_labels_for_simulation = load_unlabeled_pool_with_true_labels_for_simulation
    
    run_active_learning_simulation(num_queries=5, batch_size_per_query=2)