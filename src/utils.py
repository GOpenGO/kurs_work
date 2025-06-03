import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_history(accuracies, num_total_queries, initial_accuracy_exists=True):
    """
    Строит график изменения точности в процессе активного обучения.
    """
    plt.figure(figsize=(10, 6))
    
    num_points = len(accuracies)
    x_axis = np.arange(num_points)
    
    labels = []
    if initial_accuracy_exists and num_points > 0:
        labels.append("Initial")
        # Остальные метки для запросов
        labels.extend([f"Q{i+1}" for i in range(num_points - 1)])
    else:
        labels.extend([f"Q{i+1}" for i in range(num_points)])


    plt.plot(x_axis, accuracies, marker='o', linestyle='-')
    plt.title("Model Accuracy Over Active Learning Queries")
    plt.xlabel("Query Iteration / Batch")
    plt.ylabel("Accuracy")
    if labels:
        plt.xticks(x_axis, labels, rotation=45, ha="right")
    
    plt.ylim(0, 1.05) # Точность от 0 до 1
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_data_overview(X_labeled, y_labeled, X_unlabeled, title="Data Overview"):
    """
    Простая визуализация 2D данных (если признаки позволяют).
    """
    if X_labeled.shape[1] != 2 or X_unlabeled.shape[1] != 2:
        print("Data is not 2D, cannot plot overview.")
        return

    plt.figure(figsize=(8, 6))
    if X_labeled.shape[0] > 0:
        plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled, marker='o',
                    label='Labeled Data', cmap='coolwarm', edgecolors='k', s=80)
    if X_unlabeled.shape[0] > 0:
        plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c='lightgray', marker='x',
                    label='Unlabeled Data', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    # Тест plot_accuracy_history
    dummy_accuracies = [0.5, 0.6, 0.65, 0.75, 0.78, 0.80]
    plot_accuracy_history(dummy_accuracies, num_total_queries=5, initial_accuracy_exists=True)

    dummy_accuracies_no_initial = [0.6, 0.65, 0.75, 0.78, 0.80]
    plot_accuracy_history(dummy_accuracies_no_initial, num_total_queries=5, initial_accuracy_exists=False)

    # Тест plot_data_overview
    X_l_dummy = np.random.rand(10, 2)
    y_l_dummy = np.random.randint(0,2,10)
    X_u_dummy = np.random.rand(50,2)
    plot_data_overview(X_l_dummy, y_l_dummy, X_u_dummy)