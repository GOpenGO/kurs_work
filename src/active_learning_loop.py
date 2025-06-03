import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import yaml  # Для загрузки конфигов

from .datasets.medical_image_dataset import MedicalImageDataset
from .datasets.transforms import get_medical_image_transforms
from .models.simple_cnn import SimpleCNN  # Предполагаем, что это наша основная модель
# from .strategies.uncertainty import UncertaintySampling # Будет импортировано динамически
# from .strategies.random_sampling import RandomSampling # Будет импортировано динамически
from .utils.metrics import calculate_accuracy  # Предполагаем, что такая функция есть
from .utils.visualization import plot_learning_curves  # Предполагаем


def get_strategy_instance(strategy_name, config):
    if strategy_name == "UncertaintySampling":
        from .strategies.uncertainty import UncertaintySampling
        return UncertaintySampling(config)
    elif strategy_name == "RandomSampling":
        from .strategies.random_sampling import RandomSampling  # Нужно будет создать этот файл
        return RandomSampling(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_preds += inputs.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds.double() / total_preds
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds.double() / total_preds
    # Здесь можно добавить другие метрики, используя all_labels, all_preds
    # Например, F1, precision, recall из utils.metrics
    return epoch_loss, epoch_acc


def active_learning_simulation(model_config_path, al_config_path):
    # --- Загрузка конфигураций ---
    with open(model_config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    with open(al_config_path, 'r') as f:
        al_cfg = yaml.safe_load(f)

    device = torch.device(model_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Подготовка данных (заглушки) ---
    # В реальном проекте здесь будет загрузка с диска, разделение на train/val/test/pool
    print("Simulating data loading...")
    num_total_samples = al_cfg['simulation_params']['initial_labeled_size'] + al_cfg['simulation_params'][
        'unlabeled_pool_size']
    # Для примера создадим случайные "изображения" (тензоры) и метки
    # Вместо путей к файлам, сразу создадим данные
    # Предполагаем входной размер для SimpleCNN, например 64x64
    image_h, image_w = 64, 64
    all_data_sim = torch.randn(num_total_samples, model_cfg['architecture']['input_channels'], image_h, image_w)
    all_labels_sim = torch.randint(0, model_cfg['architecture']['num_classes'], (num_total_samples,))

    # Разделение на начальные размеченные, пул неразмеченных и тестовый набор (для оценки)
    # Для простоты, возьмем часть из `all_data_sim` как "тестовую" один раз.
    # Остальное будет делиться на initial_labeled и unlabeled_pool.

    test_size_ratio = 0.2  # 20% на тест
    num_test_samples = int(num_total_samples * test_size_ratio)
    test_indices = np.random.choice(num_total_samples, num_test_samples, replace=False)
    train_pool_indices = np.setdiff1d(np.arange(num_total_samples), test_indices)

    X_test_sim = all_data_sim[test_indices]
    y_test_sim = all_labels_sim[test_indices]

    X_train_pool_sim = all_data_sim[train_pool_indices]
    y_train_pool_sim_true = all_labels_sim[train_pool_indices]  # Истинные метки для симуляции оракула

    initial_labeled_size = al_cfg['simulation_params']['initial_labeled_size']
    if initial_labeled_size > len(X_train_pool_sim):
        initial_labeled_size = len(X_train_pool_sim)  # Не может быть больше, чем доступно

    initial_labeled_indices_in_pool = np.random.choice(len(X_train_pool_sim), initial_labeled_size, replace=False)

    # Создаем маску для отслеживания размеченных/неразмеченных в общем пуле (X_train_pool_sim)
    is_labeled_mask = np.zeros(len(X_train_pool_sim), dtype=bool)
    is_labeled_mask[initial_labeled_indices_in_pool] = True

    # Трансформации (без аугментации для простоты симуляции)
    img_transforms = get_medical_image_transforms(image_size=(image_h, image_w), augment=False)

    # --- Инициализация модели, оптимизатора, критерия потерь ---
    model = SimpleCNN(config=model_cfg).to(device)
    optimizer_name = model_cfg['training_params'].get('optimizer', 'Adam')
    lr = model_cfg['training_params'].get('learning_rate', 0.001)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:  # SGD по умолчанию
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion_name = model_cfg['training_params'].get('loss_function', 'CrossEntropyLoss')
    if criterion_name == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {criterion_name}")

    # --- Инициализация стратегии активного обучения ---
    query_strategy_name = al_cfg.get('active_learning_strategy', 'UncertaintySampling')
    al_strategy = get_strategy_instance(query_strategy_name, al_cfg)
    n_instances_per_query = al_cfg['query_params'].get('n_instances_per_query', 10)

    # --- Цикл активного обучения ---
    num_total_queries = al_cfg['simulation_params'].get('num_total_queries', 20)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'labeled_size': []}

    # Тестовый DataLoader (остается неизменным)
    # Для симуляции датасета, передаем тензоры напрямую
    test_dataset_sim = MedicalImageDataset(X_test_sim.numpy(), labels=y_test_sim.numpy(),
                                           transform=img_transforms)  # transform ожидает numpy HxW или PIL
    test_dataloader = DataLoader(test_dataset_sim, batch_size=model_cfg['training_params']['batch_size'], shuffle=False)

    for query_iteration in range(num_total_queries + 1):  # +1 для начального обучения
        print(
            f"\n{'=' * 10} Active Learning Iteration: {'Initial Training' if query_iteration == 0 else query_iteration} {'=' * 10}")

        # Формируем текущие размеченные и неразмеченные датасеты
        current_labeled_indices = np.where(is_labeled_mask)[0]
        current_unlabeled_indices = np.where(~is_labeled_mask)[0]

        print(f"Currently labeled: {len(current_labeled_indices)}, Unlabeled in pool: {len(current_unlabeled_indices)}")
        history['labeled_size'].append(len(current_labeled_indices))

        if len(current_labeled_indices) == 0 and query_iteration > 0:  # Не должно случиться, если initial_labeled_size > 0
            print("No labeled data to train on. Skipping training.")
            # Записываем плохие метрики или пропускаем
            history['train_loss'].append(float('inf'))
            history['train_acc'].append(0.0)
            history['val_loss'].append(float('inf'))
            history['val_acc'].append(0.0)
        else:
            # Создаем DataLoader для текущих размеченных данных
            # Для MedicalImageDataset передаем np.array данных и меток
            X_labeled_current_data = X_train_pool_sim[current_labeled_indices].numpy()
            y_labeled_current_labels = y_train_pool_sim_true[current_labeled_indices].numpy()

            # Убедимся, что данные имеют правильный формат для MedicalImageDataset
            # MedicalImageDataset ожидает, что если это не пути, то это массив numpy изображений (N, H, W) или (N, H, W, C)
            # Наши X_train_pool_sim это (N, C, H, W) тензоры, нужно изменить для MedicalImageDataset, если он ожидает H,W
            # Для простоты сейчас будем считать, что MedicalImageDataset принимает (N, C, H, W) тензоры напрямую, если is_unlabeled=False
            # и transform=None, или он должен быть адаптирован.
            # Проще всего будет передать X_train_pool_sim[current_labeled_indices].numpy().transpose(0,2,3,1) если C на последнем месте
            # или X_train_pool_sim[current_labeled_indices].numpy() если С уже там где надо для ToPILImage.
            # Для простоты, если MedicalImageDataset принимает тензоры, то ОК.
            # Если он принимает numpy и делает ToPILImage, то ему нужно (H,W) или (H,W,C)

            # Для простоты этого примера, пусть MedicalImageDataset адаптируется к тензорам (N,C,H,W) если transform=None
            # Или мы создаем его так, чтобы он принимал пути и грузил их
            # Сейчас для заглушки, создадим "фейковые" пути или используем данные напрямую
            # Для этого примера, будем считать, что мы создаем датасет из уже загруженных numpy данных.
            # И transform ожидает numpy (H, W) или (H, W, C).
            # all_data_sim у нас (N, C, H, W). Нужно (N, H, W) для одноканального, если C=1 и transform его ожидает.

            # Переделаем X_labeled_current_data, если это тензор (N,1,H,W) в (N,H,W) np.array
            if X_labeled_current_data.ndim == 4 and X_labeled_current_data.shape[1] == 1:  # (N,1,H,W)
                X_labeled_current_data_np = X_labeled_current_data.squeeze(1)  # (N,H,W)
            else:  # Предполагаем, что это уже (N,H,W) или (N,H,W,C)
                X_labeled_current_data_np = X_labeled_current_data

            if len(X_labeled_current_data_np) > 0:
                labeled_train_dataset = MedicalImageDataset(
                    X_labeled_current_data_np,
                    labels=y_labeled_current_labels,
                    transform=img_transforms
                )
                train_dataloader = DataLoader(labeled_train_dataset,
                                              batch_size=model_cfg['training_params']['batch_size'], shuffle=True)

                # Обучение модели
                num_epochs = model_cfg['training_params']['num_epochs_initial'] if query_iteration == 0 else \
                model_cfg['training_params']['num_epochs_active_retraining']
                print(f"Training for {num_epochs} epochs...")
                for epoch in range(num_epochs):
                    train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
                    # print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc.item())
                print(f"Finished training. Final Train Loss: {train_loss:.4f}, Train Acc: {train_acc.item():.4f}")
            else:  # Если размеченных нет (только при query_iteration=0 и initial_labeled_size=0)
                print("No labeled data for initial training. Skipping.")
                history['train_loss'].append(float('inf'))
                history['train_acc'].append(0.0)

            # Оценка на тестовом наборе
            if X_test_sim.shape[0] > 0:
                val_loss, val_acc = evaluate_model(model, test_dataloader, criterion, device)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc.item())
                print(f"Evaluation on Test Set - Val Loss: {val_loss:.4f}, Val Acc: {val_acc.item():.4f}")
            else:
                history['val_loss'].append(float('inf'))
                history['val_acc'].append(0.0)

        # Если это не последний шаг (т.е. мы еще будем делать запросы)
        if query_iteration < num_total_queries:
            if len(current_unlabeled_indices) == 0:
                print("No unlabeled samples left in the pool. Stopping simulation.")
                break

            # Формируем DataLoader для неразмеченных данных для стратегии запроса
            # MedicalImageDataset должен уметь работать с is_unlabeled=True и возвращать индексы
            X_unlabeled_current_data = X_train_pool_sim[current_unlabeled_indices].numpy()
            if X_unlabeled_current_data.ndim == 4 and X_unlabeled_current_data.shape[1] == 1:
                X_unlabeled_current_data_np = X_unlabeled_current_data.squeeze(1)
            else:
                X_unlabeled_current_data_np = X_unlabeled_current_data

            unlabeled_query_dataset = MedicalImageDataset(
                X_unlabeled_current_data_np,
                transform=img_transforms,
                is_unlabeled=True
            )
            # Batch size для запроса может быть больше, чтобы быстрее обработать пул
            unlabeled_query_dataloader = DataLoader(unlabeled_query_dataset,
                                                    batch_size=model_cfg['training_params']['batch_size'] * 2,
                                                    shuffle=False)

            print("Querying new instances for labeling...")
            # Стратегия AL возвращает оригинальные индексы относительно всего X_train_pool_sim
            # Поэтому нам нужно передать ей индексы current_unlabeled_indices, чтобы она знала,
            # какие из них она выбрала. Либо, она возвращает индексы относительно переданного ей даталоадера,
            # а MedicalImageDataset возвращает оригинальные индексы.
            # В `src/strategies/uncertainty.py` мы сделали так, что он работает с даталоадером, который возвращает оригинальные индексы.

            indices_to_label_original = al_strategy.query(model, unlabeled_query_dataloader, n_instances_per_query,
                                                          device)

            if len(indices_to_label_original) == 0:
                print("Query strategy returned no instances. Stopping.")
                break

            # Обновляем маску: помечаем выбранные экземпляры как размеченные
            # Убедимся, что indices_to_label_original - это индексы в рамках X_train_pool_sim
            # (Наша стратегия и датасет должны это обеспечивать)
            is_labeled_mask[indices_to_label_original] = True
            print(f"Marked {len(indices_to_label_original)} new instances as labeled.")
        else:
            print("Reached maximum number of queries.")

    print("\n--- Active Learning Simulation Finished ---")
    # plot_learning_curves(history) # Вызываем визуализацию
    return history


if __name__ == '__main__':
    # Для запуска этого файла напрямую, нужно создать dummy config файлы
    # или передать пути к реальным.
    # Создадим dummy конфиги для теста:

    # Dummy model_config.yaml
    dummy_model_config_content = """
model_name: "SimCNN_AL_Test"
architecture:
  num_classes: 2
  input_channels: 1
  conv_layers:
    - {out_channels: 4, kernel_size: 3, stride: 1, padding: 1} # Меньше каналов для скорости
    - {out_channels: 8, kernel_size: 3, stride: 1, padding: 1}
  fc_layers:
    - {out_features: 32}
training_params:
  learning_rate: 0.005
  batch_size: 4
  num_epochs_initial: 2 # Мало эпох для быстрого теста
  num_epochs_active_retraining: 1
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
device: "cpu" # Для теста на CPU
"""
    with open("dummy_model_config.yaml", "w") as f:
        f.write(dummy_model_config_content)

    # Dummy active_learning_config.yaml
    dummy_al_config_content = """
active_learning_strategy: "UncertaintySampling" 
query_params:
  n_instances_per_query: 3 
  uncertainty_method: "entropy" 
simulation_params:
  num_total_queries: 3 # Мало запросов для быстрого теста
  initial_labeled_size: 5
  unlabeled_pool_size: 50 
"""
    with open("dummy_al_config.yaml", "w") as f:
        f.write(dummy_al_config_content)

    history_data = active_learning_simulation("dummy_model_config.yaml", "dummy_al_config.yaml")
    print("\nSimulation History Keys:", history_data.keys())
    print("Labeled sizes per iteration:", history_data['labeled_size'])
    print("Validation accuracies per iteration:", history_data['val_acc'])

    # Очистка dummy файлов
    import os

    os.remove("dummy_model_config.yaml")
    os.remove("dummy_al_config.yaml")