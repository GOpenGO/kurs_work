import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class SimpleCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Извлекаем параметры из конфига
        input_channels = self.config['architecture'].get('input_channels', 1)
        num_classes = self.config['architecture'].get('num_classes', 2)

        # Для примера, используем одну из конфигураций сверточных слоев
        # В реальном проекте здесь была бы более гибкая генерация слоев на основе конфига
        conv_conf = self.config['architecture'].get('conv_layers', [
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ])

        self.conv1 = nn.Conv2d(input_channels, conv_conf[0]['out_channels'],
                               kernel_size=conv_conf[0]['kernel_size'],
                               stride=conv_conf[0]['stride'],
                               padding=conv_conf[0]['padding'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_conf[0]['out_channels'], conv_conf[1]['out_channels'],
                               kernel_size=conv_conf[1]['kernel_size'],
                               stride=conv_conf[1]['stride'],
                               padding=conv_conf[1]['padding'])

        # Размер выхода после сверток и пулинга нужно будет рассчитать
        # для входа в полносвязные слои. Это зависит от размера входного изображения.
        # Для примера, предположим вход 128x128. После conv1+pool -> 64x64. После conv2+pool -> 32x32.
        # self.fc_input_features = conv_conf[1]['out_channels'] * 32 * 32 # Если вход 128x128
        # Это заглушка, в реальной модели нужно динамически вычислять или передавать
        self.fc_input_features_placeholder = conv_conf[1]['out_channels'] * 8 * 8  # Для входа 32x32 после 2 пулов

        fc_conf = self.config['architecture'].get('fc_layers', [{'out_features': 128}])

        self.fc1 = nn.Linear(self.fc_input_features_placeholder, fc_conf[0]['out_features'])
        self.fc2 = nn.Linear(fc_conf[0]['out_features'], num_classes)

        print(f"SimpleCNN ({self.config.get('model_name', 'SimpleCNN')}) initialized.")
        print(f"  Input Channels: {input_channels}, Num Classes: {num_classes}")
        print(f"  Convolutional Layer 1: {self.conv1}")
        print(f"  Convolutional Layer 2: {self.conv2}")
        print(f"  FC Layer 1: {self.fc1}")
        print(f"  FC Layer 2: {self.fc2}")

    def _calculate_fc_input_features(self, dummy_input_shape=(1, 1, 128, 128)):
        """
        Вспомогательный метод для динамического расчета размера входа для FC слоя.
        Вызывается один раз при инициализации или перед первым forward.
        """
        # dummy_input_shape = (batch_size, channels, height, width)
        # input_channels = self.config['architecture'].get('input_channels', 1)
        # dummy_input = torch.randn(1, input_channels, 128, 128) # Пример для 128x128
        dummy_input = torch.randn(dummy_input_shape)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        num_features = x.view(x.size(0), -1).shape[1]
        return num_features

    def forward(self, x):
        # Динамический расчет размера входа для FC слоя, если еще не был сделан
        # и если self.fc1 еще не был инициализирован с правильным in_features.
        # Это немного "грязно" для forward, лучше делать при __init__.
        # Но для примера динамики:
        if self.fc1.in_features == self.fc_input_features_placeholder and x.ndim == 4:
            try:
                actual_fc_in_features = self._calculate_fc_input_features(dummy_input_shape=x.shape)
                if actual_fc_in_features != self.fc1.in_features:
                    print(
                        f"Dynamically adjusting FC1 in_features from {self.fc1.in_features} to {actual_fc_in_features} based on input shape {x.shape}")
                    fc_conf = self.config['architecture'].get('fc_layers', [{'out_features': 128}])
                    num_classes = self.config['architecture'].get('num_classes', 2)
                    self.fc1 = nn.Linear(actual_fc_in_features, fc_conf[0]['out_features'])
                    self.fc2 = nn.Linear(fc_conf[0]['out_features'], num_classes)
                    self.fc1.to(x.device)  # Перемещаем новые слои на то же устройство
                    self.fc2.to(x.device)

            except Exception as e:
                print(f"Error calculating FC input features dynamically: {e}. Using placeholder.")
                # Продолжаем с placeholder, если что-то пошло не так

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Логиты для CrossEntropyLoss
        return x


if __name__ == '__main__':
    # Пример использования SimpleCNN
    dummy_config = {
        'model_name': "TestCNN",
        'architecture': {
            'input_channels': 1,
            'num_classes': 2,
            'conv_layers': [
                {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ],
            'fc_layers': [{'out_features': 64}]
        }
    }
    cnn_model = SimpleCNN(config=dummy_config)

    # Попробуем передать тензор разного размера
    # test_input_32 = torch.randn(2, 1, 32, 32) # batch_size=2, channels=1, H=32, W=32
    # output_32 = cnn_model(test_input_32)
    # print("Output shape for 32x32 input:", output_32.shape) # Ожидаем (2, num_classes)

    test_input_64 = torch.randn(2, 1, 64, 64)  # batch_size=2, channels=1, H=64, W=64
    output_64 = cnn_model(test_input_64)
    print("Output shape for 64x64 input:", output_64.shape)  # Ожидаем (2, num_classes)