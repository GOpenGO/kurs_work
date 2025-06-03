
import argparse
from src.active_learning_loop import active_learning_simulation # Импортируем из src
from src.utils.visualization import plot_learning_curves # Предполагаем, что эта функция есть

def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Simulation for Medical Images.")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml", # Путь по умолчанию
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--al_config",
        type=str,
        default="configs/active_learning_config.yaml", # Путь по умолчанию
        help="Path to the active learning configuration YAML file."
    )
    args = parser.parse_args()

    print(f"Using Model Config: {args.model_config}")
    print(f"Using AL Config: {args.al_config}")

    history = active_learning_simulation(args.model_config, args.al_config)

    if history:
        print("\nPlotting learning curves...")
        plot_learning_curves(history) # Отображаем графики

if __name__ == "__main__":
    main()