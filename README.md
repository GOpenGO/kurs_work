# Active Learning for Medical Image Dataset Creation (Course Project Simulation)

This project provides a simulated framework for exploring active learning (AL) methodologies aimed at optimizing the creation of datasets from medical images. It is developed as part of a course project to investigate the theoretical principles and practical considerations of applying AL techniques in the context of medical image analysis, with a conceptual leaning towards deep learning approaches using PyTorch.

## Project Overview

The core objective of this project is to simulate and understand the iterative cycle of active learning. Medical image annotation is a notoriously time-consuming and expensive process, often requiring expert medical knowledge. Active learning seeks to alleviate this bottleneck by intelligently selecting the most informative unlabeled samples for annotation, thereby maximizing model performance gain for a given labeling budget.

This simulation focuses on:
-   Implementing a modular structure for different components of an AL system (data loading, model training, query strategies).
-   Simulating the interaction between a learning model, an unlabeled data pool, an active learning strategy, and an oracle (expert annotator).
-   Conceptually aligning with deep learning practices, such as using PyTorch-style datasets and model structures (though the actual model complexity is simplified for this simulation).
-   Allowing for configuration-driven experiments via YAML files.

## Project Structure

The project is organized into the following directories and key files:

-   `configs/`: Contains YAML configuration files for model architecture, training parameters, and active learning settings.
    -   `model_config.yaml`: Defines model type, layers, training hyperparameters.
    -   `active_learning_config.yaml`: Specifies AL strategy, query parameters, and simulation settings.
-   `data/`: Placeholder intended for sample medical image data.
    -   `initial_labeled_images/`: For a small seed set of labeled images (and/or masks).
    -   `unlabeled_pool_images/`: For the larger pool of unlabeled images.
    *(Note: Currently, the simulation primarily uses synthetically generated data for simplicity).*
-   `src/`: Contains the core Python source code for the simulation.
    -   `datasets/`: Modules for data handling.
        -   `medical_image_dataset.py`: A custom PyTorch `Dataset` class for medical images.
        -   `transforms.py`: Image transformation pipelines using `torchvision.transforms`.
    -   `models/`: Modules related to predictive models.
        -   `base_model.py`: An abstract base class for models.
        -   `simple_cnn.py`: A basic Convolutional Neural Network (CNN) implemented in PyTorch.
    -   `strategies/`: Implementations of different active learning query strategies.
        -   `base_strategy.py`: An abstract base class for query strategies.
        -   `uncertainty.py`: Uncertainty-based sampling strategies (e.g., least confident, margin, entropy).
        -   `random_sampling.py`: A baseline random sampling strategy.
    -   `utils/`: Utility functions.
        -   `visualization.py`: Functions for plotting results (e.g., learning curves).
        -   `metrics.py`: Functions for calculating performance metrics (e.g., accuracy, F1-score).
    -   `active_learning_loop.py`: Orchestrates the main active learning simulation cycle, integrating data, model, and strategy.
-   `notebooks/`: (Optional) Jupyter notebooks for exploratory data analysis, model testing, or detailed visualization of AL iterations.
-   `main_simulation.py`: The primary executable script to run the active learning simulation from the command line.
-   `evaluation_runner.py`: (Placeholder) A script intended for running final evaluations on a "test" set after the AL process.
-   `requirements.txt`: A list of Python package dependencies for the project.
-   `README.md`: This descriptive file.

## Core Active Learning Cycle Simulated

The simulation implements the following iterative steps:

1.  **Initialization:**
    *   Load model and active learning configurations.
    *   Prepare initial (small) labeled dataset and a larger pool of unlabeled data (simulated).
    *   Initialize a predictive model (e.g., `SimpleCNN`).
2.  **Initial Training:** Train the model on the initial labeled dataset.
3.  **Active Learning Loop (repeated for `num_total_queries`):**
    a.  **Query Instance Selection:**
        i.  The model makes predictions on the current unlabeled data pool.
        ii. An active learning strategy (e.g., `UncertaintySampling`) analyzes these predictions to select the `n_instances_per_query` most "informative" samples.
    b.  **Oracle Annotation:** The selected samples are "sent" to a simulated oracle, which provides their true labels.
    c.  **Dataset Augmentation:** The newly labeled samples are added to the labeled training set and removed from the unlabeled pool.
    d.  **Model Retraining:** The model is retrained (or fine-tuned) on the augmented labeled dataset.
    e.  **Evaluation:** The model's performance is evaluated (e.g., on a conceptual hold-out test set) to track progress.

## Technologies and Libraries

-   **Python 3.x**
-   **PyTorch**: For building and training neural network models and handling datasets.
-   **Torchvision**: For image transformations.
-   **NumPy**: For numerical operations.
-   **Scikit-learn**: Conceptually for some metrics or simpler model comparisons (though the main model is PyTorch-based).
-   **Matplotlib**: For plotting results and visualizations.
-   **PyYAML**: For managing configuration files.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd active_learning_medical_images
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the simulation (Optional):**
    Modify `configs/model_config.yaml` and `configs/active_learning_config.yaml` to change model parameters, training settings, or active learning strategies and their parameters.

5.  **Run the simulation:**
    ```bash
    python main_simulation.py --model_config configs/model_config.yaml --al_config configs/active_learning_config.yaml
    ```
    Or simply:
    ```bash
    python main_simulation.py
    ```
    (This will use the default config paths specified in `main_simulation.py`)

## Project Goals & Disclaimer

This project serves as an **educational simulation** to explore and demonstrate the mechanisms of active learning within the context of medical image analysis.
-   It **does not use real patient data**. All data is synthetically generated for illustrative purposes.
-   The implemented models (e.g., `SimpleCNN`) are simplified and not intended for clinical diagnostic use.
-   The primary aim is to understand the workflow, component interactions, and potential benefits/challenges of active learning rather than achieving state-of-the-art performance on a specific medical task.

This work is part of a course requirement and is intended to showcase an understanding of active learning principles and their conceptual application.
