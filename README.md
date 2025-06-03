# Active Learning for Medical Image Dataset Creation (Course Project Simulation)

This project simulates the core components of an active learning system tailored for the conceptual task of automating dataset creation from medical images. It is developed as part of a course project to explore and understand the principles of active learning.

## Project Structure

- `data/`: Placeholder for example image data (currently uses simulated data).
  - `initial_labeled_images/`: For a small seed set of labeled images.
  - `unlabeled_pool_images/`: For the larger pool of unlabeled images.
- `src/`: Contains the Python source code for the simulation.
  - `data_loader.py`: Handles (simulated) data loading.
  - `model_trainer.py`: Encapsulates model training and prediction logic (uses a simple scikit-learn model).
  - `active_learner.py`: Implements active learning query strategies (e.g., uncertainty sampling).
  - `oracle.py`: Simulates an expert annotator.
  - `main.py`: The main script to run the active learning simulation loop.
  - `utils.py`: Utility functions, e.g., for plotting results.
- `notebooks/`: (Optional) Jupyter notebooks for experiments and visualization.
- `requirements.txt`: Project dependencies.
- `README.md`: This file.

## Core Idea

The project demonstrates a simplified active learning cycle:
1. An initial model is trained on a small set of labeled data.
2. The model makes predictions on a pool of unlabeled data.
3. An active learning strategy (e.g., uncertainty sampling) selects the most "informative" unlabeled samples.
4. A simulated "oracle" provides labels for these queried samples.
5. The newly labeled samples are added to the training set.
6. The model is retrained, and the cycle repeats.

## How to Run (Conceptual)

(This section would describe how to run `main.py` if it were a fully functional application)

```bash
pip install -r requirements.txt
python src/main.py