active_learning_strategy: "UncertaintySampling"  # or "RandomSampling"
query_params:
n_instances_per_query: 10  # Number of samples to query in each AL iteration
# Strategy-specific params for UncertaintySampling
uncertainty_method: "least_confident"  # "least_confident", "margin", "entropy"

simulation_params:
num_total_queries: 20  # Total number of AL iterations
initial_labeled_size: 20
unlabeled_pool_size: 1000  # For simulation from a larger abstract pool

