from utils import compare_model_metrics, split_cartesian_workspace, compare_continual_learning_metrics

# metrics_path = 'model/metrics/final_*_training_metrics.npz'
# compare_model_metrics(metrics_path)

cl_metrics_path = 'continual_learning/metrics/final_*_quadII_metrics.npz'
compare_continual_learning_metrics(cl_metrics_path)

# split_cartesian_workspace('datasets/new_dataset.txt')
