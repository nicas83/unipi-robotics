from utils.utils import compare_model_metrics, split_cartesian_workspace, compare_continual_learning_metrics

kinematics = 'inverse'
metrics_path = 'model/metrics/final_*'+kinematics+'_training_metrics.npz'
## all: plotta tutto
## loss: plotta i loss
## execution: plotta il tempo di esecuzione del training
## #epoch: plotta il tempo di ciascuna epoca
## epoch-loss: plotta il tempo di ciascuna epoca e gli sovrappone il validation loss per epoca
compare_model_metrics(metrics_path, kinematics, kind='loss')

cl_metrics_path = 'continual_learning/metrics/final_*_quadII_metrics.npz'
# compare_continual_learning_metrics(cl_metrics_path)

# split_cartesian_workspace('datasets/new_dataset.txt')
