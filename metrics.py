import numpy as np

#TODO fix all this code
#TODO put also loss functions here

def compute_cell_based_rmse(wf1, wf2):
    diff = wf1 - wf2
    return np.sqrt(np.mean(np.square(diff), axis=1))

def compute_field_based_rmse(wf1, wf2):
    diff = wf1 - wf2
    return np.sqrt(np.mean(np.square(diff)))



def compute_cell_based_mae(wf1, wf2):
    diff = wf1 - wf2
    return np.mean(np.abs(diff), axis=1)

def compute_field_based_mae(wf1, wf2):
    diff = wf1 - wf2
    return np.mean(np.abs(diff))


#TODO WIP (remove these functions from model files and put there only here)
import torch #TODO
def MSE(y_predicted:torch.Tensor, y_target:torch.Tensor):
    """
    Returns a single value tensor with
    the mean of squared errors (MSE) between the predicted and target
    values:
    
    """
    error = y_predicted - y_target # element-wise substraction
    return torch.sum(error**2 ) / error.numel() # mean (sum/n)

def mean_speed_error(y_predicted:torch.Tensor, y_target:torch.Tensor):
    error = y_predicted - y_target
    return torch.mean(error, 1)