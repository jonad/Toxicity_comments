import random
import torch
import numpy as np
import os


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def compute_weights(df):
    train = df
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    # Overall
    weights = np.ones((len(train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    return loss_weight, weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))