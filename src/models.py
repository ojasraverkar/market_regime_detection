'''
trains hmm and decodes states
'''
import numpy as np
from hmmlearn import hmm

def train_hmm(features: np.ndarray, n_states: int, covariance_type: str = 'full', n_iter: int = 1000, random_state: int = 42) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(features)
    return model

def decode_states(model: hmm.GaussianHMM, features: np.ndarray) -> np.ndarray:
    return model.predict(features)      # predict most likely states for given features

def print_model_params(model: hmm.GaussianHMM, feature_names: list = None):
    print('\n=== learned model parameters ===')
    print('transition matrix    (rows=from, cols=to)')
    print(model.transmat_)

    if feature_names is None:
        feature_names = [f"feat{i}" for i in range(model.means_.shape[1])]

    print("\nstate means:")
    for i, means in enumerate(model.means_):
        print(f" state {i}: "+",".join(f"{name}={val:.2f}" for name, val in zip(feature_names, means)))
    
    print("\nstate covariances:")
    for i, cov in enumerate(model.covars_):
        print(f" state {i}:\n{cov}")
