from sklearn.metrics import r2_score
import os
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import mne
import os
import sys
# sys.path.append("..")
# from utils.paths import paths
# from utils.load import *
# from utils.setup import * 
import numpy as np
from ..utils.functions import (start_stop_samples_trigg , start_samples_trigg, add_trial_info_to_events,
    plot_fix_durs_mem_vs,plot_fix_durs_all_phases,closest_tuple,plot_trial, create_full_metadata,
    pick_chs,get_effect_name_from_cond_dict,fig,is_inside_window)
from scipy.interpolate import BSpline
import matplotlib.patches as patches
import pandas as pd
from  .plot_general import rose_plot, plot_eye_movements, plot_et_cond
from .winrej import cont_ArtifactDetect 
from sklearn.linear_model import LinearRegression, Ridge
#----------import experimental unfold------
from .analysis_functions import (define_events ,saccades_intercept_evts_prev,saccades_intercept_evts_next,
    saccades_intercept_evts_nearest,balcklisted_trials)
# from pydeconv.pydeconv import PyDeconv, create_design_matrix
import config


def compute_vif_intercept_ave(model, X):
    # delays_ = model.delays_
    vifregr = LinearRegression()
    delays_ = np.arange(int(np.round(model.tmin * model.sfreq)), int(np.round(model.tmax * model.sfreq) + 1))
    # print(f'delays are : {delays_}')
    cols = X.shape[1]
    list_of_vif =[]
    for coef in range(len(model.feature_names)+1):
        intercept_vif = []
        delays_vif = range(len(delays_)*coef,len(delays_)*(coef+1))
        num_samples = 20 # number of delay samples per predictor to average
        step_size = len(delays_vif) // num_samples
        # Create a subsample of indices equally spaced
        delays_vif = delays_vif[::step_size][:num_samples]

        for i, delay in enumerate(delays_vif):
            print(f'coef: {coef} VIF: Fitting the delay {i} of {len(delays_vif)} , {100*i/len(delays_vif):.2f}%')
            x_i = X[:, delay].toarray()  # Convert to dense array
            mask = np.arange(cols) != delay
            x_noti = X[:, mask]
            #print(f"x_noti shape: {x_noti.shape}, x_i shape: {x_i.shape}")
            #print(f"x_noti type: {type(x_noti)}, x_i type: {type(x_i)}")
            vifregr.fit( x_noti,x_i)
            intercept_r_sq = vifregr.score( x_noti,x_i,)
            intercept_vif.append(1./(1.-intercept_r_sq))
        
        list_of_vif.append(np.array(intercept_vif).mean())
    return np.array(list_of_vif)

def compute_categorical_balance(model, X):
    # delays_ = model.delays_
    delays_ = np.arange(int(np.round(model.tmin * model.sfreq)), int(np.round(model.tmax * model.sfreq) + 1))
    list_of_n_coeffs = []


    for coef in range(len(model.feature_names)+1):
        first_delay = len(delays_)*coef
       # number of delay samples per predictor to average

        x_i = X[:, first_delay].toarray()  # Convert to dense array
        list_of_n_coeffs.append(np.count_nonzero(x_i))

    return list_of_n_coeffs


def create_spline_matrix(sacc_metadata):
    print('creating default splines: 5 cubic B-splines matrix')
    # reference on how to define knots
    # https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html
    event_sac = sacc_metadata.loc[sacc_metadata.type=='saccade']
    x_data = np.array(event_sac['sac_amplitude']).reshape(1,-1)[0] #here I should have the data i.e. list or array of sac_amplitudes of events to regress
    k = 3  # cubic splines
    knots = 3 # 
    t = np.quantile(x_data, np.linspace(0,1,knots))# internal knots
    t = np.r_[[t[0]]*k, t, [t[-1]]*k]   # add boundary knots
    splines_design_matrix = BSpline.design_matrix(x_data, t, k).toarray()
    # this is what unfold does to choose the spline to be remove. It finds the location of the predictor closest to the mean.
    meanid = np.abs(x_data-np.mean(x_data)).argmin()
    row_mean = splines_design_matrix[meanid, :]  #
    spline_to_kill = np.argmax(row_mean)

    new_matrix = np.delete(splines_design_matrix, spline_to_kill, axis=1)
    return new_matrix

def add_spline_features(sacc_metadata):
    new_matrix = create_spline_matrix(sacc_metadata)
    features = []
    event_sac = sacc_metadata.loc[sacc_metadata.type=='saccade']
    assert event_sac.shape[0] == new_matrix.shape[0], "Number of rows in DataFrame does not match number of rows in matrix"
    for i in range(new_matrix.shape[1]):
        # Append the string with the column index to the feature_names list
        features.append(f'spline_{i}')
    # define pandas metadata for the interest events

    event_sac = pd.concat([event_sac.reset_index(drop=True), pd.DataFrame(new_matrix, columns=features)], axis=1)
    return event_sac , features

# Define a function to calculate log-likelihood
def ll_aic_bic_cp(y_true, y_pred, X, lamb):
    from sklearn.metrics import mean_squared_error
    from scipy.sparse.linalg import svds
    import os
    os.environ['SCIPY_USE_PROPACK'] = "True"#'1'  # Enable propack solver for svds

    n_channels = y_true.shape[1]
    n = len(y_true)
    ll_values = np.zeros(n_channels)
    AIC_values = np.zeros(n_channels)
    BIC_values = np.zeros(n_channels)
    # Cp_values = np.zeros(n_channels)
    SSR = np.sum((y_true - y_pred) ** 2, axis=0)
    # X = X.todense()
    
    # ld = lamb * np.diag(np.ones(X.shape[1]))
    
    
    
    # intent of useing SVD
    # Perform SVD on the design matrix X
    # U, s, Vt = np.linalg.svd(X, full_matrices=False)
    dense = np.array(X.todense())
    # Find rows where all elements are zeros
    nonzero_rows = np.any( dense != 0, axis=1)

    # Filter out rows with all zeros
    filtered_matrix = dense[nonzero_rows]
    # H = np.dot(filtered_matrix, np.dot(np.linalg.inv(np.dot(filtered_matrix.T, filtered_matrix) + lamb), filtered_matrix.T))
    # df = np.trace(H)
    n_samples, n_features = filtered_matrix.shape
    k = min(n_samples, n_features)
    U, s, Vt = svds(filtered_matrix, k, solver = 'propack')  # Adjust the number of singular values computed (e.g., 50) based on your needs

    # Compute effective degrees of freedom
    df = np.sum(s ** 2 / (s ** 2 + lamb))
    
    # MSE = mean_squared_error(y_true, y_pred,multioutput='raw_values' )
    # ll_values = -n / 2 * (np.log(2 * np.pi * MSE) + 1)
    # k = n_params  # Number of parameters including intercept
    AIC_values = X.shape[0] * np.log(SSR) + 2 * df  
    BIC_values  = X.shape[0] * np.log(SSR) + 2 * df * np.log(X.shape[0])

    # Cp_values = ll_values + 2 * k * sigma_squared / n
    llaicbiccp = np.stack((AIC_values,BIC_values ), axis=1)
    return llaicbiccp

import re

def parse_wilkinson_formula(formula):
    # Remove spaces and split the formula
    parts = formula.replace(" ", "").split("~")
    
    if len(parts) != 2:
        raise ValueError("Invalid Wilkinson formula format")
    
    right_side = parts[1]
    
    # Check for intercept
    has_intercept = "1+" in right_side
    
    # Remove intercept if present
    if has_intercept:
        right_side = right_side.replace("1+", "")
    
    # Find all interactions
    star_interactions = re.findall(r'([A-Za-z0-9]+)\*([A-Za-z0-9]+)', right_side)
    colon_interactions = re.findall(r'([A-Za-z0-9]+):([A-Za-z0-9]+)', right_side)
    
    # Process star interactions
    additive_features = set()
    formatted_interactions = []
    for a, b in star_interactions:
        additive_features.add(a)
        additive_features.add(b)
        formatted_interactions.append(f"{a}:{b}")
        right_side = right_side.replace(f"{a}*{b}", "")
    
    # Process colon interactions
    for a, b in colon_interactions:
        formatted_interactions.append(f"{a}:{b}")
        right_side = right_side.replace(f"{a}:{b}", "")
    
    # Add remaining terms to additive features
    additive_features.update([term for term in right_side.split("+") if term])
    
    return {
        "intercept": has_intercept,
        "interactions": formatted_interactions,
        "additive_features": list(additive_features)}


def analyze_data():
    # Accessing configuration settings
    first_intercept_ev = config.events_of_interest['first_intercept_event_type']
    second_intercept_ev = config.events_of_interest['second_intercept_event_type']
    second_delay = config.events_of_interest['second_delay']
    
    model_name = config.model['model_name']
    formula = config.model['formula']
    tmin = config.model['tmin']
    tmax = config.model['tmax']
    use_splines = config.model['use_splines']
    alpha = config.model['alpha']
    solver = config.model['solver']
    parsed_formula = parse_wilkinson_formula(formula)
    eeg_chns = config.model['eeg_chns']
    
    # Example use of these settings in your analysis
    print(f"Analyzing data with model: {model_name}")
    print(f"Time range: {tmin} to {tmax}")
    print(f"Solver: {solver}")
    print(f"Alpha value: {alpha}")
    print(f"Alpha value: {alpha}")
 
    # Add your analysis logic here
    # For demonstration purposes, let's assume we generate some analysis results
    results = {
        'first_intercept_event_type': first_intercept_ev,
        'second_intercept_event_type': second_intercept_ev,
        'second_delay': second_delay,
        'model_name': model_name,
        'formula': formula,
        'tmin': tmin,
        'tmax': tmax,
        'use_splines': use_splines,
        'alpha': alpha,
        'analysis_outcome': 'example_outcome',  # Placeholder for actual analysis results
        'eeg_chns' : eeg_chns,
        'parsed_formula': parsed_formula, 
        'solver' : solver
    }

    return results
