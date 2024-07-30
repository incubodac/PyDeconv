import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import mne
import os
import sys
sys.path.append("..")
from utils.paths import paths
from utils.load import *
from utils.setup import * 
import numpy as np
from utils.functions import (start_stop_samples_trigg , start_samples_trigg, add_trial_info_to_events,
    plot_fix_durs_mem_vs,plot_fix_durs_all_phases,closest_tuple,plot_trial, create_full_metadata,
    pick_chs,get_effect_name_from_cond_dict,fig,is_inside_window)

import matplotlib.patches as patches
import pandas as pd
from  utils.plot_general import rose_plot, plot_eye_movements, plot_et_cond,plot_categorical_balance
from utils.winrej import cont_ArtifactDetect 
from sklearn.linear_model import LinearRegression, Ridge, LassoLarsIC
#----------import experimental unfold------
from utils.analysis_functions import (define_events ,saccades_intercept_evts_prev,saccades_intercept_evts_next,
    saccades_intercept_evts_nearest,balcklisted_trials)
from unfoldpy.unfoldpy import Unfolder, create_design_matrix
from unfoldpy.unfoldpy_functions import compute_vif_intercept_ave, add_spline_features,compute_categorical_balance,ll_aic_bic_cp
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
# from cuml.linear_model import LinearRegression as cuLinearRegression
import time
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description')
    
    parser.add_argument('--model', type=str, default='default_model.pth', help='Path to the model file')
    parser.add_argument('--shrinkage', type=float, default=0.5, help='Shrinkage parameter')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')

    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_arguments()
    config_file = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_file)
    model_config = config['model']
    events_type = config['events_of_interest']
    params = config['params']
    model_combined_name = model_config['model_name']
    # Use args.model, args.shrinkage, and config as needed in your script logic

    alph = model_config['alpha']
    try_torch = False
    try_cv = True
    ########################--MODEL NAME--###########################################################

    #####################
    #------Experiment---#
    ##################### 
    experiments = 'UON'
    print(experiment)
    model_name = experiment + model_combined_name
    ########################--MODEL-DEFINITION--#####################################################
    #----------------parameters for events of interest (base cond should have None)-----------------#
    intercept_ev= events_type['intercept_ev']
    second_intercept_ev= events_type['second_intercept_ev']
    second_delay = events_type['second_delay']# Set this as True to  add different time span for the second intercept

    additive_feats = model_config['additive_feats']#,'mss'] #None# 'mss' #scrank
    use_splines = model_config['use_splines'] #add 4 cubic splines as features to model second intercept event

    #-------------------define multiplicative interaction-----------------
    #e.g.: A:B 
    interaction_feat = model_config['interaction_feat']#'ontarget:mss'#None#'correct:mss'#'correct:rank' 
    #----------------time limists--------------#
    tmin ,tmax = model_config['tmin'] , model_config['tmax']
    #################################################################################################


    n_chs = 64 #change to 128 if running for UBA experiment only without reduced heads
    path , info_sujs = load_experiment(experiment)
    #NOW UBA include 'ALL' SUBS JAN 9  2024 to include 16 subjects

    rej_sujs = info_sujs.rejected_subjects
    #-----------------------------------------#
    # AMPLITUD REJECTION OPTION with different threshoulds---------
    rejection = True # set to False to reproduce SAN's POSTER results
    use_reduced_heads = False
    #----- Save data and display figures --------------------------
    save_data = True
    save_fig = True
    save_vif = False
    save_balance = False#only to try second delay bc some error when try it 
    aic_bic = False

    display_figs = False
    if display_figs:
        plt.ion()
    else:
        plt.ioff()
    #----- Parameters -----#
    model_saccades = True
    model_button   = False

    #---------------------------------------------------------

    ############################--SOLVER--##########################################
    if alph ==0:
        solver = LinearRegression()    
    elif alph=='lasso': 
        solver = LassoLarsIC(criterion='aic')
    else:
        solver = Ridge(alpha=alph) 


    ##################################
    #----- Paths -------#
    ###################
    model_path = path.model_path(model_name)
    model_save_path = model_path + '/regr_coeffs/' 
    model_fig_path = model_path + '/Plots/'          # Save figures paths

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(model_fig_path, exist_ok=True)
    print('Model path was created as:',model_path)
    #-----------------#

    metadata_path = path.full_metadata_path()
    subjects_ids = info_sujs.subjects_ids 
    subjects_ids = [item for item in subjects_ids if item not in rej_sujs]

    ###################RUN MODEL#############################################
    nfeats = 1 #main intercept
    if second_intercept_ev is not None:
        nfeats +=1
    if use_splines:
        nfeats +=4
    if additive_feats is not None:
        if isinstance(additive_feats,str):
            nfeats +=1
        else: 
            nfeats +=len(additive_feats)
    if interaction_feat is not None:
        if isinstance(interaction_feat,str):
            nfeats +=1
        else: 
            nfeats +=len(interaction_feat)


    print('feature_names before unf',nfeats)
    intercept_vif_aves = []
    categorical_balance_sujs = []
    aic_bic_sujs = []
    subject_code = '629959' 
    #laod subject object
    suj = subject(info_sujs,subject_code)
    # Data filenames
    
    #---------------------suj------------------------------------------------
    eeg = suj.load_analysis_eeg(reduced=use_reduced_heads)
    #eeg = suj.load_electrode_positions(eeg) #uncomment this for UBA 128
    if (n_chs == 64) and use_reduced_heads:
        eeg = eeg.set_montage('biosemi64') #not for UBA 128


    evts = suj.load_metadata()
    evts = balcklisted_trials(suj,evts) #dicard trials with wrong searchimage size

    sr = eeg.info['sfreq']
    # evts['scrank'] = evts.groupby('trial')['rank'].transform(lambda x: (x - x.min()) / (x.max() - x.min())) #this is to create momentarily a scaled version o rank between each trial
    #----------------AMPLITUD BASED REJECTION-----------------------
    
    if rejection:
        ampth = 300e-6 
        chanArray = range(n_chs)
        winms = 2000
        stepms = 1000
        winrej = cont_ArtifactDetect(eeg, amplitudeThreshold=ampth, windowsize=winms, channels=chanArray, stepsize=stepms, combineSegments=1000)

    #-------------------define interest events based on params------------------------------
    interest_events_metadata, int_ev, _ ,_ = define_events(evts,**params)
    #active Search can be apply to the meta data via 'add_active_to_evts()' function 2024
    #interest_events_metadata = add_active_to_evts(interest_events_metadata,thr_dur=200)
    
    ########################################---MSS-encoding--################################################
    # interest_events_metadata['mss'] =   interest_events_metadata['mss'].replace({1:0,2:1,4:3})  #put mss=1 into the intercept
    interest_events_metadata['mss'] =   interest_events_metadata['mss'].replace({1:0,2:1,4:1})  #put mss=1 and  2 and 4 to the effect into the intercept
    # interest_events_metadata.loc[1058,'scrank'] = 0 #bug hardoded for S116 trial with one fixation scaler didnt managed it  
    ########################################--SCRANK--#######################################################
    # Filter out trials with only one instance
    interest_events_metadata = interest_events_metadata.groupby('trial').filter(lambda x: len(x) > 1)

    # Apply transformation
    interest_events_metadata['scrank'] = interest_events_metadata.groupby('trial')['rank'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    ########################################--SACCADES-taking-part--#########################################
    #second_intercept_events_metadata, sec_int_ev, _ ,_ = define_events(evts,**second_intercept_ev_params)
    second_intercept_events_metadata, sec_int_ev = saccades_intercept_evts_nearest(evts, interest_events_metadata)
    #second_intercept_events_metadata, sec_int_ev = saccades_intercept_evts_next(evts, interest_events_metadata)

    #----------------------------mark as bad using 'inside_window' amplitud rejection with True/False values--
    interest_events_metadata['bad'] = interest_events_metadata['latency'].apply(is_inside_window,args=(winrej,))
    second_intercept_events_metadata['bad'] = second_intercept_events_metadata['latency'].apply(is_inside_window,args=(winrej,))
    #-----------reject events marked as bad_ET by the engbert algorithm and by amplitud rejection-------------
    interest_events_metadata        = interest_events_metadata.loc[interest_events_metadata.bad==0]
    second_intercept_events_metadata = second_intercept_events_metadata.loc[second_intercept_events_metadata.bad==0]
    # Let's say 'fixations_df' has the desired number of events
    desired_num_events = len(interest_events_metadata)

    # Subsample 'saccades_df' to match the number of events in 'fixations_df'
    # second_intercept_events_metadata = second_intercept_events_metadata.sample(n=desired_num_events, random_state=42) #comment this to avoid subsampling

    #-----------------------------------------add spline features ----------------------------------------------
    if use_splines:
        print('WARNING: B-SPLINES ARE BEING APPLIED')
        second_intercept_events_metadata, second_intercept_features = add_spline_features(second_intercept_events_metadata)
    else:
        second_intercept_features = None
    #NAN PROBLEMS!!!!!!!!
    #------------------------------------------instantiate model----------------------------------------------
    if interaction_feat is None:
        interaction_feat_arg = []
    else:
        interaction_feat_arg = [interaction_feat]
        
    if second_intercept_features is not None:
        second_intercept_features_arg = second_intercept_features
    else:
        second_intercept_features_arg = []
        
        
    features =  additive_feats+interaction_feat_arg+[second_intercept_ev]+second_intercept_features_arg
    #MOD2024
    unf=Unfolder(
            tmin, tmax, sr, feature_names=features, estimator=solver, scoring='r2',chans_to_ana=n_chs,second_delay=second_delay,tmin2=-.1,tmax2=.18
            )
    print(unf)
    #------------------create matrix-----------------------------------------
    X = create_design_matrix(eeg,tmin,tmax,sr, interest_events_metadata, intercept_ev, additive_feats,interaction=interaction_feat)



    #for second delay use
    if second_delay is True:
        X_sacc = create_design_matrix(eeg,tmin,.3,sr, second_intercept_events_metadata, second_intercept_ev, second_intercept_features,interaction=None)
    if second_delay is None:
        X_sacc = create_design_matrix(eeg,tmin,tmax,sr, second_intercept_events_metadata, second_intercept_ev, second_intercept_features,interaction=None)

    from scipy.sparse import hstack 
    concatenated_matrix = hstack([X, X_sacc])
    print('type of hthe main martix',type(X))
    print('size of main matrix',X.shape)
    print('size of main+sacc matrix',concatenated_matrix.shape)
    X=concatenated_matrix
    # dense = np.array(X.todense())
    if alph=='lasso': 
        X = X.toarray()

    # # Find rows where all elements are zeros
    # nonzero_rows = np.any( dense != 0, axis=1)

    # # Filter out rows with all zeros
    # filtered_matrix = dense[nonzero_rows]
    # print(f'size of the matrix with zeros {X.shape}')
    # print(f'size of the matrix without zeros {filtered_matrix.shape}')


    #--------train model-----------------------------------------------------
    channel_data = eeg.get_data().T
    y  = channel_data[:,:n_chs]
    if try_torch:
        print('Not implemented yet')
    elif try_cv:
        print('Trying CV')
        solver = Ridge() 
        num_folds = 5
        num_quantiles = 5
        param_grid = {'alpha': [5.0, 40.0, 50.0, 60.0, 80.0, 100.0, 
                                140.0, 160.0, 200.0, 250.0, 280.0, 320.0, 350.0,360,380,400,430 ]}# params Anthony [10.0, 40.0, 50.0, 60.0, 80.0, 100.0,140.0, 160.0, 200.0, 250.0, 280.0, 320.0, 350.0 ]
        # Create StratifiedKFold object
        kf = KFold(n_splits=num_folds)
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=solver, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf)
        grid_search.fit(X, y)
        
        # Extract results
        cv_results = grid_search.cv_results_
        alphas = param_grid['alpha']
        mean_test_scores = cv_results['mean_test_score']

        # Plot validation score for each lambda value
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, mean_test_scores, marker='o')
        plt.title('Validation Score vs. Lambda (alpha)')
        plt.xlabel('Lambda (alpha)')
        plt.ylabel('Negative Mean Squared Error')
        plt.grid(True)
        # plt.xscale('log')  # Logarithmic scale for better visualization
        # plt.show()
        solver = grid_search.best_estimator_
        plt.savefig(model_fig_path + f'average_validation_score_{subject_code}.png')
        unf=Unfolder(
            tmin, tmax, sr, feature_names=features, estimator=solver, scoring='r2',chans_to_ana=n_chs,second_delay=second_delay,tmin2=-.1,tmax2=.18
            )
        unf.coef_ = solver.coef_
        np.save(model_path+f'/cv_scores_{subject_code}.npy',mean_test_scores)

    else:
        start_time = time.time()
        unf.fit(X,y)
        end_time = time.time()
        # Calculating the time taken
        time_taken = end_time - start_time
        print("Time taken for fitting the model:", time_taken, "seconds")
    
    #--------save and plot data----------------------------------------------
    if isinstance(additive_feats,str): #only one additive feature
        additive_feats = [additive_feats]
    
    if second_delay is False:
        if interaction_feat is None and second_delay:      
            if isinstance(additive_feats,str):
                assert(unf.coef_.shape[1]/unf.delays_==2)
            if second_intercept_ev is not None:
                assert(unf.coef_.shape[1]/unf.delays_==len(additive_feats)+2)
            else:
                assert(unf.coef_.shape[1]/unf.delays_==len(additive_feats)+1)
        else:
            if isinstance(additive_feats,str):
                assert(unf.coef_.shape[1]/unf.delays_==3)
                interaction_feat = [interaction_feat]
            else:
                assert(unf.coef_.shape[1]/unf.delays_==len(additive_feats+[second_intercept_ev])+2)


    if use_splines:
        nfeats_to_plot = nfeats - 4
    else:
        nfeats_to_plot = nfeats

    for coef in range(nfeats_to_plot):
        coef_data_fname = f'rFRP_Subject_{suj.subject_id}_coeff_{coef}-ave.fif'
        coef_save_path  = model_save_path + f'coeff_{coef}/'
        coef_fig_save_path  = model_fig_path + f'coeff_{coef}/'

    
        #----- rFRP -----#
        if coef == 3 and second_delay is not None:
            data = unf.coef_[:,unf.delays_*coef:(unf.delays_*coef+unf.delays2_)]
            base_lims = [-.1, 0 ]
            times = np.linspace(-.1,.18,unf.delays2_)
        else:
            data = unf.coef_[:,unf.delays_*coef:unf.delays_*(coef+1)]
            base_lims = [-.2, 0 ]
            times = np.linspace(unf.tmin,unf.tmax,unf.delays_)
        # print()
        # Sample numpy array data (replace this with your own data)

        # Create an info dictionary to describe your data (replace with your own info)
        info = eeg.pick_channels(eeg.ch_names[:n_chs]).info

        # Create an Evoked object from the data and info
        rFRP_coef = mne.EvokedArray(data, info,tmin=times[0])
        print(f'epoked from data created for coefficient {coef}')
        # You can add additional information if needed, e.g., condition descriptions
        rFRP_coef.info['description'] = 'pyunfol rFRP'

        # You can also set the time points (in seconds) associated with your data
        rFRP_coef.apply_baseline((None,0))  
        print(f'baseline applied for coefficient {coef}')  
        rFRP_fig_name   = f'rFRP_suj_{suj.subject_id}_coeff_{coef}'
        figure = rFRP_coef.plot_joint(title= rFRP_fig_name,show=False)
        fig(figure,coef_fig_save_path,rFRP_fig_name)
        
        if save_data:
            # Save evoked data
            os.makedirs(coef_save_path, exist_ok=True)
            rFRP_coef.save( coef_save_path + coef_data_fname ,overwrite=True, verbose=None)
    if save_vif:
        intercept_vif_aves.append(compute_vif_intercept_ave(unf,X))
    if save_balance:
        categorical_balance_sujs.append(compute_categorical_balance(unf,X))
    if aic_bic:
        print(f'CALCULATING AIC FOR SUBJECT {subject_code}')
        aic_bic_sujs.append(ll_aic_bic_cp(y,unf.predict(X),X,alph))

    if save_vif:
        # Save evoked data
        intercept_vif_aves = np.array(intercept_vif_aves)
        np.save( model_path+'/intercept_vif_aves.npy',intercept_vif_aves)
    if save_balance:
        matrix_balance = np.array(categorical_balance_sujs)  # Assuming you want to concatenate along rows
        np.save( model_path+'/matrix_balance.npy',matrix_balance)
    if aic_bic:
        ll_aic_bic_cp_values = np.stack(aic_bic_sujs, axis=0)
        np.save(model_path+'/ll_aic_bic_cp.npy',ll_aic_bic_cp_values)
    print('ANALYSIS FINISHED')



if __name__ == "__main__":
    main()