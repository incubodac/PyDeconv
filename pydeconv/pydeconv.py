from sklearn.base import BaseEstimator, is_regressor 
from sklearn.linear_model import LinearRegression, Ridge, LassoLarsIC
# from utils.winrej import cont_ArtifactDetect 
import numpy as np
from scipy.sparse import hstack 
import numbers
from .utils.pydeconv_functions import add_spline_features
from .utils.plot_general import plot_model_results


class PyDeconv(BaseEstimator):
    def __init__(self,
        settings,
        features,
        eeg
    ):  
        self.model_name = settings['model_name']
        self.fs_inter_ev = settings['first_intercept_event_type']
        self.sd_inter_ev = settings['second_intercept_event_type']
        self.data = eeg
        self.features = features
        self.sfreq = float(eeg.info['sfreq'])
        self.tmin = settings['tmin']
        self.tmax = settings['tmax']
        self.fit_intercept = None
        self.scoring = "r2"
        self.n_jobs = None
        self.delays_ = len(_delays(self.tmin,self.tmax,self.sfreq))
        self.second_delay = settings['second_delay']
        if self.second_delay is not None:
            self.delays2_ = len(_delays(tmin2,tmax2,sfreq))
        else:
            self.delays2_ = None
        self.verbose = None
        self.chans_to_ana = settings['eeg_chns']
        self.tmin2 = None
        self.tmax2 = None
        self.intercept = settings['parsed_formula']['intercept']
        self.interactions = settings['parsed_formula']['interactions']
        self.additive_features = settings['parsed_formula']['additive_features']
        self.use_splines = settings['use_splines']
        if settings['solver'] == 'ridge':
            self.estimator = Ridge()
        self.estimator = 0.0 if settings['solver'] is None else self.estimator
       # Print the class attributes and model description
        self._print_model_info()
                
    def _print_model_info(self):
        print("\n" + "="*40)
        print(f"Model Name: {self.model_name}")
        print(f"First Intercept Event Type: {self.fs_inter_ev}")
        print(f"Second Intercept Event Type: {self.sd_inter_ev}")
        print(f"Sampling Frequency: {self.sfreq}")
        print(f"Time Window: {self.tmin} to {self.tmax}")
        print(f"Channels to Analyze: {self.chans_to_ana}")
        print("\nModel Description:")
        print(f"Intercept: {self.intercept}")
        print(f"Additive Features: {self.additive_features}")
        print(f"Interactions: {self.interactions}")
        print("="*40 + "\n")


    def __repr__(self):  
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        estimator = self.estimator
        if not isinstance(estimator, str):
            estimator = type(self.estimator)
        s += "estimator : %s, " % (estimator,)
        if hasattr(self, "coef_"):
            if self.feature_names is not None:
                feats = self.feature_names
                if isinstance(feats,str):
                    s += "feature: %s, " % feats
                else:
                    s += "features : [%s, ..., %s], " % (feats[0], feats[-1])
            s += "fit: True"
        else:
            s += "fit: False"
        if hasattr(self, "scores_"):
            s += "scored (%s)" % self.scoring
        return "<PyDeconv | %s>" % s
        
        
    
    def fit(self, X, y):
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_features)
            The input features for the model.
        y : array, shape (n_times[, n_epochs][, n_outputs])
            The output features for the model.

        Returns
        -------
        self : instance
            The instance so you can chain operations.
        """
        from scipy import linalg

        if self.scoring not in _SCORERS.keys():
            raise ValueError(
                "scoring must be one of %s, got"
                "%s " % (sorted(_SCORERS.keys()), self.scoring)
            )
        from sklearn.base import clone

        X, y, _, self._y_dim = self._check_dimensions(X, y)

        if self.tmin > self.tmax:
            raise ValueError(
                "tmin (%s) must be at most tmax (%s)" % (self.tmin, self.tmax)
            )
        # Initialize delays
        #self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # Define the slice that we should use in the middle
        #self.valid_samples_ = _delays_to_slice(self.delays_)


        if is_regressor(self.estimator):
            estimator = clone(self.estimator)
            if (
                self.fit_intercept is not None
                and estimator.fit_intercept != self.fit_intercept
            ):
                raise ValueError(
                    "Estimator fit_intercept (%s) != initialization "
                    "fit_intercept (%s), initialize ReceptiveField with the "
                    "same fit_intercept value or use fit_intercept=None"
                    % (estimator.fit_intercept, self.fit_intercept)
                )
            self.fit_intercept = estimator.fit_intercept
        else:
            raise ValueError(
                "`estimator` must be a float or an instance"
                " of `BaseEstimator`,"
                " got type %s." % type(self.estimator)
            )
        self.estimator_ = estimator
        del estimator
        #_check_estimator(self.estimator_)

        # Create input features
        n_times, n_feats = X.shape
        n_outputs = y.shape[-1]
        n_delays = self.delays_
        n_delays2 = self.delays2_

        chns = self.chans_to_ana
        if self.feature_names==None:
            n_predictors = 0
        else:
            if type(self.feature_names) is list:
                n_predictors =  len(self.feature_names)
                print('features names len',n_predictors)
            elif type(self.feature_names) is str:
                n_predictors =  1

        if n_delays2 is None:
            print(f'features times delays {(n_predictors+1)*n_delays } \n  n_feats {n_feats}')
            if (self.feature_names is not None) and (( n_predictors+1)*n_delays != n_feats):
                raise ValueError(
                    "n_features in X does not match feature names "
                    "(%s != %s)" % (n_feats, 42) 
                )
        else:
            print(f'features times delays {(n_predictors)*n_delays+ n_delays2 } \n  n_feats {n_feats}')

        # Update feature names if we have none
        #print('n pred +1',(n_predictors+1)*n_delays ,'n feats',n_feats)
            if (self.feature_names is not None) and (( n_predictors*n_delays+ n_delays2) != n_feats):
                raise ValueError(
                    "n_features in X does not match feature names "
                    "(%s != %s)" % (n_feats, 42) 
                )

        # Create input features
        #X, y = self._delay_and_reshape(X, y)
        if n_delays2 is None:

            self.estimator_.fit(X, y)
            coef = get_coef(self.estimator_, "coef_")  # (n_targets, n_features)
            shape = [chns, n_delays*(n_predictors+1)]#change the harcoded chans number later.
        
            self.coef_ = coef.reshape(shape)

            coef = np.reshape(self.coef_, (n_feats , n_outputs))
        else:

            self.estimator_.fit(X, y)
            coef = get_coef(self.estimator_, "coef_")  # (n_targets, n_features)
            shape = [chns, n_delays*(n_predictors)+n_delays2]#change the harcoded chans number later.
        
            self.coef_ = coef.reshape(shape)

            coef = np.reshape(self.coef_, (n_feats , n_outputs))
  

        return self
    
    def predict(self, X):
        # Implement the prediction logic for your estimator
        # X: array-like, shape (n_samples, n_features)
        # Your prediction code here
        return self.estimator_.predict(X)

    
    def score(self, X, y):
        """Score predictions.

        This calls ``self.predict``, then masks the output of this
        and ``y` with ``self.valid_samples_``. Finally, it passes
        this to a :mod:`sklearn.metrics` scorer.

        Parameters
        ----------
        X : array, shape (n_times, n_channels)
            The input features for the model.
        y : array, shape (n_times, n_outputs])
            Used for scikit-learn compatibility.

        Returns
        -------
        scores : list of float, shape (n_outputs,)
            The scores estimated by the model for each output (e.g. mean
            R2 of ``predict(X)``).
        """
        # Create our scoring object
        scorer_ = _SCORERS[self.scoring]

        # Generate predictions, then reshape so we can mask time
        X, y = self._check_dimensions(X, y, predict=True)[:2]
        n_times, n_epochs, n_outputs = y.shape
        y_pred = self.predict(X)
        y_pred = y_pred[self.valid_samples_]
        y = y[self.valid_samples_]

        # Re-vectorize and call scorer
        y = y.reshape([-1, n_outputs], order="F")
        y_pred = y_pred.reshape([-1, n_outputs], order="F")
        assert y.shape == y_pred.shape
        scores = scorer_(y, y_pred, multioutput="raw_values")
        return scores
    
    def _check_dimensions(self, X, y, predict=False):
        X_dim = X.ndim
        y_dim = y.ndim if y is not None else 0
        if X_dim != 2:           
            raise ValueError(
                "X must be shape (n_times, features*time delays+1]"
            )
        if y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "X and y do not have the same n_times\n"
                    "%s != %s" % (X.shape[0], y.shape[0])
                )
            if predict and y.shape[-1] != len(self.estimator_.coef_):
                raise ValueError(
                    "Number of outputs does not match"
                    " estimator coefficients dimensions"
                )
        return X, y, X_dim, y_dim
    
    def get_filtered_data(self,  thr = 300 ):
        "Not implemented"
        return self.data

    def get_data(self):
        n_chs = self.chans_to_ana
        channel_data = self.data.get_data().T
        y  = channel_data[:,:n_chs]
        return y
    def get_nonzero_data(self):
        y = self.get_data()
        return y[self.non_zero_rows]
    def create_matrix(self):
        X = create_design_matrix(self.data,
                                 self.tmin,
                                 self.tmax,
                                 self.sfreq, 
                                 self.features, 
                                 self.fs_inter_ev, 
                                 self.additive_features,
                                 self.interactions)
        if isinstance(self.use_splines,int):
            print('WARNING: B-SPLINES ARE BEING APPLIED')
            second_intercept_events_metadata, second_intercept_features = add_spline_features(self.features)
        else:
            second_intercept_features = None

        #for second delay use
        if self.second_delay is True:
            X_sacc = create_design_matrix(self.data,
                                          tmin,
                                          .3,
                                          self.sfreq,
                                          second_intercept_events_metadata, 
                                          self.sd_inter_ev, 
                                          second_intercept_features,
                                          interaction=None)
        if self.second_delay is None:
            X_sacc = create_design_matrix(self.data,
                                          self.tmin,
                                          self.tmax,
                                          self.sfreq,
                                          second_intercept_events_metadata, 
                                          self.sd_inter_ev, 
                                          second_intercept_features,
                                          interaction=None)
        concatenated_matrix = hstack([X, X_sacc])
        # Print the size of the original matrix and data
        print("\n" + "="*40)
        print("Original Design Matrix Shape:")
        print(f"X_design shape: {concatenated_matrix.shape}")
        print(f"y_data shape: {self.get_data().shape}")
        print("="*40 + "\n")
        # Identify rows with all zero values
        non_zero_rows = concatenated_matrix.getnnz(axis=1) > 0
        self.non_zero_rows = non_zero_rows
        return concatenated_matrix[non_zero_rows]

    def plot_coefs(self):
        list_of_coeffs = []
        plot_model_results(self,list_of_coeffs, figsize=[10,5],top_topos=True)

    
def _times_to_samples(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to samples."""
    # Convert seconds to samples
    smp_ix = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))
    return smp_ix




# def _times_to_delays(tmin, tmax, sfreq):
#     """Convert a tmin/tmax in seconds to delays."""
#     # Convert seconds to samples
#     delays = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))
#     return delays


# def _delays_to_slice(delays):
#     """Find the slice to be taken in order to remove missing values."""
#     # Negative values == cut off rows at the end
#     min_delay = None if delays[-1] <= 0 else delays[-1]
#     # Positive values == cut off rows at the end
#     max_delay = None if delays[0] >= 0 else delays[0]
#     return slice(min_delay, max_delay)
def closest_indices(arr1, arr2):
    closest_indices = np.empty_like(arr2, dtype=np.intp)

    # Iterate over the values in arr2.
    for i, val in np.ndenumerate(arr2):
        # Find the index of the closest value in arr1.
        closest_index = np.abs(arr1 - val).argmin()
        closest_indices[i] = closest_index
    return closest_indices


def create_design_matrix(raw,tmin,tmax,sr,events,intercept_evt, feature_cols,interaction=None):
    #building design matrix from scratch
    from scipy.sparse import csr_matrix
    #modify this to yield 0 pred for empty 1 for str and num of elements for a list
    if feature_cols==None:
        n_predictors = 0
        feature_cols = []
    else:
        if type(feature_cols) is list:
            n_predictors =  len(feature_cols)
        elif type(feature_cols) is str:
            n_predictors =  1
            feature_cols = [feature_cols]
          
    delays = _delays(tmin,tmax,sr)
    n_samples_window = len(delays)
    #timelimits = [-.2,.4] #307  samples per predictor * 4
    zero_idx=closest_indices(delays,0)
    # Create a copy of the filtered DataFrame to avoid SettingWithCopyWarning
    evt_to_model = events[events['type'] == intercept_evt].copy()

    # Set the 'type' column values to 1 using column indexing to avoid FutureWarning
    evt_to_model[evt_to_model.columns[evt_to_model.columns.get_loc('type')]] = 1
    
    if interaction is not None:
        inter_feats = interaction[0].split(':')
        beta_int0 = inter_feats[0]
        beta_int1 = inter_feats[1]
        interaction_col = 'interaction' 
        feature_cols = feature_cols + [interaction_col]
        n_predictors = n_predictors + 1
        evt_to_model[interaction_col] = evt_to_model[beta_int0]*evt_to_model[beta_int1]
        #seguir aqui!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    expanded_params = (1+n_predictors)*n_samples_window
    signal_longitud_in_samples = raw.n_times

    X = np.zeros([signal_longitud_in_samples, expanded_params])

    count_events = len(evt_to_model)
    evt_lat = evt_to_model['latency']
    evt_lat = evt_lat.values.astype(int)
    features = ['type'] + feature_cols



    for beta in range(n_predictors+1):
        
        j_idx = np.arange(beta*n_samples_window,beta*n_samples_window+n_samples_window)   
        for j in j_idx:   
            for i in range(count_events):
                X[evt_lat[i]+j-beta*n_samples_window-zero_idx,j] = evt_to_model[features[beta]].values[i]
    



    X_sparse = csr_matrix(X)
    return X_sparse

def _delays(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to delays."""
    # Convert seconds to samples
    delays = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))
    return delays



def _r2_score(y_true, y, multioutput=None):
    from sklearn.metrics import r2_score

    return r2_score(y_true, y, multioutput=multioutput)


_SCORERS = {"r2": _r2_score}#, "corrcoef": _corr_score}

if __name__=='__main__':
    from sklearn.linear_model import LinearRegression
    # Using os.path
    

    info = exp_info()
    su = 2
    suj = load.subject(info,su)
    eeg = suj.load_analysis_eeg()
    evts = suj.load_metadata()
    from unfoldpy import Unfolder, create_design_matrix
    #----------parameters-------------
    from sklearn.linear_model import LinearRegression
    feature_cols = []#'mss'
    intercept_evt   = 'fixation'
    tmin ,tmax = -.2 , .4
    sr = 500
    unf=Unfolder(
            tmin, tmax, sr, feature_cols, estimator=LinearRegression(),scoring='r2'
    )
    print(unf)

    X = create_design_matrix(eeg,tmin,tmax,evts,intercept_evt, feature_cols,sr)
    print(X)