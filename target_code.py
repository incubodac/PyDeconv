
import pandas as pd
import mne
from pydeconv.utils import *
from pydeconv import *
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error


# Load parameter, data and features
#==================================
data_path = "./example_data/"
settings = analyze_data()
features = pd.read_csv(data_path + "629959_full_metadata.csv") 
raw     = mne.io.read_raw_eeglab(data_path + "629959_analysis.set", preload=True)

# Initialize the model
#=====================
rERP_model = PyDeconv(settings = settings , features = features, eeg = raw)
X_design = rERP_model.create_matrix()
y_data   = rERP_model.get_nonzero_data()

# Model Selection 
#================
solver = rERP_model.estimator
num_folds = 5
alphas = np.linspace(5, 300, 13)
param_grid = {'alpha': alphas.tolist()}
# Create StratifiedKFold object
kf = KFold(n_splits=num_folds)
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=solver, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf,verbose=5)
# grid_search.fit(X_design, y_data)
# rERP_model.estimator.set_params(alpha = 40)
rERP_model.fit(X_design, y_data)

# # Extract results
# #================
# cv_results = grid_search.cv_results_
# best_model = grid_search.best_estimator_
# rERP_model.coef_ = best_model.coef_

fig = rERP_model.plot_coefs()
plt.show()

# rERP_model.coef_.shape
# rERP_model.plot_coefs()

