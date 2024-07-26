import config
import pydeconv.PyDeconv
import pydeconv.pydeconv_functions
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LassoLarsIC
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error


settings = analyze_data()
settings
# Initialize the model
ERPdeconv= PyDeconv(settings = settings)

X_design = ERPdeconv.create_matrix()
y_data   = ERPdeconv.data()


# num_folds = 5
# param_grid = {'alpha': np.linspace(5, 500, 17)}
# # Create StratifiedKFold object
# kf = KFold(n_splits=num_folds)

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator = ERPdeconv, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf)
# grid_search.fit(X_design_, y_data)

# # Extract results
# cv_results = grid_search.cv_results_
# alphas = param_grid['alpha']
# mean_test_scores = cv_results['mean_test_score']

# solver = grid_search.best_estimator_

# plt.savefig(model_fig_path + f'average_validation_score_{subject_code}.png')


# bestERPdeconv = PyDeconv(settings = settings,solver)
# bestERPs.coef_ = bestERPdeconv.coef_
