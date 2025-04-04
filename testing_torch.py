
import pandas as pd
import mne
from pydeconv.utils import *
from pydeconv import *
from sklearn.model_selection import KFold, GridSearchCV
import torch
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Check for available device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Class to test GPU support
#==========================
# class Ridge:
#     def __init__(self, alpha=1.0, fit_intercept=True, batch_size=32, device='cpu'):
#         self.alpha = alpha
#         self.fit_intercept = fit_intercept
#         self.batch_size = batch_size
#         self.device = device
#         self.w = None
#         self.b = None
#         self.loss_history = []

#     def fit(self, X, y, epochs=1000, lr=0.01):
#         n_samples, n_features = X.shape
#         n_outputs = y.shape[1]  # Adjust for multiple outputs
#         self.w = torch.randn(n_features, n_outputs, requires_grad=True, device=self.device)
#         if self.fit_intercept:
#             self.b = torch.randn(1, n_outputs, requires_grad=True, device=self.device)

#         optimizer = optim.SGD([self.w] + ([self.b] if self.fit_intercept else []), lr=lr)

#         for epoch in range(epochs):
#             # Shuffle data indices at the beginning of each epoch
#             indices = torch.randperm(n_samples)
            
#             epoch_loss = 0.0
#             num_batches = (n_samples + self.batch_size - 1) // self.batch_size

#             for i in range(0, n_samples, self.batch_size):
#                 batch_indices = indices[i:i + self.batch_size]
                
#                 # Move mini-batches to the GPU in each iteration
#                 # X_batch = X[batch_indices].to(self.device)
#                 X_batch = torch.tensor(X[batch_indices].todense(), device=self.device, dtype=torch.float32)
#                 y_batch = torch.tensor(y[batch_indices], device=self.device, dtype=torch.float32)

#                 # Forward pass
#                 predictions = self.model(X_batch)
#                 loss = self.loss(predictions, y_batch)
#                 epoch_loss += loss.item()

#                 # Backward pass and optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             avg_loss = epoch_loss / num_batches
#             self.loss_history.append(avg_loss)
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#     def model(self, x):
#         return x @ self.w + (self.b if self.fit_intercept else 0)

#     def loss(self, predictions, y):
#         mse_loss = torch.mean((predictions - y) ** 2)
#         ridge_penalty = self.alpha * torch.sum(self.w ** 2)
#         return mse_loss + ridge_penalty

#     def predict(self, X):
#         return self.model(X).detach().cpu().numpy()

#     def plot_learning_curve(self):
#         plt.plot(self.loss_history, label="Training Loss")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         plt.title("Learning Curve")
#         plt.legend()
#         plt.show()
class Ridge:
    def __init__(self, input_dim, output_dim, alpha=1.0, fit_intercept=True, batch_size=32, device='cpu'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        self.device = device
        self.loss_history = []
        
        # Define the linear model
        self.linear = nn.Linear(input_dim, output_dim, bias=fit_intercept).to(device)

    def fit(self, X, y, epochs=1000, lr=0.01):
        n_samples = X.shape[0]
        
        # Convert to PyTorch tensors (handling sparse matrices correctly)
        X = torch.tensor(X.todense(), dtype=torch.float32, device=self.device) if hasattr(X, "todense") else torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # optimizer = optim.Adam(self.linear.parameters(), lr=lr)
        optimizer = optim.SGD(self.linear.parameters(), lr=lr, momentum=0.9, weight_decay=self.alpha)
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            num_batches = (n_samples + self.batch_size - 1) // self.batch_size

            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                predictions = self.model(X_batch)
                loss = self.loss(predictions, y_batch, n_samples)  # Normalize by samples
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            

    def model(self, x):
        return self.linear(x)

    def loss(self, predictions, y, n_samples):
        mse_loss = torch.mean((predictions - y) ** 2)
        # Only regularize weights, not bias
        ridge_penalty = (self.alpha / n_samples) * torch.sum(self.linear.weight ** 2)
        return mse_loss + ridge_penalty

    def predict(self, X):
        X = torch.tensor(X.todense(), dtype=torch.float32, device=self.device) if hasattr(X, "todense") else torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model(X).detach().cpu().numpy()

    def plot_learning_curve(self):
        plt.plot(self.loss_history, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()

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


# GPU FIT
#========


# PyTorch Ridge Model
# torch_ridge = Ridge(alpha=1e-3, fit_intercept=True, batch_size=1000, device=device)
# torch_ridge.fit(X_design, y_data, epochs=30, lr=0.01)
torch_ridge = Ridge(input_dim=X_design.shape[1], output_dim= y_data.shape[1], alpha= 40 , fit_intercept=False, batch_size=10000, device=device)
torch_ridge.fit(X_design, y_data, epochs=100, lr=0.0001)

# Plot learning curve
torch_ridge.plot_learning_curve()
coeffs =  torch_ridge.linear.weight.detach().cpu().numpy()
rERP_model.coef_ =  coeffs
# # Predictions should now have shape (13000, 3)
# torch_predictions = torch_ridge.predict(X_torch)
fig = rERP_model.plot_coefs()
plt.show()

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

