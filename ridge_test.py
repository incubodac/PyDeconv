import torch
from torch import nn
import torch.nn.functional as F

class Ridge:
    def __init__(self, alpha=0, fit_intercept=True, device=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
        # Check for device: MPS for Macs with Metal, CUDA for GPUs, else CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # Metal Performance Shaders (MPS) on macOS
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        # Move data to the specified device
        X = X.to(self.device).rename(None)
        y = y.to(self.device).rename(None).view(-1, 1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X
        rhs = X.T @ y
        
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0], device=self.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution
            
    def predict(self, X: torch.tensor) -> torch.tensor:
        # Move input to the specified device
        X = X.to(self.device).rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        return X @ self.w

if __name__ == "__main__":
    # Demo
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    X = torch.randn(100, 3).to(device)
    y = torch.randn(100, 1).to(device)  # Supports only single outputs

    model = Ridge(alpha=1e-3, fit_intercept=True, device=device)
    model.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
