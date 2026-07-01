import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class LatentSpaceNN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LatentSpaceNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.latent_layer = nn.Linear(64, latent_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)
        logits = self.classifier(latent)
        return logits, latent


class DeepSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    A minimal scikit-learn compatible wrapper around a small PyTorch network so
    it can be used in pipelines / cross_val_predict. Implements fit, predict,
    predict_proba and exposes `feature_importances_` (mean-abs input gradients)
    so existing ROI importance extraction can work.
    """

    def __init__(self, epochs=120, lr=0.001, weight_decay=1e-3, latent_dim=16, n_ensembles=1, seed=42, patience=15):
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.latent_dim = int(latent_dim)
        self.n_ensembles = int(n_ensembles)
        self.seed = int(seed)
        self.patience = int(patience)  # Added for early stopping

        # these will be set during fit
        self.model = None
        self.input_dim = None
        self.feature_importances_ = None
        self.device = None  # Hardware accelerator

    # scikit-learn compatibility helpers used by clone()
    def get_params(self, deep=True):
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "latent_dim": self.latent_dim,
            "n_ensembles": self.n_ensembles,
            "seed": self.seed,
            "patience": self.patience,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_model(self, input_dim):
        return LatentSpaceNN(input_dim, self.latent_dim)

    def fit(self, X, y):
        # X: np.ndarray, y: np.ndarray (binary 0/1)
        # Set device dynamically: MPS for Mac, CUDA for Nvidia, CPU as fallback
        self.device = "cpu"  # torch.device(
        # "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

        X_np = np.asarray(X)
        y_np = np.asarray(y).reshape(-1)

        # Determine the unique classes in the target vector
        # This is strictly required by scikit-learn for classifiers
        self.classes_ = np.unique(y_np)

        self.input_dim = X_np.shape[1]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build model and move to the selected hardware accelerator
        model = self._build_model(self.input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Move tensors to the accelerator
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1).to(self.device)

        model.train()

        # Early Stopping setup
        best_loss = float('inf')
        stagnant_epochs = 0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

            # Early stopping logic (Point 3)
            current_loss = loss.item()
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1

            if stagnant_epochs >= self.patience:
                # Stop training early to prevent overfitting and save time
                break

        # Compute mean-abs input gradients as feature importances
        model.eval()

        # FIX: Move to device FIRST, then ask for gradients so it acts as a leaf tensor
        X_explain = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        X_explain.requires_grad_(True)

        logits, _ = model(X_explain)
        logits.sum().backward()

        # Move gradients back to CPU for numpy compatibility
        grads = X_explain.grad.abs().mean(dim=0).cpu().detach().numpy()

        # Store model and feature importances
        self.model = model
        self.feature_importances_ = grads

        return self

    def predict_proba(self, X):
        X_np = np.asarray(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)  # Move to device
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            # Move results back to CPU
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs = np.vstack([1 - probs, probs]).T
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def transform_latent(self, X):
        """Return latent-layer embeddings for input X (numpy array)."""
        X_np = np.asarray(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)  # Move to device
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(X_tensor)
        # Move results back to CPU
        return latent.cpu().numpy()

