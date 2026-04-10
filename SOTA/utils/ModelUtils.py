# SOTA/utils/ModelUtils.py
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ----------------------------------------------------------
#  Base Model Trainer (interface-like class)
# ----------------------------------------------------------
class BaseModelTrainer:
    """Abstract base trainer defining a common interface."""

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y):
        """Return a dictionary of evaluation metrics."""
        y_pred = self.predict(X)
        return dict(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average="macro", zero_division=0),
            recall=recall_score(y, y_pred, average="macro", zero_division=0),
            f1_score=f1_score(y, y_pred, average="macro", zero_division=0),
        )

    def save(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        joblib.dump(self, os.path.join(folder, f"{name}.joblib"))

    @staticmethod
    def load(path):
        return joblib.load(path)


# ----------------------------------------------------------
# Random Forest Trainer (inherits from BaseModelTrainer)
# ----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

class RFModelTrainer(BaseModelTrainer):
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ----------------------------------------------------------
# TF-IDF Wrapper that uses any model trainer
# ----------------------------------------------------------
def build_full_http_text(row, fields):
    """Combine selected HTTP fields into one string."""
    parts = []
    for f in fields:
        if f in row and pd.notna(row[f]) and str(row[f]).strip():
            parts.append(f"{f}:{str(row[f]).strip()}")
    return " ".join(parts)


class TFIDFTextEncoder:
    """Encodes selected text fields using TF-IDF."""
    def __init__(self, text_fields=None, max_features=2000):
        self.text_fields = text_fields or ["protocol", "http_method", "http_status", "http_uri", "http_user_agent", "http_referer", "http_body"]
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, df):
        df["http_fulltext"] = df.apply(lambda r: build_full_http_text(r, self.text_fields), axis=1)
        return self.vectorizer.fit_transform(df["http_fulltext"]).toarray()

    def transform(self, df):
        df["http_fulltext"] = df.apply(lambda r: build_full_http_text(r, self.text_fields), axis=1)
        return self.vectorizer.transform(df["http_fulltext"]).toarray()


# ----------------------------------------------------------
# Autoencoder Trainer (inherits from BaseModelTrainer)
# ----------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ---------- Trainer ----------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import sparse as sp

def _to_dense_numpy(X):
    """Accepts np.ndarray or scipy.sparse; returns dense np.ndarray (float32)."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)

class AutoencoderClassifierTrainer:
    def __init__(
        self,
        latent_dim=64,
        ae_lr=1e-3,
        ae_epochs=10,
        ae_batch_size=512,
        ae_patience=2,
        eval_batch_size=4096,
        clf_type="rf",
        clf_params=None,
        device=None,
        num_workers=0,
    ):
        self.latent_dim = latent_dim
        self.ae_lr = ae_lr
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.eval_batch_size = eval_batch_size
        self.ae_patience = ae_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers

        self.clf_type = clf_type
        self.clf_params = clf_params or {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
        
        try:
            if torch.cuda.is_available():
                torch.cuda.mem_get_info()  # raises if driver busy
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception:
            print("[Warning] GPUs busy or no memory — using CPU.")
            self.device = "cpu"

        print(f"[Device] Using {self.device.upper()} for AE training/evaluation.")

        self.autoencoder = None
        self.classifier = None

    # --- Train AE then classifier on latent ---
    def fit(self, X_train, y_train):
        X_train = _to_dense_numpy(X_train)
        input_dim = X_train.shape[1]

        # build AE
        self.autoencoder = Autoencoder(input_dim, self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.MSELoss()

        # dataloader
        ds = TensorDataset(torch.from_numpy(X_train))
        loader = DataLoader(
            ds, batch_size=self.ae_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=(self.device == "cuda")
        )

        use_cuda_amp = (self.device == "cuda")
        # scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)
        best_loss, no_improve = float("inf"), 0
        best_state = None

        # ---- AE training loop (with AMP on CUDA) ----
        for epoch in range(self.ae_epochs):
            self.autoencoder.train()
            total_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                # with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                with torch.amp.autocast('cuda', enabled=use_cuda_amp, dtype=torch.bfloat16):
                    x_hat, _ = self.autoencoder(xb)
                    loss = criterion(x_hat, xb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * xb.size(0)

            epoch_loss = total_loss / len(ds)
            print(f"[AE+CLF] AE Epoch {epoch+1}/{self.ae_epochs} | Loss={epoch_loss:.6f}")

            if epoch_loss < best_loss - 1e-6:
                best_loss, no_improve = epoch_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.autoencoder.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.ae_patience:
                    print(f"[AE+CLF] Early stopping at epoch {epoch+1}")
                    break

        # restore best AE
        if best_state is not None:
            self.autoencoder.load_state_dict(best_state)
        self.autoencoder.to(self.device).eval()

        # ---- Encode train to latent (batched, no-grad) ----
        Z_train = self.encode(X_train)

        # ---- Train classifier on latent (CPU sklearn) ----
        if self.clf_type == "rf":
            self.classifier = RandomForestClassifier(**self.clf_params)
        else:
            raise NotImplementedError(f"Classifier type {self.clf_type} not supported yet")

        self.classifier.fit(Z_train, y_train)

    # --- Encode to latent (batched) ---
    def encode(self, X):
        X = _to_dense_numpy(X)
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            ds, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=(self.device == "cuda")
        )
        zs = []
        self.autoencoder.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                _, z = self.autoencoder(xb)
                zs.append(z.detach().cpu())
        Z = torch.cat(zs, dim=0).numpy()
        return Z

    # --- Compute reconstruction errors (batched) ---
    def recon_errors(self, X):
        X = _to_dense_numpy(X)
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            ds, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=(self.device == "cuda")
        )
        errs = []
        self.autoencoder.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                x_hat, _ = self.autoencoder(xb)
                e = torch.mean((x_hat - xb) ** 2, dim=1)
                errs.append(e.detach().cpu())
        return torch.cat(errs, dim=0).numpy()

    # --- Evaluate AE + classifier ---
    def evaluate(self, X_test, y_true):
        X_test = _to_dense_numpy(X_test)
        # recon errors
        recon_error = self.recon_errors(X_test)
        # latent + classify
        Z_test = self.encode(X_test)
        y_pred = self.classifier.predict(Z_test)

        metrics = {
            "recon_mse_mean": float(np.mean(recon_error)),
            "recon_mse_std": float(np.std(recon_error)),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        return metrics, recon_error


# ---------- CAE Model ----------
class ContrastiveModelTrainer:
    """
    Trainer implementing CAE loss = MSE(x, x̂) + λ * Contrastive(z_i, z_j)
    Contrastive term pulls same-label pairs together, pushes different labels apart.
    """
    def __init__(
        self,
        latent_dim=64,
        cae_lr=1e-3,
        cae_epochs=10,
        cae_batch_size=512,
        cae_patience=2,
        lambda_contrast=0.1,
        margin=10.0,
        clf_type="rf",
        clf_params=None,
        device=None,
        num_workers=0,
    ):
        self.latent_dim = latent_dim
        self.cae_lr = cae_lr
        self.cae_epochs = cae_epochs
        self.cae_batch_size = cae_batch_size
        self.cae_patience = cae_patience
        self.lambda_contrast = lambda_contrast
        self.margin = margin
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers

        self.clf_type = clf_type
        self.clf_params = clf_params or {"n_estimators": 100, "random_state": 42, "n_jobs": -1}

        self.model = None
        self.classifier = None
        print(f"[Device] Using {self.device.upper()} for CAE training.")

    # --- Contrastive loss (pairwise within batch) ---
    def _contrastive_loss(self, z, y):
        dist = torch.cdist(z, z, p=2)                # (B,B)
        same = (y.unsqueeze(1) == y.unsqueeze(0)).float()
        pos = dist * same                            # same-class distances
        neg = torch.relu(self.margin - dist) * (1 - same)
        return pos.mean() + neg.mean()

    # --- Fit contrastive AE + classifier ---
    def fit(self, X_train, y_train):
        X_train = _to_dense_numpy(X_train)
        y_train = np.asarray(y_train)

        if 'Autoencoder' not in globals() and 'Autoencoder' not in locals():
            print("[Warning] 'Autoencoder' class not found. Make sure it is defined.")
            
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        input_dim = X_train.shape[1]
        self.model = Autoencoder(input_dim, self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.cae_lr)
        mse_loss = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_encoded))
        loader = DataLoader(ds, batch_size=self.cae_batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=(self.device == "cuda"))

        use_amp = (self.device == "cuda")
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # --- ADDED: Early stopping variables ---
        best_loss = float("inf")
        no_improve = 0
        best_state = None
    

        for epoch in range(self.cae_epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    x_hat, z = self.model(xb)
                    loss_recon = mse_loss(x_hat, xb)
                    loss_contr = self._contrastive_loss(z, yb)
                    loss = loss_recon + self.lambda_contrast * loss_contr

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * xb.size(0)
            
            epoch_loss = total_loss / len(ds)
            print(f"[CAE] Epoch {epoch+1}/{self.cae_epochs} | Loss={epoch_loss:.6f}")
            if epoch_loss < best_loss - 1e-6:
                best_loss, no_improve = epoch_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.cae_patience:
                    print(f"[CAE] Early stopping at epoch {epoch+1}")
                    break
        if best_state is not None:
            print(f"[CAE] Restoring best model (Loss={best_loss:.6f})")
            self.model.load_state_dict(best_state)

        self.model.to(self.device).eval()

        # Encode training set → latent
        Z_train = self.encode(X_train)

        # Train classifier on latent space (same interface as AE trainer)
        if self.clf_type == "rf":
            self.classifier = RandomForestClassifier(**self.clf_params)
            self.classifier.fit(Z_train, y_train)
        else:
            raise NotImplementedError(f"Classifier {self.clf_type} not supported.")

    # --- Encode (latent extraction) ---
    def encode(self, X):
        X = _to_dense_numpy(X)
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=4096, shuffle=False,
                            num_workers=self.num_workers, pin_memory=(self.device == "cuda"))
        zs = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                _, z = self.model(xb)
                zs.append(z.cpu())
        return torch.cat(zs, dim=0).numpy()

    # --- Reconstruction errors ---
    def recon_errors(self, X):
        X = _to_dense_numpy(X)
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=4096, shuffle=False,
                            num_workers=self.num_workers, pin_memory=(self.device == "cuda"))
        errs = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                x_hat, _ = self.model(xb)
                e = torch.mean((x_hat - xb) ** 2, dim=1)
                errs.append(e.cpu())
        return torch.cat(errs, dim=0).numpy()

    # --- Evaluate (recon + classifier metrics) ---
    def evaluate(self, X_test, y_true):
        X_test = _to_dense_numpy(X_test)
        recon_error = self.recon_errors(X_test)
        Z_test = self.encode(X_test)
        y_pred = self.classifier.predict(Z_test)

        metrics = {
            "recon_mse_mean": float(np.mean(recon_error)),
            "recon_mse_std": float(np.std(recon_error)),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        return metrics, recon_error