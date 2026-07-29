"""
Bearing anomaly detection training script
=========================================

This script trains a feature-based autoencoder for bearing anomaly detection.
It keeps only the useful parts from the notebook:

1. Load CWRU .mat vibration files
2. Split signals into windows
3. Extract 23 features per window
4. Train an autoencoder only on normal data
5. Choose an anomaly threshold from normal validation data
6. Evaluate normal vs abnormal detection
7. Save the model, scaler, threshold, and metadata

Expected dataset structure:

    data/cwru/
    ├── normal/
    │   ├── Normal_0.mat
    │   ├── Normal_1.mat
    │   ├── Normal_2.mat
    │   └── Normal_3.mat
    ├── inner/
    │   └── IR007_3.mat
    ├── ball/
    │   └── B007_3.mat
    └── outer/
        └── OR007_6_3.mat

Run:
    python bearing_anomaly_training.py --data-root data/cwru --output-dir models
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Configuration
# =============================================================================

FS = 12_000              # CWRU 12 kHz drive-end data
WINDOW_SIZE = 1024       # Must match the FFT size used later on Zynq
HOP_SIZE = 512           # 50% overlap for training data generation
SEED = 42
EPS = 1e-8

FEATURE_NAMES = [
    "mean",
    "std",
    "var",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "skewness",
    "kurtosis",
    "total_energy",
    "energy_0_500",
    "energy_500_1500",
    "energy_1500_3000",
    "energy_3000_6000",
    "ratio_0_500",
    "ratio_500_1500",
    "ratio_1500_3000",
    "ratio_3000_6000",
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "dominant_mag",
]


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data loading and preprocessing
# =============================================================================


def load_cwru_mat(path: Path) -> np.ndarray:
    """
    Load the drive-end vibration signal from a CWRU .mat file.

    CWRU files contain keys such as:
        X105_DE_time, X105_FE_time, X105_BA_time, X105RPM

    We use the key containing "DE_time" because it is the drive-end signal.
    """
    mat = sio.loadmat(path)

    signal_key = None
    for key in mat.keys():
        if "DE_time" in key:
            signal_key = key
            break

    if signal_key is None:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise ValueError(f"No DE_time key found in {path}. Available keys: {available}")

    return mat[signal_key].squeeze().astype(np.float32)



def make_windows(signal: np.ndarray,
                 window_size: int = WINDOW_SIZE,
                 hop_size: int = HOP_SIZE) -> np.ndarray:
    """Split a long 1D signal into overlapping fixed-size windows."""
    windows = []
    for start in range(0, len(signal) - window_size + 1, hop_size):
        windows.append(signal[start:start + window_size])

    return np.asarray(windows, dtype=np.float32)



def preprocess_window_for_fft(x_raw: np.ndarray) -> np.ndarray:
    """
    Preprocessing used only before FFT feature extraction.

    Steps:
        1. Remove mean / DC offset
        2. RMS normalize the centered signal
        3. Apply Hann window

    Note:
        Time-domain features are computed from the raw window.
        Frequency-domain features are computed from this preprocessed window.
    """
    x = x_raw.astype(np.float32)

    # 1. Mean removal
    x = x - np.mean(x)

    # 2. RMS normalization
    rms = np.sqrt(np.mean(x * x) + EPS)
    x = x / rms

    # 3. Hann window
    hann = np.hanning(len(x)).astype(np.float32)
    x = x * hann

    return x.astype(np.float32)


# =============================================================================
# Feature extraction
# =============================================================================


def extract_features_from_window(x_raw: np.ndarray,
                                 fs: int = FS) -> np.ndarray:
    """
    Extract the 23 features expected by the autoencoder.

    Feature order is exactly FEATURE_NAMES.
    Keep this order unchanged for deployment.
    """
    x_raw = x_raw.astype(np.float32)

    # -------------------------------------------------------------------------
    # 1. Time-domain features, computed from the raw window
    # -------------------------------------------------------------------------
    mean = np.mean(x_raw)
    std = np.std(x_raw)
    var = np.var(x_raw)
    rms = np.sqrt(np.mean(x_raw ** 2) + EPS)

    abs_x = np.abs(x_raw)
    peak = np.max(abs_x)
    peak_to_peak = np.max(x_raw) - np.min(x_raw)
    crest_factor = peak / (rms + EPS)

    centered = x_raw - mean
    skewness = np.mean(centered ** 3) / ((std ** 3) + EPS)
    kurtosis = np.mean(centered ** 4) / ((std ** 4) + EPS)

    # -------------------------------------------------------------------------
    # 2. Frequency-domain features, computed after FFT preprocessing
    # -------------------------------------------------------------------------
    x_pre = preprocess_window_for_fft(x_raw)

    # Python training uses real FFT, so it produces N/2 + 1 positive-frequency bins.
    # On Zynq, if the FFT IP outputs all N complex bins, use only bins 0..N/2.
    fft_vals = np.fft.rfft(x_pre)
    magnitude = np.abs(fft_vals).astype(np.float32)
    energy = (magnitude * magnitude).astype(np.float32)
    freqs = np.fft.rfftfreq(len(x_pre), d=1.0 / fs)

    total_energy = np.sum(energy) + EPS

    def band_energy(f_low: float, f_high: float) -> np.float32:
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(energy[mask]).astype(np.float32)

    energy_0_500 = band_energy(0, 500)
    energy_500_1500 = band_energy(500, 1500)
    energy_1500_3000 = band_energy(1500, 3000)
    energy_3000_6000 = band_energy(3000, 6000)

    dominant_idx = int(np.argmax(magnitude))
    dominant_freq = freqs[dominant_idx]
    dominant_mag = magnitude[dominant_idx]

    mag_sum = np.sum(magnitude) + EPS
    spectral_centroid = np.sum(freqs * magnitude) / mag_sum
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / mag_sum
    )
    spectral_flatness = (
        np.exp(np.mean(np.log(magnitude + EPS))) / (np.mean(magnitude) + EPS)
    )

    features = np.array([
        mean,
        std,
        var,
        rms,
        peak,
        peak_to_peak,
        crest_factor,
        skewness,
        kurtosis,
        total_energy,
        energy_0_500,
        energy_500_1500,
        energy_1500_3000,
        energy_3000_6000,
        energy_0_500 / total_energy,
        energy_500_1500 / total_energy,
        energy_1500_3000 / total_energy,
        energy_3000_6000 / total_energy,
        dominant_freq,
        spectral_centroid,
        spectral_bandwidth,
        spectral_flatness,
        dominant_mag,
    ], dtype=np.float32)

    return features



def extract_features_from_file(path: Path) -> np.ndarray:
    """Load one .mat file and return one feature vector per window."""
    signal = load_cwru_mat(path)
    windows = make_windows(signal)
    features = [extract_features_from_window(w) for w in windows]
    return np.asarray(features, dtype=np.float32)


# =============================================================================
# Dataset construction for anomaly detection
# =============================================================================


def verify_required_files(data_root: Path) -> None:
    """Check that all required training/evaluation files exist."""
    required_files = [
        data_root / "normal" / "Normal_0.mat",
        data_root / "normal" / "Normal_1.mat",
        data_root / "normal" / "Normal_2.mat",
        data_root / "normal" / "Normal_3.mat",
        data_root / "inner" / "IR007_3.mat",
        data_root / "ball" / "B007_3.mat",
        data_root / "outer" / "OR007_6_3.mat",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))



def build_autoencoder_datasets(data_root: Path):
    """
    Build train/validation/test sets for anomaly detection.

    Training:
        Normal_0, Normal_1, Normal_2 only.

    Validation threshold:
        First half of Normal_3.

    Final binary test:
        Second half of Normal_3 as normal.
        IR007_3, B007_3, OR007_6_3 as abnormal.
    """
    normal_train_files = [
        data_root / "normal" / "Normal_0.mat",
        data_root / "normal" / "Normal_1.mat",
        data_root / "normal" / "Normal_2.mat",
    ]
    normal_holdout_file = data_root / "normal" / "Normal_3.mat"
    fault_test_files = [
        (data_root / "inner" / "IR007_3.mat", "inner"),
        (data_root / "ball" / "B007_3.mat", "ball"),
        (data_root / "outer" / "OR007_6_3.mat", "outer"),
    ]

    print("\nBuilding normal training features")
    normal_train_features = []
    for path in normal_train_files:
        features = extract_features_from_file(path)
        normal_train_features.append(features)
        print(f"  {path.name:15s}: {features.shape[0]} windows")
    x_train_normal_raw = np.vstack(normal_train_features).astype(np.float32)

    print("\nBuilding normal validation/test features")
    normal_signal = load_cwru_mat(normal_holdout_file)
    normal_windows = make_windows(normal_signal)
    split_idx = len(normal_windows) // 2

    normal_val_windows = normal_windows[:split_idx]
    normal_test_windows = normal_windows[split_idx:]

    x_val_normal_raw = np.asarray(
        [extract_features_from_window(w) for w in normal_val_windows],
        dtype=np.float32,
    )
    x_test_normal_raw = np.asarray(
        [extract_features_from_window(w) for w in normal_test_windows],
        dtype=np.float32,
    )
    print(f"  {normal_holdout_file.name:15s}: {len(normal_windows)} total windows")
    print(f"  validation normal: {x_val_normal_raw.shape[0]} windows")
    print(f"  test normal      : {x_test_normal_raw.shape[0]} windows")

    print("\nBuilding abnormal test features")
    fault_features = []
    fault_names = []
    for path, fault_name in fault_test_files:
        features = extract_features_from_file(path)
        fault_features.append(features)
        fault_names.extend([fault_name] * len(features))
        print(f"  {path.name:15s}: {features.shape[0]} windows | {fault_name}")
    x_test_fault_raw = np.vstack(fault_features).astype(np.float32)

    # Binary test labels: 0 = normal, 1 = abnormal
    x_test_raw = np.vstack([x_test_normal_raw, x_test_fault_raw]).astype(np.float32)
    y_test_binary = np.concatenate([
        np.zeros(len(x_test_normal_raw), dtype=np.int64),
        np.ones(len(x_test_fault_raw), dtype=np.int64),
    ])
    test_label_text = ["normal"] * len(x_test_normal_raw) + fault_names

    # Feature scaling is fitted ONLY on normal training data.
    scaler = StandardScaler()
    x_train_normal = scaler.fit_transform(x_train_normal_raw).astype(np.float32)
    x_val_normal = scaler.transform(x_val_normal_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    print("\nDataset summary")
    print(f"  x_train_normal: {x_train_normal.shape}")
    print(f"  x_val_normal  : {x_val_normal.shape}")
    print(f"  x_test        : {x_test.shape}")
    print(f"  normal test   : {np.sum(y_test_binary == 0)}")
    print(f"  abnormal test : {np.sum(y_test_binary == 1)}")

    return x_train_normal, x_val_normal, x_test, y_test_binary, test_label_text, scaler


# =============================================================================
# PyTorch dataset and model
# =============================================================================


class AutoEncoderDataset(Dataset):
    """For an autoencoder, input and target are the same vector."""

    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        value = self.x[idx]
        return value, value


class BearingAutoEncoder(nn.Module):
    """Small dense autoencoder: 23 -> 16 -> 8 -> 16 -> 23."""

    def __init__(self, input_dim: int = 23):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# =============================================================================
# Training and evaluation
# =============================================================================


def train_autoencoder(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      device: torch.device,
                      epochs: int = 100,
                      lr: float = 1e-3,
                      patience: int = 15) -> dict:
    """
    Train the autoencoder with early stopping on normal validation loss.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, _ in train_loader:
            xb = xb.to(device)

            optimizer.zero_grad()
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                x_hat = model(xb)
                loss = criterion(x_hat, xb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history



def reconstruction_errors(model: nn.Module,
                          x: np.ndarray,
                          device: torch.device) -> np.ndarray:
    """Compute mean squared reconstruction error for each feature vector."""
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        x_hat = model(x_tensor)
        errors = torch.mean((x_tensor - x_hat) ** 2, dim=1)

    return errors.cpu().numpy()



def evaluate_anomaly_detector(model: nn.Module,
                              x_val_normal: np.ndarray,
                              x_test: np.ndarray,
                              y_test_binary: np.ndarray,
                              test_label_text: list[str],
                              device: torch.device):
    """
    Select threshold from normal validation errors and evaluate binary detection.
    """
    val_errors = reconstruction_errors(model, x_val_normal, device)
    test_errors = reconstruction_errors(model, x_test, device)

    # Final threshold used in the notebook: maximum normal validation error.
    threshold = float(np.max(val_errors))
    y_pred = (test_errors > threshold).astype(np.int64)

    print("\nValidation normal reconstruction errors")
    print(f"  min : {np.min(val_errors):.6f}")
    print(f"  mean: {np.mean(val_errors):.6f}")
    print(f"  max : {np.max(val_errors):.6f}")
    print(f"  threshold = max validation error = {threshold:.6f}")

    print("\nFinal binary classification report")
    print(classification_report(y_test_binary, y_pred, target_names=["normal", "abnormal"]))

    cm = confusion_matrix(y_test_binary, y_pred)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    print("\nReconstruction error by state")
    for label in sorted(set(test_label_text)):
        idx = np.asarray([name == label for name in test_label_text])
        errors = test_errors[idx]
        print(
            f"  {label:8s} | count: {len(errors):4d} | "
            f"min: {np.min(errors):12.6f} | "
            f"mean: {np.mean(errors):12.6f} | "
            f"max: {np.max(errors):12.6f}"
        )

    return threshold, val_errors, test_errors, y_pred


# =============================================================================
# Saving model package
# =============================================================================


def save_package(output_dir: Path,
                 model: nn.Module,
                 scaler: StandardScaler,
                 threshold: float,
                 args: argparse.Namespace) -> None:
    """Save everything needed for deployment/evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "bearing_autoencoder.pth")
    joblib.dump(scaler, output_dir / "feature_scaler.pkl")

    metadata = {
        "fs": FS,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_scaling": "StandardScaler fitted only on normal training data",
        "anomaly_threshold": threshold,
        "model_type": "feature_autoencoder",
        "architecture": "23-16-8-16-23",
        "epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved model package")
    print(f"  {output_dir / 'bearing_autoencoder.pth'}")
    print(f"  {output_dir / 'feature_scaler.pkl'}")
    print(f"  {output_dir / 'metadata.json'}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bearing anomaly autoencoder.")
    parser.add_argument("--data-root", type=Path, default=Path("data/cwru"),
                        help="Path to the CWRU dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Directory where model package will be saved.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")

    verify_required_files(args.data_root)

    x_train_normal, x_val_normal, x_test, y_test_binary, test_label_text, scaler = \
        build_autoencoder_datasets(args.data_root)

    train_loader = DataLoader(
        AutoEncoderDataset(x_train_normal),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        AutoEncoderDataset(x_val_normal),
        batch_size=256,
        shuffle=False,
    )

    model = BearingAutoEncoder(input_dim=len(FEATURE_NAMES)).to(device)
    print("\nModel")
    print(model)

    print("\nTraining autoencoder")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    threshold, _, _, _ = evaluate_anomaly_detector(
        model=model,
        x_val_normal=x_val_normal,
        x_test=x_test,
        y_test_binary=y_test_binary,
        test_label_text=test_label_text,
        device=device,
    )

    save_package(args.output_dir, model, scaler, threshold, args)


if __name__ == "__main__":
    main()