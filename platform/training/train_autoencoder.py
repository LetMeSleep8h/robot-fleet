"""
Unsupervised autoencoder for FleetOS observation representation learning.

Trains an autoencoder on 86-dim telemetry observations to learn a
compressed 32-dim latent representation. The encoder can later serve
as a feature extractor for downstream RL policies.

Data source: observation vectors from S3 NDJSON experience files
(only the 'obs' field is used; actions and rewards are ignored).

Usage:
    python train_autoencoder.py \
        --job-id AE_001 \
        --experience-prefix experience/ \
        --epochs 200 \
        --s3-endpoint minio:9000

Outputs:
    - encoder.pt: Encoder-only weights (86 -> 32)
    - autoencoder.pt: Full autoencoder weights (86 -> 32 -> 86)
    - results.json: Training metrics and reconstruction loss
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class Encoder(nn.Module):
    """Compresses 86-dim observation to latent_dim."""

    def __init__(self, obs_dim: int = 86, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs 86-dim observation from latent_dim."""

    def __init__(self, latent_dim: int = 32, obs_dim: int = 86):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class ObservationAutoencoder(nn.Module):
    """Autoencoder: 86 -> 32 -> 86."""

    def __init__(self, obs_dim: int = 86, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim)
        self.decoder = Decoder(latent_dim, obs_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def load_observations_from_s3(args) -> np.ndarray:
    """Load observation vectors from S3 NDJSON experience files."""
    from minio import Minio

    client = Minio(
        args.s3_endpoint,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        secure=False,
    )

    prefix = args.experience_prefix
    objects = list(
        client.list_objects(args.s3_bucket, prefix=prefix, recursive=True)
    )
    print(f"[ae] Found {len(objects)} experience files under {prefix}")

    observations = []
    for obj in objects[:200]:  # limit files to avoid OOM
        try:
            response = client.get_object(args.s3_bucket, obj.object_name)
            for line in response.read().decode().strip().split("\n"):
                if not line:
                    continue
                t = json.loads(line)
                obs = t.get("obs")
                if obs and len(obs) == 86:
                    observations.append(obs)
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"[ae] Skip {obj.object_name}: {e}")

    print(f"[ae] Loaded {len(observations)} observations")
    return np.array(observations, dtype=np.float32)


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ae] Starting autoencoder training")
    print(f"[ae] Job ID: {args.job_id}")
    print(f"[ae] Latent dim: {args.latent_dim}")
    print(f"[ae] Epochs: {args.epochs}")

    # Load data
    obs_data = load_observations_from_s3(args)
    if len(obs_data) < 100:
        print(f"[ae] ERROR: Not enough observations ({len(obs_data)})")
        results = {
            "job_id": args.job_id,
            "status": "failed",
            "error_message": f"Insufficient observations: {len(obs_data)}",
        }
        with open(str(output_dir / "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Normalize
    obs_mean = obs_data.mean(axis=0)
    obs_std = obs_data.std(axis=0) + 1e-8
    obs_normalized = (obs_data - obs_mean) / obs_std

    # Train/val split (80/20)
    dataset = TensorDataset(torch.FloatTensor(obs_normalized))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model
    obs_dim = obs_data.shape[1]
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model = ObservationAutoencoder(obs_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    start_time = time.time()
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(output_dir / "autoencoder_best.pt"))
            torch.save(
                model.encoder.state_dict(), str(output_dir / "encoder_best.pt")
            )

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(
                f"[ae] Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}"
            )

    train_duration = time.time() - start_time

    # Save final model and normalization stats
    torch.save(model.state_dict(), str(output_dir / "autoencoder.pt"))
    torch.save(model.encoder.state_dict(), str(output_dir / "encoder.pt"))
    np.savez(
        str(output_dir / "obs_stats.npz"), mean=obs_mean, std=obs_std
    )

    # Compute per-dimension reconstruction error on validation set
    model.eval()
    all_errors = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            errors = (recon - batch).abs().cpu().numpy()
            all_errors.append(errors)
    all_errors = np.concatenate(all_errors, axis=0)
    per_dim_mae = all_errors.mean(axis=0).tolist()

    results = {
        "job_id": args.job_id,
        "status": "completed",
        "task_type": "autoencoder",
        "algorithm": "autoencoder",
        "artifact_url": f"training/{args.job_id}/autoencoder.pt",
        "encoder_url": f"training/{args.job_id}/encoder.pt",
        "train_duration_seconds": round(train_duration, 1),
        "num_observations": len(obs_data),
        "obs_dim": obs_dim,
        "latent_dim": args.latent_dim,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(train_losses[-1], 6),
        "final_val_loss": round(val_losses[-1], 6),
        "per_dim_mae_top5": sorted(
            enumerate(per_dim_mae), key=lambda x: -x[1]
        )[:5],
        "hyperparameters": {
            "algorithm": "autoencoder",
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "architecture": f"{obs_dim}->64->{args.latent_dim}->64->{obs_dim}",
        },
    }
    with open(str(output_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[ae] Training complete: val_loss={best_val_loss:.6f}, "
        f"duration={train_duration:.0f}s"
    )

    if args.s3_endpoint:
        _upload_to_s3(args, output_dir)

    if args.callback_url:
        _notify_callback(args.callback_url, results)

    return results


def _upload_to_s3(args, output_dir: Path):
    try:
        from minio import Minio

        client = Minio(
            args.s3_endpoint, access_key=args.s3_access_key,
            secret_key=args.s3_secret_key, secure=False,
        )
        if not client.bucket_exists(args.s3_bucket):
            client.make_bucket(args.s3_bucket)
        for f in output_dir.iterdir():
            if f.is_file():
                key = f"training/{args.job_id}/{f.name}"
                client.fput_object(args.s3_bucket, key, str(f))
                print(f"[s3] Uploaded {key}")
    except Exception as e:
        print(f"[s3] Upload failed (non-fatal): {e}")


def _notify_callback(url: str, results: dict):
    try:
        import urllib.request

        req = urllib.request.Request(
            url, data=json.dumps(results).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[callback] Notified {url}")
    except Exception as e:
        print(f"[callback] Failed (non-fatal): {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train observation autoencoder"
    )
    p.add_argument("--job-id", required=True)
    p.add_argument("--experience-prefix", required=True,
                   help="S3 prefix for NDJSON experience files")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="/tmp/training-output")
    p.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", ""))
    p.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", "fleetos-models"))
    p.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "fleetos"))
    p.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "fleetos123"))
    p.add_argument("--callback-url", default=os.getenv("TRAINING_CALLBACK_URL", ""))
    args = p.parse_args()

    train(args)
