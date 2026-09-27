from __future__ import annotations

import torch
import torch.nn as nn
from model.tcn_ae import TCNAE, TCNKPIRegressor, DescriptorInjectedResidualTCNAE
from typing import List, Dict, Optional


def train_descriptor_injected_residual(
    model: DescriptorInjectedResidualTCNAE,
    Y: torch.Tensor,
    descriptors_standardized: torch.Tensor,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: Optional[int] = None,
    lam_cross: float = 0.05,
    lam_residual: float = 1e-4,
):
    """Train reconstruction while keeping descriptor coordinates exact.

    ``descriptors_standardized`` is never predicted by the network.  It is
    concatenated with encoder-produced residual coordinates before decoding.
    A cross-correlation penalty discourages residual coordinates from simply
    duplicating the prescribed descriptors, and a small residual-energy
    penalty prevents unbounded latent scaling.
    """
    model.train()
    if Y.dim() == 2:
        Y = Y.unsqueeze(1)
    if descriptors_standardized.dim() != 2:
        raise ValueError("descriptors_standardized must have shape (N, K).")
    if descriptors_standardized.shape[0] != Y.shape[0]:
        raise ValueError("Y and descriptors_standardized must share samples.")
    if descriptors_standardized.shape[1] != model.descriptor_dim:
        raise ValueError("Descriptor dimension does not match the model.")
    descriptors_standardized = descriptors_standardized.to(Y.device)

    n_total = int(Y.shape[0])
    batch_size_eff = (
        n_total
        if batch_size is None or batch_size <= 0 or batch_size >= n_total
        else int(batch_size)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    mse = nn.MSELoss()

    for epoch in range(int(max(0, epochs))):
        permutation = torch.randperm(n_total, device=Y.device)
        loss_sum = 0.0
        recon_sum = 0.0
        cross_sum = 0.0
        count = 0
        for start in range(0, n_total, batch_size_eff):
            indices = permutation[start:start + batch_size_eff]
            response_batch = Y[indices]
            descriptor_batch = descriptors_standardized[indices]
            latent, reconstructed = model(response_batch, descriptor_batch)
            target = response_batch.squeeze(1) if reconstructed.dim() == 2 else response_batch
            loss_reconstruction = mse(reconstructed, target)

            residual = latent[:, model.descriptor_dim:]
            if residual.shape[1] > 0 and residual.shape[0] > 1:
                semantic_centered = descriptor_batch - descriptor_batch.mean(dim=0)
                residual_centered = residual - residual.mean(dim=0)
                semantic_norm = semantic_centered / (semantic_centered.std(dim=0) + 1e-8)
                residual_norm = residual_centered / (residual_centered.std(dim=0) + 1e-8)
                cross = semantic_norm.T @ residual_norm / max(int(residual.shape[0]) - 1, 1)
                loss_cross = torch.mean(cross ** 2)
                loss_residual = torch.mean(residual ** 2)
            else:
                loss_cross = latent.new_tensor(0.0)
                loss_residual = latent.new_tensor(0.0)

            loss = (
                loss_reconstruction
                + float(lam_cross) * loss_cross
                + float(lam_residual) * loss_residual
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            recon_sum += float(loss_reconstruction.detach().cpu())
            cross_sum += float(loss_cross.detach().cpu())
            count += 1

        if (epoch + 1) % 10 == 0 or (epoch + 1) == int(max(0, epochs)):
            print(
                f"[descriptor-injected] Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss_sum/max(count,1):.4f} | "
                f"Recon: {recon_sum/max(count,1):.4f} | "
                f"Cross: {cross_sum/max(count,1):.4f}"
            )
    return model

def train_with_physics(
    model: TCNAE,
    Y: torch.Tensor,
    p_true: Dict[str, torch.Tensor],
    kpi_true: Optional[torch.Tensor] = None,
    epochs: int = 500,
    lr: float = 1e-3,
    lam_phy: float = 1.0,
    lam_ortho: float = 0.1,
    lam_kpi: float = 0.0,
    batch_size: Optional[int] = None,
    semantic_loss_mode: str = "zscore_mse",
    semantic_loss_alpha: float = 0.5,
    semantic_dim: Optional[int] = None,
    lam_cross: float = 0.0,
    pretrain_epochs: int = 0,
):
    """
    Train the Phy-TCN-AE with Scalar Physics Loss and Orthogonality Loss.
    (No Physical Projection during training as per updated requirement).
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    if Y.dim() == 2:
        Y = Y.unsqueeze(1)

    if kpi_true is not None:
        if kpi_true.dim() == 1:
            kpi_true = kpi_true.unsqueeze(1)
        if kpi_true.device != Y.device:
            kpi_true = kpi_true.to(Y.device)
        
    N_total = Y.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= N_total:
        batch_size_eff = N_total
    else:
        batch_size_eff = int(batch_size)

    feature_names = list(p_true.keys())
    if semantic_dim is None:
        semantic_dim_eff = min(len(feature_names), int(getattr(model, "num_single_field", len(feature_names))))
    else:
        semantic_dim_eff = int(max(0, semantic_dim))

    def _semantic_alignment_loss(z_col: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        z_std = (z_col - torch.mean(z_col)) / (torch.std(z_col) + 1e-8)
        t_std = (target_raw - torch.mean(target_raw)) / (torch.std(target_raw) + 1e-8)
        mode = str(semantic_loss_mode or "zscore_mse").strip().lower()
        mse_term = mse_loss(z_std, t_std)
        if mode == "zscore_mse":
            return mse_term
        corr = torch.mean(z_std * t_std)
        corr_term = 1.0 - torch.clamp(corr, -1.0, 1.0)
        if mode == "corr":
            return corr_term
        if mode == "hybrid":
            alpha = float(min(max(semantic_loss_alpha, 0.0), 1.0))
            return alpha * mse_term + (1.0 - alpha) * corr_term
        return mse_term

    def _run_stage(
        stage_epochs: int,
        stage_name: str,
        stage_lam_phy: float,
        stage_lam_ortho: float,
        stage_lam_kpi: float,
        stage_lam_cross: float,
    ) -> None:
        for epoch in range(stage_epochs):
            perm = torch.randperm(N_total, device=Y.device)
            recon_sum = 0.0
            phy_sum = 0.0
            ortho_sum = 0.0
            kpi_sum = 0.0
            count = 0

            for start in range(0, N_total, batch_size_eff):
                idx = perm[start:start + batch_size_eff]
                Y_b = Y[idx]
                z, y_hat = model(Y_b)

                if y_hat.dim() != Y_b.dim():
                    if y_hat.dim() == 2 and Y_b.dim() == 3:
                        Y_loss = Y_b.squeeze(1)
                    elif y_hat.dim() == 3 and Y_b.dim() == 2:
                        Y_loss = Y_b.unsqueeze(1)
                    else:
                        Y_loss = Y_b
                else:
                    Y_loss = Y_b

                loss_recon = mse_loss(y_hat, Y_loss)

                loss_phy = torch.tensor(0.0, device=z.device)
                if stage_lam_phy != 0.0 and len(feature_names) > 0:
                    for i, name in enumerate(feature_names):
                        if i < z.shape[1] and i < semantic_dim_eff:
                            target_raw = p_true[name][idx]
                            if target_raw.dim() == 2 and target_raw.shape[1] == 1:
                                target_raw = target_raw.squeeze(1)
                            loss_phy = loss_phy + _semantic_alignment_loss(z[:, i], target_raw)

                loss_kpi = torch.tensor(0.0, device=z.device)
                if stage_lam_kpi != 0.0 and kpi_true is not None:
                    kpi_hat = model.predict_kpi(z)
                    if kpi_hat is not None:
                        kpi_target = kpi_true[idx]
                        kpi_hat_centered = kpi_hat - torch.mean(kpi_hat, dim=0)
                        kpi_hat_std = torch.std(kpi_hat, dim=0) + 1e-8
                        kpi_hat_norm = kpi_hat_centered / kpi_hat_std

                        kpi_target_centered = kpi_target - torch.mean(kpi_target, dim=0)
                        kpi_target_std = torch.std(kpi_target, dim=0) + 1e-8
                        kpi_target_norm = kpi_target_centered / kpi_target_std

                        loss_kpi = mse_loss(kpi_hat_norm, kpi_target_norm)

                N_b, K = z.shape
                if N_b <= 1:
                    loss_ortho = torch.tensor(0.0, device=z.device)
                    loss_cross = torch.tensor(0.0, device=z.device)
                else:
                    z_centered = z - torch.mean(z, dim=0)
                    z_std = torch.std(z, dim=0) + 1e-8
                    z_norm = z_centered / z_std
                    corr_matrix = (1.0 / (N_b - 1)) * torch.matmul(z_norm.t(), z_norm)
                    identity = torch.eye(K).to(z.device)
                    loss_ortho = torch.sum((corr_matrix - identity) ** 2)
                    if semantic_dim_eff > 0 and semantic_dim_eff < K:
                        cross_block = corr_matrix[:semantic_dim_eff, semantic_dim_eff:]
                        loss_cross = torch.mean(cross_block ** 2)
                    else:
                        loss_cross = torch.tensor(0.0, device=z.device)

                loss_total = (
                    loss_recon
                    + stage_lam_phy * loss_phy
                    + stage_lam_ortho * loss_ortho
                    + stage_lam_kpi * loss_kpi
                    + stage_lam_cross * loss_cross
                )

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                recon_sum += float(loss_recon.detach().cpu().item())
                phy_sum += float(loss_phy.detach().cpu().item())
                ortho_sum += float(loss_ortho.detach().cpu().item())
                ortho_sum += float(loss_cross.detach().cpu().item())
                kpi_sum += float(loss_kpi.detach().cpu().item())
                count += 1

            if (epoch + 1) % 10 == 0 or (epoch + 1) == stage_epochs:
                print(
                    f"[{stage_name}] Epoch {epoch+1}/{stage_epochs} | Recon: {recon_sum/max(count,1):.4f} | "
                    f"Phy: {phy_sum/max(count,1):.4f} | Ortho: {ortho_sum/max(count,1):.4f} | KPI: {kpi_sum/max(count,1):.4f}"
                )

    pretrain_epochs_eff = int(max(0, pretrain_epochs))
    finetune_epochs_eff = int(max(0, epochs))
    if pretrain_epochs_eff > 0:
        _run_stage(
            stage_epochs=pretrain_epochs_eff,
            stage_name="pretrain",
            stage_lam_phy=0.0,
            stage_lam_ortho=0.0,
            stage_lam_kpi=0.0,
            stage_lam_cross=0.0,
        )
    if finetune_epochs_eff > 0:
        _run_stage(
            stage_epochs=finetune_epochs_eff,
            stage_name="finetune" if pretrain_epochs_eff > 0 else "train",
            stage_lam_phy=lam_phy,
            stage_lam_ortho=lam_ortho,
            stage_lam_kpi=lam_kpi,
            stage_lam_cross=lam_cross,
        )
            
    return model


def train_kpi_regressor(
    model: TCNKPIRegressor,
    Y: torch.Tensor,
    kpi_true: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: Optional[int] = None,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    if Y.dim() == 2:
        Y = Y.unsqueeze(1)

    if kpi_true.dim() == 1:
        kpi_true = kpi_true.unsqueeze(1)
    if kpi_true.device != Y.device:
        kpi_true = kpi_true.to(Y.device)

    N_total = Y.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= N_total:
        batch_size_eff = N_total
    else:
        batch_size_eff = int(batch_size)

    for epoch in range(epochs):
        perm = torch.randperm(N_total, device=Y.device)
        loss_sum = 0.0
        count = 0

        for start in range(0, N_total, batch_size_eff):
            idx = perm[start:start + batch_size_eff]
            Y_b = Y[idx]
            _, kpi_hat = model(Y_b)
            kpi_target = kpi_true[idx]

            kpi_hat_centered = kpi_hat - torch.mean(kpi_hat, dim=0)
            kpi_hat_std = torch.std(kpi_hat, dim=0) + 1e-8
            kpi_hat_norm = kpi_hat_centered / kpi_hat_std

            kpi_target_centered = kpi_target - torch.mean(kpi_target, dim=0)
            kpi_target_std = torch.std(kpi_target, dim=0) + 1e-8
            kpi_target_norm = kpi_target_centered / kpi_target_std

            loss = mse_loss(kpi_hat_norm, kpi_target_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu().item())
            count += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | KPI: {loss_sum/max(count,1):.4f}")

    return model

def train_dynamic(
    model: TCNAE,
    weight_net: WeightNet,
    Y: torch.Tensor,
    operators: List[PhysicalOperator],
    p_true: Dict[str, torch.Tensor],
    epochs: int = 500,
    lr: float = 1e-3
):
    """
    Keep existing train_dynamic for reference or other uses.
    """
    model.train()
    weight_net.train()
    combined_params = list(model.parameters()) + list(weight_net.parameters())
    optimizer = torch.optim.Adam(combined_params, lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        z, y_hat = model(Y)
        z_phy_1 = z[:, 0:1]
        z_phy_2 = z[:, 1:2]
        loss_phy_1 = mse_loss(z_phy_1, p_true[operators[0].name])
        loss_phy_2 = mse_loss(z_phy_2, p_true[operators[1].name])
        physics_losses = [loss_phy_1, loss_phy_2]
        loss_recon = mse_loss(y_hat, Y)
        loss_ortho = torch.tensor(0.0) 
        dummy_input = torch.ones(Y.shape[0], 1) 
        recon_weight, phy_weights, ortho_weight = weight_net(dummy_input)
        recon_weight = recon_weight.mean()
        ortho_weight = ortho_weight.mean()
        total_loss = recon_weight * loss_recon + ortho_weight * loss_ortho
        for i, loss in enumerate(physics_losses):
            total_loss += phy_weights[:, i].mean() * loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss.item():.4f} | Recon: {loss_recon.item():.4f}")
    return model, weight_net

def train(model, Y, pamp, pphase, epochs=500, lr=1e-3, lam=1.0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    Yt = torch.tensor(Y, dtype=torch.float32)
    pamp_t = torch.tensor(pamp, dtype=torch.float32).unsqueeze(1)
    pphase_t = torch.tensor(pphase, dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        z, yhat = model(Yt)
        z_amp, z_phase, z_res = model.get_disentangled_latent(z)
        loss_recon = mse(yhat, Yt)
        loss_phy = mse(z_amp, pamp_t) + mse(z_phase, pphase_t)
        loss_total = loss_recon + lam * loss_phy
        opt.zero_grad()
        loss_total.backward()
        opt.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_total.item():.4f} (Recon: {loss_recon.item():.4f}, Phy: {loss_phy.item():.4f})")
    return model
