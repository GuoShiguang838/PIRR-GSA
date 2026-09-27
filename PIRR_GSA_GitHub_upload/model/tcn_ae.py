import torch
import torch.nn as nn
from typing import Optional

class TCNAE(nn.Module):
    def __init__(
        self,
        T,
        latent_dim=4,
        input_channels=1,
        num_single_field: int = 2,
        num_coupling: int = 0,
        kpi_dim: Optional[int] = None,
        kpi_hidden_dim: Optional[int] = None,
    ):
        super(TCNAE, self).__init__()
        self.T = T
        self.input_channels = input_channels
        
        # Latent space dimensions
        self.num_single_field = num_single_field
        self.num_coupling = num_coupling
        self.latent_dim = latent_dim # User can specify total dim, but we track physical dims
        
        # Encoder for long and multiscale temporal responses.
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, T)
            dummy_encoded = self.encoder_conv(dummy_input)
            self.flattened_size = dummy_encoded.shape[1]
            
        self.encoder_fc = nn.Linear(self.flattened_size, self.latent_dim)
        
        # Enhanced Decoder
        self.decoder_fc = nn.Linear(self.latent_dim, self.flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, self.flattened_size // 256)),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        if kpi_dim is None or int(kpi_dim) <= 0:
            self.kpi_head = None
        else:
            hidden = int(kpi_hidden_dim) if kpi_hidden_dim is not None else max(16, int(self.latent_dim))
            self.kpi_head = nn.Sequential(
                nn.Linear(self.latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(kpi_dim)),
            )

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)
        y = self.decoder_conv(h)
        
        if y.shape[2] != self.T:
            y = torch.nn.functional.interpolate(y, size=self.T, mode='linear', align_corners=False)
        
        if self.input_channels == 1:
            return y.squeeze(1)
        else:
            return y

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return z, y

    def predict_kpi(self, z):
        if getattr(self, "kpi_head", None) is None:
            return None
        return self.kpi_head(z)


class DescriptorInjectedResidualTCNAE(nn.Module):
    """TCN autoencoder with exact descriptor coordinates and learned residuals.

    The first ``descriptor_dim`` latent coordinates are supplied explicitly
    rather than approximated by the encoder.  Only the remaining
    ``residual_dim`` coordinates are learned from the temporal response.  The
    decoder consumes both blocks, so the representation preserves a fixed,
    named descriptor interface while retaining response information that is
    not captured by those descriptors.

    Descriptor standardization is deliberately performed by the caller using
    statistics fitted on the training design.  This keeps the model free of
    test-set state and makes the exact-coordinate contract easy to audit.
    """

    def __init__(
        self,
        T: int,
        descriptor_dim: int,
        residual_dim: int = 2,
        input_channels: int = 1,
        descriptor_operator: Optional[torch.Tensor] = None,
        descriptor_mean: Optional[torch.Tensor] = None,
        descriptor_scale: Optional[torch.Tensor] = None,
        descriptor_lift: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.T = int(T)
        self.input_channels = int(input_channels)
        self.descriptor_dim = int(descriptor_dim)
        self.residual_dim = int(residual_dim)
        if self.descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive.")
        if self.residual_dim < 0:
            raise ValueError("residual_dim must be non-negative.")
        self.latent_dim = self.descriptor_dim + self.residual_dim
        self.num_single_field = self.descriptor_dim
        self.num_coupling = 0

        if descriptor_operator is None:
            self.register_buffer("descriptor_operator", None)
            self.register_buffer("descriptor_lift", None)
            self.register_buffer("descriptor_mean", None)
            self.register_buffer("descriptor_scale", None)
        else:
            operator = torch.as_tensor(descriptor_operator, dtype=torch.float32)
            if operator.shape != (self.descriptor_dim, self.T):
                raise ValueError(
                    "descriptor_operator must have shape (descriptor_dim, T)."
                )
            if descriptor_lift is None:
                gram = operator @ operator.T
                lift = operator.T @ torch.linalg.pinv(gram)
            else:
                lift = torch.as_tensor(descriptor_lift, dtype=torch.float32)
                if lift.shape != (self.T, self.descriptor_dim):
                    raise ValueError(
                        "descriptor_lift must have shape (T, descriptor_dim)."
                    )
                identity_error = torch.max(
                    torch.abs(operator @ lift - torch.eye(self.descriptor_dim))
                )
                if float(identity_error) > 1e-4:
                    raise ValueError("descriptor_lift must satisfy operator @ lift = I.")
            mean = torch.as_tensor(descriptor_mean, dtype=torch.float32).reshape(-1)
            scale = torch.as_tensor(descriptor_scale, dtype=torch.float32).reshape(-1)
            if mean.numel() != self.descriptor_dim or scale.numel() != self.descriptor_dim:
                raise ValueError("descriptor_mean and descriptor_scale must have length K.")
            if torch.any(scale <= 0):
                raise ValueError("descriptor_scale must be positive.")
            self.register_buffer("descriptor_operator", operator)
            self.register_buffer("descriptor_lift", lift)
            self.register_buffer("descriptor_mean", mean)
            self.register_buffer("descriptor_scale", scale)

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.T)
            flattened_size = int(self.encoder_conv(dummy).shape[1])
        self.flattened_size = flattened_size
        self.encoder_fc = (
            nn.Linear(flattened_size, self.residual_dim)
            if self.residual_dim > 0
            else None
        )

        self.decoder_fc = nn.Linear(self.latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, flattened_size // 256)),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, self.input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def descriptor_values(self, descriptors_standardized: torch.Tensor) -> torch.Tensor:
        if self.descriptor_operator is None:
            raise RuntimeError("No fixed descriptor operator is configured.")
        return (
            descriptors_standardized * self.descriptor_scale[None, :]
            + self.descriptor_mean[None, :]
        )

    def lift_descriptors(self, descriptors_standardized: torch.Tensor) -> torch.Tensor:
        values = self.descriptor_values(descriptors_standardized)
        return values @ self.descriptor_lift.T

    def project_residual(self, response: torch.Tensor) -> torch.Tensor:
        """Project a response onto the null space of the descriptor operator."""
        if self.descriptor_operator is None:
            return response
        descriptor_part = response @ self.descriptor_operator.T
        return response - descriptor_part @ self.descriptor_lift.T

    def encode_residual(
        self,
        response: torch.Tensor,
        descriptors_standardized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response.dim() == 2:
            response = response.unsqueeze(1)
        if self.descriptor_operator is not None:
            if descriptors_standardized is None:
                raise ValueError("descriptors_standardized is required for residual projection.")
            response_2d = response.squeeze(1)
            response = self.project_residual(
                response_2d - self.lift_descriptors(descriptors_standardized)
            ).unsqueeze(1)
        features = self.encoder_conv(response)
        if self.encoder_fc is None:
            return features.new_zeros((features.shape[0], 0))
        return self.encoder_fc(features)

    def assemble_latent(
        self, descriptors_standardized: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        if descriptors_standardized.dim() != 2:
            raise ValueError("descriptors_standardized must have shape (N, K).")
        if descriptors_standardized.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"Expected {self.descriptor_dim} descriptor coordinates, "
                f"got {descriptors_standardized.shape[1]}."
            )
        if residual.dim() != 2 or residual.shape[1] != self.residual_dim:
            raise ValueError(
                f"Expected residual coordinates with shape (N, {self.residual_dim})."
            )
        if descriptors_standardized.shape[0] != residual.shape[0]:
            raise ValueError("Descriptor and residual batches must have equal size.")
        return torch.cat([descriptors_standardized, residual], dim=1)

    def encode(
        self, response: torch.Tensor, descriptors_standardized: torch.Tensor
    ) -> torch.Tensor:
        residual = self.encode_residual(response, descriptors_standardized)
        return self.assemble_latent(descriptors_standardized, residual)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_fc(latent)
        response = self.decoder_conv(hidden)
        if response.shape[2] != self.T:
            response = torch.nn.functional.interpolate(
                response, size=self.T, mode="linear", align_corners=False
            )
        if self.input_channels == 1:
            response = response.squeeze(1)
            if self.descriptor_operator is not None:
                descriptor_coordinates = latent[:, : self.descriptor_dim]
                response = self.project_residual(response) + self.lift_descriptors(
                    descriptor_coordinates
                )
            return response
        return response

    def forward(
        self, response: torch.Tensor, descriptors_standardized: torch.Tensor
    ):
        latent = self.encode(response, descriptors_standardized)
        return latent, self.decode(latent)


class TCNKPIRegressor(nn.Module):
    def __init__(
        self,
        T,
        latent_dim: int = 4,
        input_channels: int = 1,
        kpi_dim: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.T = T
        self.input_channels = input_channels
        self.latent_dim = int(latent_dim)
        self.kpi_dim = int(kpi_dim)

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, T)
            dummy_encoded = self.encoder_conv(dummy_input)
            flattened_size = int(dummy_encoded.shape[1])

        self.encoder_fc = nn.Linear(flattened_size, self.latent_dim)

        hid = int(hidden_dim) if hidden_dim is not None else max(16, self.latent_dim)
        self.kpi_head = nn.Sequential(
            nn.Linear(self.latent_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, self.kpi_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        kpi_hat = self.kpi_head(z)
        return z, kpi_hat

    def get_disentangled_latent(self, z):
        """
        Splits the latent space z into single-field, coupling, and residual parts.
        """
        z_single = z[:, :self.num_single_field]
        z_coupling = z[:, self.num_single_field:self.num_single_field + self.num_coupling]
        z_res = z[:, self.num_single_field + self.num_coupling:]
        return z_single, z_coupling, z_res
