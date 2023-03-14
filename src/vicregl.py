from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vicreg import VICRegLoss


class VICRegLLoss(nn.Module):
    def __init__(
        self,
        num_matches: int = 10,
        alpha: float = 0.75,
        inv_coeff: float = 25.0,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()

        assert 0.0 <= alpha <= 1.0

        self.num_matches = num_matches
        self.alpha = alpha
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

        self.global_vicreg_loss = VICRegLoss(inv_coeff, var_coeff, cov_coeff, gamma)

    def local_vicreg_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Computes the VICReg loss taking into account the patches.
        This means that the variance and the covariance are computed
        per patch and not globally.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, n_patches, representation_size].
            y: Features map.
                Shape of [batch_size, n_patches, representation_size].

        ---
        Returns:
            The VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = dict()

        # Invariance loss.
        inv = F.mse_loss(x, y)
        metrics["inv-loss"] = self.inv_coeff * inv

        # Variance loss.
        var_x = VICRegLLoss.variance_loss(x, self.gamma)
        var_y = VICRegLLoss.variance_loss(y, self.gamma)
        metrics["var-loss"] = self.var_coeff * (var_x + var_y) / 2

        # Covariance loss.
        cov_x = VICRegLLoss.covariance_loss(x)
        cov_y = VICRegLLoss.covariance_loss(y)
        metrics["cov-loss"] = self.cov_coeff * (cov_x + cov_y) / 2

        metrics["loss"] = sum(metrics.values())
        return metrics

    def local_loss(
        self,
        x1_maps: torch.Tensor,
        x2_maps: torch.Tensor,
        x1_locations: torch.Tensor,
        x2_locations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the VICReg loss taking into account the patches and the location.
        It gather the closest patches of each element in the batch and compute the
        VICReg loss on them. Patches are considered close based on their locations and
        on their features.

        ---
        Args:
            x1_maps: Features map.
                Shape of [batch_size, n_patches, representation_size].
            x2_maps: Features map.
                Shape of [batch_size, n_patches, representation_size].
            x1_locations: Absolute locations of the patches.
                Shape of [batch_size, n_patches, 2].
            x2_locations: Absolute locations of the patches.
                Shape of [batch_size, n_patches, 2].

        ---
        Returns:
            The local VICReg losses.
        """

        def location_loss(
            input_maps: torch.Tensor,
            candidate_maps: torch.Tensor,
            input_locations: torch.Tensor,
            candidate_loctaions: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Location-based local loss."""
            distances = torch.cdist(input_locations, candidate_loctaions, p=2)
            input_maps, candidate_maps = VICRegLLoss.neirest_neighbores(
                input_maps, candidate_maps, distances, self.num_matches
            )
            return self.local_vicreg_loss(input_maps, candidate_maps)

        def feature_loss(
            input_maps: torch.Tensor,
            candidate_maps: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Feature-based local loss."""
            distances = torch.cdist(input_maps, candidate_maps, p=2)
            input_maps, candidate_maps = VICRegLLoss.neirest_neighbores(
                input_maps, candidate_maps, distances, self.num_matches
            )
            return self.local_vicreg_loss(input_maps, candidate_maps)

        x1_loss = location_loss(x1_maps, x2_maps, x1_locations, x2_locations)
        x2_loss = location_loss(x2_maps, x1_maps, x2_locations, x1_locations)
        loss = {
            f"location/{key}": (x1_loss[key] + x2_loss[key]) / 2
            for key in x1_loss.keys()
        }

        x1_loss = feature_loss(x1_maps, x2_maps)
        x2_loss = feature_loss(x2_maps, x1_maps)
        loss.update(
            {
                f"feature/{key}": (x1_loss[key] + x2_loss[key]) / 2
                for key in x1_loss.keys()
            }
        )

        return loss

    def global_loss(
        self, x1_glob: torch.Tensor, x2_glob: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """The global loss is the original VICReg loss applied
        to the global features.
        """
        global_loss = self.global_vicreg_loss(x1_glob, x2_glob)
        global_loss = {f"global/{k}": v for k, v in global_loss.items()}
        return global_loss

    def forward(
        self,
        x1_maps: torch.Tensor,
        x2_maps: torch.Tensor,
        x1_glob: torch.Tensor,
        x2_glob: torch.Tensor,
        x1_locations: torch.Tensor,
        x2_locations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Computes the VICRegL loss.

        ---
        Args:
            x1_maps: Features map.
                Shape of [batch_size, n_patches, local_size].
            x2_maps: Features map.
                Shape of [batch_size, n_patches, local_size].
            x1_glob: Global features after pooling layer.
                Shape of [batch_size, global_size].
            x2_glob: Global features after pooling layer.
                Shape of [batch_size, global_size].
            x1_locations: Absolute locations of the patches.
                Shape of [batch_size, n_patches, 2].
            x2_locations: Absolute locations of the patches.
                Shape of [batch_size, n_patches, 2].

        ---
        Returns:
            The VICRegL loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = {
            "loss": torch.tensor(0.0, device=x1_maps.device),
        }

        if self.alpha > 0.0:
            metrics.update(self.global_loss(x1_glob, x2_glob))
            metrics["loss"] = metrics["loss"] + self.alpha * metrics["global/loss"]

        if self.alpha < 1.0:
            metrics.update(
                self.local_loss(x1_maps, x2_maps, x1_locations, x2_locations)
            )
            metrics["loss"] = (
                metrics["loss"]
                + (1 - self.alpha)
                * (metrics["feature/loss"] + metrics["location/loss"])
                / 2
            )

        return metrics

    @staticmethod
    def neirest_neighbores(
        input_maps: torch.Tensor,
        candidate_maps: torch.Tensor,
        distances: torch.Tensor,
        num_matches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the first `num_matches` closest elements from the candidates to
        the input maps.
        Each element in the input is mapped to its closest element among the candidates.

        ---
        Args:
            input_maps: Features map.
                Shape of [batch_size, n_elements, hidden_size].
            candidate_maps: Features map.
                Shape of [batch_size, n_candidates, hidden_size].
            distances: Distances between the elements and the candidates.
                Shape of [batch_size, n_elements, n_candidates].
            num_matches: Number of matches to find.

        ---
        Returns:
            filtered_input_maps: The features of the first `n_matches` inputs.
                Shape of [batch_size, n_matches, hidden_size].
            filtered_candidate_maps: The features of the first `n_matches` candidates.
                Shape of [batch_size, n_matches, hidden_size].
        """
        assert input_maps.shape[1] >= num_matches
        assert candidate_maps.shape[1] >= num_matches

        # Find the closest candidate to each input element in the distance matrix.
        topk_values, topk_indices = distances.topk(k=1, dim=-1, largest=False)

        # To [batch_size, n_elements].
        topk_values = topk_values.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        # Only keep the first `num_matches` candidates.
        _, element_indices = topk_values.topk(k=num_matches, dim=-1, largest=False)
        candidate_indices = topk_indices.gather(dim=-1, index=element_indices)

        # Gather the rights features map.
        filtered_input_maps = input_maps.gather(
            dim=1, index=element_indices.unsqueeze(-1)
        )
        filtered_candidate_maps = candidate_maps.gather(
            dim=1, index=candidate_indices.unsqueeze(-1)
        )

        return filtered_input_maps, filtered_candidate_maps

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the local variance loss.
        This is slightly different than the global variance loss because the
        variance is computed per patch.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, n_patches, representation_size].

        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=1, keepdim=True)
        std = x.std(dim=1)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """Computes the local covariance loss.
        This is slightly different than the global covariance loss because the
        covariance is computed per patch.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, n_patches, representation_size].

        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=1, keepdim=True)
        x_T = x.transpose(1, 2)
        cov = (x_T @ x) / (x.shape[1] - 1)

        non_diag = ~torch.eye(x.shape[2], device=x.device, dtype=torch.bool)
        cov_loss = cov[..., non_diag].pow(2).sum() / (x.shape[2] * x.shape[0])
        return cov_loss
