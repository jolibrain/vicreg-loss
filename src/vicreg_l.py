from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vicreg import VICRegLoss


class VICRegLLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.vicreg_loss = VICRegLoss()

    def local_loss(
        self,
        x1_maps: torch.Tensor,
        x2_maps: torch.Tensor,
        x1_locations: torch.Tensor,
        x2_locations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pass

    def global_loss(
        self, x1_glob: torch.Tensor, x2_glob: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """The global loss is the original VICReg loss applied
        to the global features.
        """
        global_loss = self.vicreg_loss(x1_glob, x2_glob)
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
                Shape of [batch_size, n_patches, representation_size].
            x2_maps: Features map.
                Shape of [batch_size, n_patches, representation_size].
            x1_glob: Global features after pooling layer.
                Shape of [batch_size, representation_size].
            x2_glob: Global features after pooling layer.
                Shape of [batch_size, representation_size].
            x1_locations: Locations of the patches.
                Shape of [batch_size, n_patches, 2].
            x2_locations: Locations of the patches.
                Shape of [batch_size, n_patches, 2].

        ---
        Returns:
            The VICRegL loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = dict()
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
