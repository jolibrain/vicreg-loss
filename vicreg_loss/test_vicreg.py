from typing import Tuple

import pytest
import torch

from .vicregl import VICRegLLoss


@pytest.mark.parametrize(
    "input_maps, candidate_maps, distances, num_matches",
    [
        (
            torch.randn(1, 10, 3),
            torch.randn(1, 10, 3),
            torch.randn(1, 10, 10),
            2,
        ),
        (
            torch.randn(4, 10, 3),
            torch.randn(4, 12, 3),
            torch.randn(4, 10, 12),
            5,
        ),
        (
            torch.randn(4, 10, 3),
            torch.randn(4, 12, 3),
            torch.randn(4, 10, 12),
            10,
        ),
    ],
)
def test_neirest_neighbores(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor,
    num_matches: int,
):
    def naive_neirest_neighbores(
        input_maps: torch.Tensor,
        candidate_maps: torch.Tensor,
        distances: torch.Tensor,
        num_matches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        best_candidates = distances.argmin(dim=2)
        best_distances = torch.zeros(distances.shape[0], distances.shape[1])
        for batch_id in range(distances.shape[0]):
            for element_id in range(distances.shape[1]):
                best_distances[batch_id, element_id] = distances[
                    batch_id, element_id, best_candidates[batch_id, element_id]
                ]

        filtered_candidate_maps = torch.zeros(
            distances.shape[0], distances.shape[1], candidate_maps.shape[2]
        )
        for batch_id in range(distances.shape[0]):
            for element_id in range(distances.shape[1]):
                filtered_candidate_maps[batch_id, element_id] = candidate_maps[
                    batch_id, best_candidates[batch_id, element_id]
                ]

        naive_input_maps = torch.zeros(
            distances.shape[0], num_matches, input_maps.shape[2]
        )
        naive_candidate_maps = torch.zeros(
            distances.shape[0], num_matches, candidate_maps.shape[2]
        )
        for batch_id in range(distances.shape[0]):
            sort = torch.argsort(best_distances[batch_id])

            for i in range(num_matches):
                naive_input_maps[batch_id, i] = input_maps[batch_id, sort[i]]
                naive_candidate_maps[batch_id, i] = filtered_candidate_maps[
                    batch_id, sort[i]
                ]

        return naive_input_maps, naive_candidate_maps

    proposed_input_maps, proposed_candidate_maps = VICRegLLoss.neirest_neighbores(
        input_maps, candidate_maps, distances, num_matches
    )
    naive_input_maps, naive_candidate_maps = naive_neirest_neighbores(
        input_maps, candidate_maps, distances, num_matches
    )

    assert torch.all(proposed_input_maps == naive_input_maps)
    assert torch.all(proposed_candidate_maps == naive_candidate_maps)
