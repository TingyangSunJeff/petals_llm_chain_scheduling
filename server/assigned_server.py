"""Extended PETALS Server with fixed block assignment from the orchestrator.

Overrides PETALS' dynamic block selection with a static assignment
dictated by GBP-CR. Prevents rebalancing since blocks are centrally managed.

Reference: Requirement 7 of the spec.
"""

from __future__ import annotations

from typing import List, Optional

from petals.server.server import Server


class AssignedServer(Server):
    """PETALS Server that loads a specific block range assigned by the orchestrator.

    Instead of using PETALS' greedy block selection, this server always loads
    the blocks specified by start_block and end_block. It also disables
    automatic rebalancing since block placement is managed centrally.
    """

    def __init__(
        self,
        *,
        start_block: int,
        end_block: int,
        cache_reservation: int = 1,
        **petals_server_kwargs,
    ):
        """Initialize with a fixed block assignment.

        Args:
            start_block: First block index (0-indexed, inclusive).
            end_block: Last block index (0-indexed, exclusive).
            cache_reservation: Parameter c for cache reservation.
            **petals_server_kwargs: All other args passed to petals.Server.
        """
        self.assigned_start = start_block
        self.assigned_end = end_block
        self.cache_reservation = cache_reservation

        # Pass block_indices as "start:end" string to PETALS Server
        petals_server_kwargs["block_indices"] = f"{start_block}:{end_block}"
        # Disable automatic num_blocks selection
        petals_server_kwargs.setdefault("num_blocks", end_block - start_block)

        super().__init__(**petals_server_kwargs)

    def _choose_blocks(self) -> List[int]:
        """Override: always return the assigned block range."""
        return list(range(self.assigned_start, self.assigned_end))

    def _should_choose_other_blocks(self) -> bool:
        """Override: never rebalance — blocks are assigned by orchestrator."""
        return False
