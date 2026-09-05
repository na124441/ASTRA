"""ProcedureGraph representation for deterministic procedural reasoning."""

from __future__ import annotations

from collections import deque
from typing import Any
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep


class ProcedureGraph:
    """
    Deterministic directed state graph compiled from a ProcedureDefinition.
    Supports linear workflows, branching choices, optional steps, and skip-path analysis.
    """

    def __init__(self, definition: ProcedureDefinition) -> None:
        self.definition = definition
        self.steps: dict[str, ProcedureStep] = {}
        self.adjacency: dict[str, list[str]] = {}
        self.reverse_adjacency: dict[str, list[str]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct adjacency lists and index steps."""
        for step in self.definition.steps:
            self.steps[step.id] = step
            self.adjacency[step.id] = []
            self.reverse_adjacency[step.id] = []

        step_ids = list(self.steps.keys())
        for idx, step in enumerate(self.definition.steps):
            if step.allowed_next:
                # Explicit outgoing transitions specified
                for next_id in step.allowed_next:
                    if next_id in self.steps:
                        self.adjacency[step.id].append(next_id)
                        self.reverse_adjacency[next_id].append(step.id)
            else:
                # Default linear progression if not terminal and not last
                is_terminal = (
                    self.definition.terminal_step_ids
                    and step.id in self.definition.terminal_step_ids
                )
                if not is_terminal and idx + 1 < len(step_ids):
                    next_id = step_ids[idx + 1]
                    self.adjacency[step.id].append(next_id)
                    self.reverse_adjacency[next_id].append(step.id)

    @property
    def initial_step(self) -> ProcedureStep | None:
        """Get the entry step of the procedure."""
        if self.definition.initial_step_id and self.definition.initial_step_id in self.steps:
            return self.steps[self.definition.initial_step_id]
        if self.definition.steps:
            return self.definition.steps[0]
        return None

    def get_step(self, step_id: str) -> ProcedureStep | None:
        """Lookup a step by ID."""
        return self.steps.get(step_id)

    def get_allowed_next_steps(self, current_step_id: str | None) -> list[ProcedureStep]:
        """
        Return all valid immediate next steps from current state.
        If current_step_id is None, returns the initial entry step(s).
        """
        if current_step_id is None:
            init = self.initial_step
            return [init] if init else []

        next_ids = self.adjacency.get(current_step_id, [])
        next_steps = [self.steps[nid] for nid in next_ids if nid in self.steps]

        # If current step is repeatable, it is allowed to transition to itself
        curr = self.get_step(current_step_id)
        if curr and curr.repeatable and curr not in next_steps:
            next_steps.append(curr)

        return next_steps

    def is_terminal_step(self, step_id: str) -> bool:
        """Check if the given step marks completion of the procedure."""
        if self.definition.terminal_step_ids:
            return step_id in self.definition.terminal_step_ids
        # If no explicit terminals defined, any step with no outgoing edges is terminal
        return len(self.adjacency.get(step_id, [])) == 0

    def match_step(
        self,
        candidate_step: ProcedureStep,
        action: str,
        object_id: str | None = None,
        target_id: str | None = None,
    ) -> bool:
        """
        Check if an observed action matches candidate step requirements.
        Case-insensitive action comparison. Object and target must match if specified.
        """
        if candidate_step.action.upper() != action.upper():
            return False

        if candidate_step.object is not None:
            if object_id is None or candidate_step.object.upper() != object_id.upper():
                return False

        if candidate_step.target is not None:
            if target_id is None or candidate_step.target.upper() != target_id.upper():
                return False

        return True

    def find_matching_next_step(
        self,
        current_step_id: str | None,
        action: str,
        object_id: str | None = None,
        target_id: str | None = None,
    ) -> ProcedureStep | None:
        """
        Find an immediate valid next step from current state that satisfies the action.
        """
        allowed = self.get_allowed_next_steps(current_step_id)
        for step in allowed:
            if self.match_step(step, action, object_id, target_id):
                return step
        return None

    def find_any_matching_steps(
        self,
        action: str,
        object_id: str | None = None,
        target_id: str | None = None,
    ) -> list[ProcedureStep]:
        """Find any step anywhere in the graph matching this action/object/target."""
        matches: list[ProcedureStep] = []
        for step in self.steps.values():
            if self.match_step(step, action, object_id, target_id):
                matches.append(step)
        return matches

    def get_path_between(self, start_id: str, end_id: str) -> list[str] | None:
        """
        Find shortest directed path (list of step IDs) from start_id to end_id using BFS.
        Returns None if unreachable.
        """
        if start_id == end_id:
            return [start_id]

        queue: deque[list[str]] = deque([[start_id]])
        visited = {start_id}

        while queue:
            path = queue.popleft()
            current = path[-1]

            for neighbor in self.adjacency.get(current, []):
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def get_skipped_steps(self, from_step_id: str | None, to_step_id: str) -> list[ProcedureStep]:
        """
        If to_step_id is reachable from from_step_id but not an immediate valid transition,
        return the non-optional intermediate steps that were skipped.
        """
        if from_step_id is None:
            init = self.initial_step
            if not init or init.id == to_step_id:
                return []
            from_step_id = init.id
            path = self.get_path_between(from_step_id, to_step_id)
            if path:
                # include initial step in skipped list if it wasn't the target
                full_path = path
                intermediate_ids = full_path[:-1]
                return [self.steps[sid] for sid in intermediate_ids if not self.steps[sid].optional]
            return []

        path = self.get_path_between(from_step_id, to_step_id)
        if not path or len(path) <= 2:
            return []

        # Intermediate steps are strictly between start and end: path[1:-1]
        skipped_ids = path[1:-1]
        return [self.steps[sid] for sid in skipped_ids if not self.steps[sid].optional]
