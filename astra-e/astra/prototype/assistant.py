"""State-Grounded Conversational Assistant ('Ask ASTRA') for Experiments (Section 13).

Grounds Q&A directly on the deterministic experiment state machine and observation history.
Never asks an LLM or neural model to infer procedure progress from raw video;
instead leverages the structured state machine belief.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from astra.prototype.alignment_engine import TemporalAlignmentEngine
from astra.prototype.schemas import (
    ConversationalAnswer,
    ConversationalQuery,
    ExperimentRuntimeState,
    ProceduralAlignmentStatus,
)

logger = logging.getLogger("astra.prototype.assistant")


class ConversationalAssistant:
    """
    Operator conversational dialogue agent.
    Provides natural verbal answers grounded strictly on the active ExperimentRuntimeState.
    """

    def __init__(self, alignment_engine: TemporalAlignmentEngine) -> None:
        self.alignment_engine = alignment_engine

    def answer_query(self, query: str | ConversationalQuery) -> ConversationalAnswer:
        """Processes natural language operator query and returns structured contextual answer."""
        text = query.question if isinstance(query, ConversationalQuery) else str(query)
        clean = text.lower().strip()
        state = self.alignment_engine.get_runtime_state()

        logger.info(f"[ASK ASTRA QUERY] '{text}' (Current state: {state.current_step_id}, Status: {state.alignment_status.value})")

        # 1. Query: "What am I supposed to do now?" / "What is next?" / "What step are we on?"
        if any(kw in clean for kw in ["what do i do", "what am i supposed", "what next", "next step", "what should i do", "where are we"]):
            return self._answer_next_step(state)

        # 2. Query: "Did I do the last step correctly?" / "Was the last step correct?"
        if any(kw in clean for kw in ["did i do the last", "was the last step", "did i do it right", "was that correct", "did i perform"]):
            return self._answer_previous_step(state)

        # 3. Query: "What went wrong?" / "Is there a deviation?" / "Why warning?"
        if any(kw in clean for kw in ["what went wrong", "deviation", "what is wrong", "why warning", "error", "problem"]):
            return self._answer_deviation_status(state)

        # 4. Query: "How many steps remain?" / "How far are we?" / "Progress?"
        if any(kw in clean for kw in ["how many steps", "remaining", "progress", "completion", "how far"]):
            return self._answer_progress(state)

        # 5. General fallback: State recap and next guidance
        return self._answer_general(state, text)

    def _answer_next_step(self, state: ExperimentRuntimeState) -> ConversationalAnswer:
        if state.alignment_status == ProceduralAlignmentStatus.COMPLETED:
            ans = "All experiment steps have been successfully completed and verified. You may proceed to payload shutdown."
            return ConversationalAnswer(
                answer=ans,
                current_step_id=None,
                alignment_status=state.alignment_status,
                suggested_action="Shutdown payload",
            )

        curr_step = self.alignment_engine.current_step
        if not curr_step:
            ans = "The experiment has not been initialized yet. Ready to begin step 1."
            return ConversationalAnswer(answer=ans, alignment_status=state.alignment_status)

        # If there's an active deviation, prioritize warning
        if state.alignment_status in (ProceduralAlignmentStatus.DEVIATION, ProceduralAlignmentStatus.CRITICAL):
            ans = f"Attention: {state.last_deviation_message or 'A procedural deviation was detected.'} {state.active_guidance_text or ''}"
            return ConversationalAnswer(
                answer=ans,
                current_step_id=curr_step.id,
                alignment_status=state.alignment_status,
                suggested_action=state.active_guidance_text or curr_step.description,
            )

        completed_count = len(state.completed_steps)
        if completed_count == 0:
            ans = f"We are at step 1. Please {curr_step.description.lower()}."
        else:
            prev_id = state.completed_steps[-1]
            prev_step = self.alignment_engine.step_map.get(prev_id)
            prev_desc = f"the {prev_step.description.lower()}" if prev_step else f"step {completed_count}"
            ans = f"You have completed {prev_desc}. Next, please {curr_step.description.lower()}."

        return ConversationalAnswer(
            answer=ans,
            current_step_id=curr_step.id,
            alignment_status=state.alignment_status,
            suggested_action=curr_step.description,
        )

    def _answer_previous_step(self, state: ExperimentRuntimeState) -> ConversationalAnswer:
        if state.alignment_status in (ProceduralAlignmentStatus.DEVIATION, ProceduralAlignmentStatus.CRITICAL):
            ans = f"No. {state.last_deviation_message or 'The last action did not match the required procedure.'} {state.active_guidance_text or ''}"
            return ConversationalAnswer(
                answer=ans,
                current_step_id=state.current_step_id,
                alignment_status=state.alignment_status,
                suggested_action=state.active_guidance_text or "",
            )

        if not state.completed_steps:
            ans = "No steps have been completed yet. The experiment is at the initial step."
            return ConversationalAnswer(
                answer=ans,
                current_step_id=state.current_step_id,
                alignment_status=state.alignment_status,
            )

        last_id = state.completed_steps[-1]
        last_step = self.alignment_engine.step_map.get(last_id)
        if last_step:
            ans = f"Yes. Step {last_step.step_number} ({last_step.description}) was confirmed and verified. The observed action was fully aligned."
        else:
            ans = f"Yes. Step {len(state.completed_steps)} was completed and verified aligned."

        return ConversationalAnswer(
            answer=ans,
            current_step_id=state.current_step_id,
            alignment_status=state.alignment_status,
        )

    def _answer_deviation_status(self, state: ExperimentRuntimeState) -> ConversationalAnswer:
        if state.alignment_status == ProceduralAlignmentStatus.ALIGNED:
            ans = "No deviations are present. All observed actions are strictly aligned with protocol EXP001."
        elif state.alignment_status == ProceduralAlignmentStatus.COMPLETED:
            ans = "The procedure is finished with zero pending deviations."
        else:
            ans = f"Deviation detected: {state.last_deviation_message} Recommended correction: {state.active_guidance_text}"

        return ConversationalAnswer(
            answer=ans,
            current_step_id=state.current_step_id,
            alignment_status=state.alignment_status,
        )

    def _answer_progress(self, state: ExperimentRuntimeState) -> ConversationalAnswer:
        total = state.total_steps
        done = len(state.completed_steps)
        remaining = total - done
        ans = f"You have completed {done} of {total} steps. {remaining} step{'s' if remaining != 1 else ''} remaining."
        return ConversationalAnswer(
            answer=ans,
            current_step_id=state.current_step_id,
            alignment_status=state.alignment_status,
        )

    def _answer_general(self, state: ExperimentRuntimeState, raw_query: str) -> ConversationalAnswer:
        curr_step = self.alignment_engine.current_step
        curr_desc = curr_step.description if curr_step else "Completion"
        ans = (
            f"ASTRA is monitoring '{state.experiment_name}'. Current step is {state.current_step_number} of {state.total_steps}: {curr_desc}. "
            f"Status is {state.alignment_status.value}."
        )
        return ConversationalAnswer(
            answer=ans,
            current_step_id=state.current_step_id,
            alignment_status=state.alignment_status,
        )
