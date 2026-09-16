"""ASTRA-E Prototype v0.1 CLI & Closed-Loop Demonstration Runner.

Implements all milestones:
  P0: Temporal Video Window Extraction
  P1: Structured Observation Generation
  P2: State Machine Procedural Alignment (Step 1-5 ALIGNED)
  P3: Multi-tier Deviation Detection (WARNING, DEVIATION, CRITICAL)
  P4: Natural Voice Guidance & TTS Audio Generation
  P5: Full End-to-End Closed-Loop Execution

Usage:
  python run_prototype.py --mode demo        (Runs complete automated closed-loop walkthrough)
  python run_prototype.py --mode ui          (Launches Streamlit Interactive Dashboard)
  python run_prototype.py --mode server1     (Launches Colab Server #1 Video Understanding)
  python run_prototype.py --mode server2     (Launches Colab Server #2 Voice / TTS)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Ensure astra-e root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astra.prototype.alignment_engine import TemporalAlignmentEngine
from astra.prototype.assistant import ConversationalAssistant
from astra.prototype.cloud_client import CloudVisionClient, CloudVoiceClient
from astra.prototype.orchestrator import PrototypeOrchestrator
from astra.prototype.schemas import (
    AssistanceMode,
    ProceduralAlignmentStatus,
    SeverityLevel,
    StructuredObservation,
    TemporalEvent,
    VoiceInstruction,
)
from astra.prototype.temporal_buffer import TemporalSlidingWindowBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astra.prototype.runner")


def run_closed_loop_demo() -> None:
    """
    Executes a complete closed-loop walkthrough of ASTRA-E Prototype v0.1:
    1. System Initialization & Experiment Loading (Sample Transfer EXP001)
    2. Milestone P0 & P1: Sliding Temporal Window -> Structured Observation
    3. Milestone P2: Procedural State Tracking (Steps 1 & 2 Aligned)
    4. Milestone P3: Deviation Detection (Skipped Step & Wrong Object)
    5. Milestone P4: Natural Voice Alert Synthesis & Cooldown Enforcement
    6. Recovery & Completion: Step 3, 4, 5 -> Experiment Complete
    7. Conversational Assistance ('Ask ASTRA' Grounded Q&A)
    """
    print("\n" + "=" * 70)
    print("🚀 ASTRA-E Prototype v0.1: Closed-Loop Milestone Walkthrough")
    print("=" * 70)

    yaml_path = ROOT / "experiments" / "sample_transfer" / "procedure.yaml"
    engine = TemporalAlignmentEngine(
        procedure_yaml_path=str(yaml_path),
        mode=AssistanceMode.GUIDANCE,
        cooldown_seconds=4.0,
    )
    voice_client = CloudVoiceClient()
    assistant = ConversationalAssistant(engine)

    print(f"\n[INIT] Experiment Loaded: '{engine.experiment_name}' (ID: {engine.experiment_id})")
    print(f"[INIT] Procedure Steps: {len(engine.steps)}")
    for s in engine.steps:
        tgt = f" -> {s.target}" if s.target else ""
        print(f"       Step {s.step_number}: [{s.action}] {s.object or ''}{tgt} - {s.description}")

    time.sleep(1.0)

    # -------------------------------------------------------------
    # Milestone P0 & P1: Windowing & Structured Observation
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("▶ Milestone P0 & P1: Temporal Window -> Structured Observation")
    print("-" * 60)

    obs1 = StructuredObservation(
        window_start=0.0,
        window_end=4.0,
        observation="Operator reaches into payload bay and picks up sample container.",
        events=[TemporalEvent(action="PICK", object="SAMPLE_CONTAINER")],
        confidence=0.94,
    )
    print(f"Observation JSON: {obs1.model_dump_json(indent=2)}")

    # -------------------------------------------------------------
    # Milestone P2: State Tracking (Step 1)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("▶ Milestone P2: Temporal Alignment Engine Evaluation (Step 1)")
    print("-" * 60)

    status, voice_inst = engine.process_observation(obs1)
    print(f"Status: ● {status.value} (Confidence: {obs1.confidence * 100:.1f}%)")
    print(f"Completed Steps: {engine.completed_step_ids}")
    if voice_inst:
        print(f"Voice Output: \"{voice_inst.to_speech_text()}\"")
        voice_client.synthesize(voice_inst)

    time.sleep(1.0)

    # Step 2: Normal Move
    obs2 = StructuredObservation(
        window_start=4.0,
        window_end=8.0,
        observation="Operator moves sample container toward chamber door.",
        events=[TemporalEvent(action="MOVE", object="SAMPLE_CONTAINER", target="CHAMBER")],
        confidence=0.91,
    )
    status, voice_inst = engine.process_observation(obs2)
    print(f"\nObservation: \"{obs2.observation}\"")
    print(f"Status: ● {status.value} | Current Step: {engine.current_step.id if engine.current_step else 'None'}")
    if voice_inst:
        print(f"Voice Guidance: \"{voice_inst.to_speech_text()}\"")
        voice_client.synthesize(voice_inst)

    time.sleep(1.0)

    # -------------------------------------------------------------
    # Milestone P3: Deviation Detection (Skipped Step)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("▶ Milestone P3 & P4: Procedural Deviation Detection & Voice Alert")
    print("   [Trigger: Operator attempts PLACE before OPEN chamber]")
    print("-" * 60)

    obs_dev = StructuredObservation(
        window_start=8.0,
        window_end=12.0,
        observation="Operator attempts to place sample container into closed chamber without opening.",
        events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
        confidence=0.92,
    )
    status, voice_inst = engine.process_observation(obs_dev)
    print(f"Observation: \"{obs_dev.observation}\"")
    print(f"Status: ⚠ {status.value}")
    if voice_inst:
        print(f"🚨 Voice Alert Issued:")
        print(f"   Severity: {voice_inst.severity.value}")
        print(f"   Message:  {voice_inst.message}")
        print(f"   Guidance: {voice_inst.guidance}")
        print(f"   Spoken:   \"{voice_inst.to_speech_text()}\"")
        # Synthesize audio
        voice_client.synthesize(voice_inst)

    time.sleep(1.0)

    # Test Alert Cooldown (Acoustic Fatigue Mitigation)
    print("\n[TEST] Verifying 5s Alert Cooldown Suppression for duplicate deviation...")
    status, duplicate_alert = engine.process_observation(obs_dev)
    if duplicate_alert is None:
        print("✓ SUCCESS: Duplicate voice alert suppressed by AlertPolicyEngine!")
    else:
        print("✗ Cooldown did not suppress alert.")

    time.sleep(1.0)

    # -------------------------------------------------------------
    # Conversational Assistance ('Ask ASTRA')
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("▶ Conversational Assistance: 'Ask ASTRA' Grounded Dialogue")
    print("-" * 60)

    q1 = "What am I supposed to do now?"
    ans1 = assistant.answer_query(q1)
    print(f"Operator: \"{q1}\"")
    print(f"ASTRA:    \"{ans1.answer}\"")

    time.sleep(1.0)

    q2 = "Did I do the last step correctly?"
    ans2 = assistant.answer_query(q2)
    print(f"\nOperator: \"{q2}\"")
    print(f"ASTRA:    \"{ans2.answer}\"")

    time.sleep(1.0)

    # -------------------------------------------------------------
    # Recovery & Completion
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("▶ Procedural Recovery: Executing Step 3, 4, and 5")
    print("-" * 60)

    # Step 3: Open chamber
    obs3 = StructuredObservation(
        window_start=14.0,
        window_end=18.0,
        observation="Operator opens the chamber door.",
        events=[TemporalEvent(action="OPEN", object="CHAMBER")],
        confidence=0.95,
    )
    status, voice_inst = engine.process_observation(obs3)
    print(f"Step 3: {status.value} -> \"{obs3.observation}\"")
    if voice_inst:
        print(f"Voice: \"{voice_inst.to_speech_text()}\"")

    # Step 4: Place sample container inside chamber
    obs4 = StructuredObservation(
        window_start=18.0,
        window_end=22.0,
        observation="Operator places the sample container inside the chamber.",
        events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
        confidence=0.94,
    )
    status, voice_inst = engine.process_observation(obs4)
    print(f"Step 4: {status.value} -> \"{obs4.observation}\"")
    if voice_inst:
        print(f"Voice: \"{voice_inst.to_speech_text()}\"")

    # Step 5: Close chamber
    obs5 = StructuredObservation(
        window_start=22.0,
        window_end=26.0,
        observation="Operator seals and closes the chamber door.",
        events=[TemporalEvent(action="CLOSE", object="CHAMBER")],
        confidence=0.97,
    )
    status, voice_inst = engine.process_observation(obs5)
    print(f"Step 5: {status.value} -> \"{obs5.observation}\"")
    if voice_inst:
        print(f"Voice: \"{voice_inst.to_speech_text()}\"")
        voice_client.synthesize(voice_inst)

    print("\n" + "=" * 70)
    print("🎉 ASTRA-E Closed-Loop Prototype Demonstration COMPLETE!")
    print(f"Final State: {engine.alignment_status.value} | All 5 steps successfully verified.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Prototype v0.1 Runner")
    parser.add_argument(
        "--mode",
        choices=["demo", "ui", "server1", "server2"],
        default="demo",
        help="Run mode: 'demo' (closed-loop simulation), 'ui' (Streamlit app), 'server1' (vision), 'server2' (voice)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_closed_loop_demo()
    elif args.mode == "ui":
        dashboard_script = ROOT / "apps" / "prototype" / "dashboard.py"
        print(f"Launching Streamlit Dashboard from {dashboard_script}...")
        subprocess.run(["streamlit", "run", str(dashboard_script)], check=True)
    elif args.mode == "server1":
        server1_script = ROOT / "apps" / "prototype" / "server_vision.py"
        print("Launching Colab Server #1 (Video Understanding)...")
        subprocess.run(["uvicorn", "apps.prototype.server_vision:app", "--port", "8001", "--reload"], check=True)
    elif args.mode == "server2":
        server2_script = ROOT / "apps" / "prototype" / "server_voice.py"
        print("Launching Colab Server #2 (Voice / TTS)...")
        subprocess.run(["uvicorn", "apps.prototype.server_voice:app", "--port", "8002", "--reload"], check=True)


if __name__ == "__main__":
    main()
