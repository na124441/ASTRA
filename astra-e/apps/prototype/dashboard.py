"""ASTRA-E Prototype v0.1: Interactive Streamlit Dashboard (Section 12).

Displays:
  - Live Payload Camera Feed
  - Real-time Experiment Procedure Step Checklist (✓ / → / ○ / ⚠)
  - Current Observation from Video Understanding (Colab Server #1)
  - Alignment Status & Perceptual Confidence
  - Guidance Mode (ON/OFF) & Voice Mute Controls
  - Interactive Deviation Injectors for Testing All Scenarios
  - Grounded Conversational Q&A ('Ask ASTRA')
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astra.prototype.alignment_engine import TemporalAlignmentEngine
from astra.prototype.orchestrator import PrototypeOrchestrator
from astra.prototype.schemas import AssistanceMode, ProceduralAlignmentStatus, SeverityLevel

# Streamlit Page Setup
st.set_page_config(
    page_title="ASTRA-E Prototype v0.1 | BAS Assistance",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .status-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #1e293b;
    }
    .status-aligned {
        background-color: #064e3b;
        color: #6ee7b7;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .status-deviation {
        background-color: #78350f;
        color: #fde68a;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .status-critical {
        background-color: #7f1d1d;
        color: #fca5a5;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .status-completed {
        background-color: #1e3a8a;
        color: #93c5fd;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .step-item {
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.95rem;
    }
    .step-completed {
        background-color: #132f23;
        color: #34d399;
        border-left: 4px solid #10b981;
    }
    .step-active {
        background-color: #1e293b;
        color: #38bdf8;
        border-left: 4px solid #0284c7;
        font-weight: 600;
    }
    .step-pending {
        background-color: #0f172a;
        color: #94a3b8;
        border-left: 4px solid #475569;
    }
    .step-deviation {
        background-color: #451a03;
        color: #fbbf24;
        border-left: 4px solid #f59e0b;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_orchestrator(vision_url: str, voice_url: str) -> PrototypeOrchestrator:
    """Singleton instance of PrototypeOrchestrator across Streamlit reruns."""
    yaml_path = str(ROOT / "experiments" / "sample_transfer" / "procedure.yaml")
    orch = PrototypeOrchestrator(
        vision_server_url=vision_url,
        voice_server_url=voice_url,
        procedure_yaml_path=yaml_path,
        camera_index=0,
        guidance_mode=AssistanceMode.GUIDANCE,
        muted=False,
    )
    orch.start()
    return orch


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/space-shuttle.png", width=64)
    st.title("ASTRA-E Config")
    st.caption("Bharatiya Antariksh Station (BAS) Onboard Prototype")

    st.subheader("Cloud Microservices")
    vision_url = st.text_input("Server #1 (Vision)", value="http://localhost:8001")
    voice_url = st.text_input("Server #2 (Voice / TTS)", value="http://localhost:8002")

    orch = get_orchestrator(vision_url, voice_url)
    orch.update_urls(vision_url, voice_url)

    st.divider()
    st.subheader("Assistance Controls")
    guidance_enabled = st.toggle("Guidance Mode", value=True, help="Proactively prompts next procedural steps")
    muted = st.toggle("Mute Voice", value=False, help="Mute natural voice audio output")

    orch.set_guidance_mode(AssistanceMode.GUIDANCE if guidance_enabled else AssistanceMode.MONITOR)
    orch.set_muted(muted)

    if st.button("🔄 Reset Experiment Run", use_container_width=True):
        orch.reset_experiment()
        st.toast("Experiment reset to initial step.")

    st.divider()
    st.subheader("Procedural Simulation")
    st.caption("Simulate real astronaut actions or procedural violations:")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        if st.button("✓ Next Aligned Step", use_container_width=True):
            curr = orch.alignment_engine.current_step
            if curr:
                if curr.id == "S01":
                    orch.inject_simulation_event("step_1_pick")
                elif curr.id == "S02":
                    orch.inject_simulation_event("step_2_move")
                elif curr.id == "S03":
                    orch.inject_simulation_event("step_3_open")
                elif curr.id == "S04":
                    orch.inject_simulation_event("step_4_place")
                elif curr.id == "S05":
                    orch.inject_simulation_event("step_5_close")
            time.sleep(0.3)
            st.rerun()

    with col_sim2:
        if st.button("⚠ Inject Skipped Step", use_container_width=True):
            orch.inject_simulation_event("deviation_skipped_step")
            time.sleep(0.3)
            st.rerun()

    col_sim3, col_sim4 = st.columns(2)
    with col_sim3:
        if st.button("🚨 Wrong Object", use_container_width=True):
            orch.inject_simulation_event("deviation_wrong_object")
            time.sleep(0.3)
            st.rerun()

    with col_sim4:
        if st.button("⚠ Premature Close", use_container_width=True):
            orch.inject_simulation_event("deviation_premature_close")
            time.sleep(0.3)
            st.rerun()


# Main Dashboard Header
st.title("🛰️ ASTRA-E: Experiment Guidance & Verification")
state = orch.get_runtime_state()

# Top Banner Columns: Live Camera (Left) | Experiment Checklist (Right)
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    st.subheader("📹 Live Payload Camera")
    preview_frame = orch.get_latest_preview_frame()
    if preview_frame is not None:
        rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb, channels="RGB", use_container_width=True)
    else:
        st.info("Awaiting camera stream...")

    # Current Observation Callout
    st.markdown("##### 👁️ Current Temporal Observation (Colab Server #1)")
    st.info(f"**\"{state.latest_observation}\"**")

with col_right:
    st.subheader(f"📋 Experiment: {state.experiment_name}")
    st.caption(f"Protocol ID: {state.experiment_id} | Step {state.current_step_number} of {state.total_steps}")

    # Checklist rendering
    for step in state.steps:
        target_info = f" ➔ {step.target}" if step.target else ""
        obj_info = f" [{step.object}]" if step.object else ""

        if step.status == "completed":
            st.markdown(
                f"<div class='step-item step-completed'>✓ Step {step.step_number}: <b>{step.action}</b>{obj_info}{target_info}<br><small>{step.description}</small></div>",
                unsafe_allow_html=True,
            )
        elif step.status == "active":
            st.markdown(
                f"<div class='step-item step-active'>➔ Step {step.step_number}: <b>{step.action}</b>{obj_info}{target_info}<br><small>{step.description}</small></div>",
                unsafe_allow_html=True,
            )
        elif step.status == "deviation":
            st.markdown(
                f"<div class='step-item step-deviation'>⚠ Step {step.step_number}: <b>{step.action}</b>{obj_info}{target_info}<br><small>{state.last_deviation_message or step.description}</small></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='step-item step-pending'>○ Step {step.step_number}: <b>{step.action}</b>{obj_info}{target_info}<br><small>{step.description}</small></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    # Alignment Status Badge
    status_classes = {
        ProceduralAlignmentStatus.ALIGNED: "status-aligned",
        ProceduralAlignmentStatus.DEVIATION: "status-deviation",
        ProceduralAlignmentStatus.CRITICAL: "status-critical",
        ProceduralAlignmentStatus.COMPLETED: "status-completed",
    }
    badge_class = status_classes.get(state.alignment_status, "status-aligned")
    icon = "●" if state.alignment_status == ProceduralAlignmentStatus.ALIGNED else "⚠"
    if state.alignment_status == ProceduralAlignmentStatus.CRITICAL:
        icon = "🚨"
    elif state.alignment_status == ProceduralAlignmentStatus.COMPLETED:
        icon = "✓"

    st.markdown(
        f"<div class='status-card {badge_class}'>{icon} STATUS: {state.alignment_status.value}</div>",
        unsafe_allow_html=True,
    )

    # Confidence Metric
    st.progress(min(1.0, max(0.0, state.confidence)), text=f"Perceptual Confidence: {state.confidence * 100:.1f}%")

    if state.active_guidance_text:
        st.success(f"🔊 **Active Voice Prompt:** {state.active_guidance_text}")


st.divider()

# Bottom Section: Conversational Assistance ('Ask ASTRA')
st.subheader("💬 Ask ASTRA (Conversational Assistance)")
st.caption("Ask questions during the experiment. Answers are grounded directly in the deterministic experiment state machine.")

col_chips1, col_chips2, col_chips3 = st.columns(3)
quick_query = None
with col_chips1:
    if st.button("❓ 'What am I supposed to do now?'", use_container_width=True):
        quick_query = "What am I supposed to do now?"
with col_chips2:
    if st.button("❓ 'Did I do the last step correctly?'", use_container_width=True):
        quick_query = "Did I do the last step correctly?"
with col_chips3:
    if st.button("❓ 'What went wrong?'", use_container_width=True):
        quick_query = "What went wrong?"

user_query = st.text_input("Enter your question for ASTRA:", value=quick_query or "", placeholder="e.g. What is the next step?")

if user_query:
    answer = orch.answer_operator_query(user_query)
    st.markdown(
        f"""
        <div style='background-color:#1e293b; padding:16px; border-radius:8px; border-left:4px solid #38bdf8;'>
            <b>ASTRA:</b> "{answer}"
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_bytes = orch.get_latest_audio_bytes()
    if audio_bytes and not orch.muted:
        st.audio(audio_bytes, format="audio/wav")

# Auto-refresh helper toggle
st.caption("Live streaming active. Dashboard automatically synchronizes with background asynchronous worker threads.")
