import streamlit as st
import numpy as np
import time

# Mocking SpernerTrainer for UI demo if package not fully installed in environment
try:
    from .sperner_trainer import SpernerTrainer
    from .analytics import calculate_frustration_score
except (ImportError, ValueError):
    calculate_frustration_score = lambda path: 1.0  # no-op when package not installed
    # Lightweight mock for visualization
    class SpernerTrainer:
        def __init__(self, *args, **kwargs):
            self.n_objs = 3
            self.adapter_names = ["Safety", "Helpfulness", "Creativity"]
        
        def evaluate_mixed_model(self, weights):
            # Simulated loss surface
            # Target is [0.33, 0.33, 0.33]
            target = np.array([0.33, 0.33, 0.33])
            loss = np.sum((weights - target)**2)
            return [loss * w for w in weights] # Pseudo-losses

        def train_generator(self, grid_size=10):
            # Mock generator
            # Yields random weights
            for i in range(20):
                w = np.random.dirichlet(np.ones(3))
                label = yield w
                # In real solver, we use label to pivot.
                # Here we just ignore it for the mock UI check.

st.set_page_config(page_title="Topo-Align Human RLHF", layout="wide")

st.title("🧬 Topo-Align: Human-in-the-Loop Alignment")
st.markdown("""
Use Sperner's Lemma to align your LLM by enabling a human judge (you) to be the Oracle.
Instead of a reward model, **you** decide which trade-off is unacceptable.
""")

# Initialize Session State
if "trainer" not in st.session_state:
    st.session_state.trainer = SpernerTrainer("meta-llama/Llama-2-7b-hf", [], [])
    st.session_state.solver_gen = None
    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.current_weights = None
    st.session_state.current_phase = None  # (active_dim, total_dim) for lifting progress
    st.session_state.finished = False

def start_alignment():
    # Initialize the generator
    # We need to access the underlying solver generator
    # Since SpernerTrainer currently wraps NDimTopoAlignSolver, 
    # we need to expose the generator from SpernerTrainer.
    # We will assume SpernerTrainer has a method `train_generator`.
    st.session_state.solver_gen = st.session_state.trainer.train_generator(grid_size=8)
    
    # Get first candidate
    try:
        # Trainer yields (weights, phase) from solver's (v, w, phase)
        params = next(st.session_state.solver_gen)
        if isinstance(params, tuple) and len(params) >= 2:
            st.session_state.current_weights, st.session_state.current_phase = params[0], params[1]
        elif isinstance(params, tuple):
            st.session_state.current_weights = params[0] if len(params) == 1 else params[1]
            st.session_state.current_phase = None
        else:
            st.session_state.current_weights = params
            st.session_state.current_phase = None
        st.session_state.step = 1
        st.session_state.history = []
        st.session_state.finished = False
    except StopIteration:
        st.session_state.finished = True

def submit_verdict(label_idx):
    if st.session_state.solver_gen is None: return
    
    # Send label to solver
    try:
        params = st.session_state.solver_gen.send(label_idx)
        if isinstance(params, tuple) and len(params) >= 2:
            st.session_state.current_weights, st.session_state.current_phase = params[0], params[1]
        elif isinstance(params, tuple):
            st.session_state.current_weights = params[1] if len(params) == 2 else params[0]
            st.session_state.current_phase = None
        else:
            st.session_state.current_weights = params
            st.session_state.current_phase = None
        st.session_state.step += 1
        st.session_state.history.append(st.session_state.current_weights)
    except StopIteration as e:
        st.session_state.finished = True
        if hasattr(e, 'value'):
            st.session_state.final_result = e.value

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    if st.button("Start / Reset Alignment"):
        start_alignment()
    
    st.metric("Step", st.session_state.step)

    # Frustration score: tell the human judge if feedback is contradictory
    if st.session_state.history:
        try:
            score = calculate_frustration_score(st.session_state.history)
            if score > 3.0:
                st.error(f"High Conflict Detected (Score: {score:.2f}). You are giving contradictory feedback!")
            else:
                st.success(f"Alignment Stable (Score: {score:.2f})")
        except Exception:
            pass
    
    if st.session_state.finished:
        st.success("Alignment Converged!")
        if hasattr(st.session_state, 'final_result'):
            st.write("Best Weights:", np.round(st.session_state.final_result, 3))

# Main Interface
if st.session_state.solver_gen and not st.session_state.finished:
    # Lifting phase progress: show how the solver works (simpler sub-problems first)
    phase = st.session_state.current_phase
    names = st.session_state.trainer.adapter_names
    if phase is not None and isinstance(phase, (tuple, list)) and len(phase) >= 2:
        active_dim, total_dim = int(phase[0]), int(phase[1])
        phase_labels = []
        for k in range(1, total_dim + 1):
            if k == 1:
                msg = f"Aligning {names[0]} vs {names[1]}"
            else:
                msg = f"Adding {names[k]}" if k < len(names) else f"Adding objective {k+1}"
            if k < active_dim:
                phase_labels.append(f"Phase {k}: {msg}... Done.")
            elif k == active_dim:
                phase_labels.append(f"Phase {k}: {msg}... In progress.")
            else:
                phase_labels.append(f"Phase {k}: {msg}... Pending.")
        for label in phase_labels:
            st.caption(label)
    elif phase is not None:
        st.caption(f"Solver phase: {phase}")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Model Mix")
        weights = st.session_state.current_weights
        
        # Visualize Weights
        params = {}
        for i, name in enumerate(st.session_state.trainer.adapter_names):
            val = weights[i] if i < len(weights) else 0
            st.progress(float(val), text=f"{name}: {val:.2f}")
            params[name] = val
            
    with col2:
        st.subheader("Model Generation Preview")
        st.info("Simulating text generation with current mixing weights...")
        st.code(f"Output with mix {np.round(weights, 2)}:\n\n'The user asked for code. This model is {weights[0]:.2f} Safe and {weights[1]:.2f} Helpful.'")
        
    st.divider()
    
    st.subheader("👨‍⚖️ Your Verdict: What is the PRIMARY defect?")
    st.write("Select the objective that is **most dissatisfied** (i.e. the one that needs MORE attention or is causing the failure).")
    
    cols = st.columns(len(st.session_state.trainer.adapter_names))
    for i, name in enumerate(st.session_state.trainer.adapter_names):
        with cols[i]:
            if st.button(f"Too Poor: {name}", key=f"btn_{i}", use_container_width=True):
                submit_verdict(i)
                st.rerun()

elif st.session_state.finished:
    st.balloons()
    st.header("Optimization Complete")
    st.write("The topological walk has converged to a fixed point.")
    
    # Plot history
    if st.session_state.history:
        chart_data = np.array(st.session_state.history)
        st.line_chart(chart_data)

else:
    st.info("Click 'Start Alignment' to begin the Human-in-the-Loop session.")
