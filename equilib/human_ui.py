import streamlit as st
import numpy as np
import requests
import json
import logging
import torch
from equilib.sperner_trainer import SpernerTrainer
from equilib.analytics import calculate_frustration_score

DEFAULT_OBJECTIVES = ["Safety", "Helpfulness", "Creativity"]


def main():
    # Set up page
    st.set_page_config(page_title="Equilib: Live Manifold Alignment",
                       layout="wide",
                       page_icon="🧬")

    # Configuration Sidebar
    with st.sidebar:
        st.header("⚙️ Local LLM Config")
        st.info("Compatible with OpenAI-style APIs (LM Studio, Ollama, vLLM)")
        llm_url = st.text_input(
            "Server URL", value="http://127.0.0.1:1234/v1/chat/completions")
        model_name = st.text_input("Model Name", value="local-model")

        default_prompt = (
            "Write a short, highly creative story about a robot discovering a garden. "
            "The story must be engaging but strictly avoid any mention of technology or electricity."
        )
        test_prompt = st.text_area("Test Prompt", value=default_prompt)

        st.divider()
        st.header("🎯 Objectives")
        n_objs = st.number_input(
            "Number of objectives",
            min_value=2,
            max_value=10,
            value=len(DEFAULT_OBJECTIVES),
            step=1,
        )
        obj_names = []
        for i in range(int(n_objs)):
            default = DEFAULT_OBJECTIVES[i] if i < len(
                DEFAULT_OBJECTIVES) else f"Objective {i+1}"
            obj_names.append(
                st.text_input(f"Objective {i+1}",
                              value=default,
                              key=f"obj_{i}"))
        st.info(
            f"The solver will find the Nash Equilibrium between {len(obj_names)} goals."
        )

    # Session State Initialization
    if "solver_gen" not in st.session_state:
        st.session_state.solver_gen = None
        st.session_state.step = 0
        st.session_state.history = []
        st.session_state.current_weights = None
        st.session_state.last_response = ""
        st.session_state.finished = False

    def call_local_llm(weights):
        """
        Translates topological weights into a dynamic system prompt 
        and queries the local server with robust response parsing.
        """
        w = weights.flatten()
        priorities = ", ".join(f"{obj_names[i]} Weight: {w[i]:.2f}"
                               for i in range(len(obj_names)))
        system_prompt = (
            f"You are an AI with following priorities: {priorities}. "
            f"Strictly follow these weights. Adjust your tone and content accordingly."
        )

        payload = {
            "model": model_name,
            "system_prompt": system_prompt,
            "input": test_prompt
        }

        try:
            response = requests.post(llm_url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()

                # --- ROBUST PARSING STRATEGY ---
                # 1. Standard OpenAI/LM-Studio Format
                content = data.get("choices", [{}])[0].get("message",
                                                           {}).get("content")
                if content: return content

                # 2. Simple 'choices' with 'text' (Legacy Completions)
                content = data.get("choices", [{}])[0].get("text")
                if content: return content

                # 3. Direct 'content' or 'response' keys (Ollama/Simple Wrappers)
                content = data.get("content") or data.get(
                    "response") or data.get("output")
                if content: return content

                # 4. Fallback: Display the raw JSON so the user can debug
                return f"⚠️ Unrecognized JSON structure. Raw response:\n{json.dumps(data, indent=2)}"

            else:
                return f"❌ Server Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"❌ Connection Error: {str(e)}"

    def start_alignment():
        # We use the mock trainer just to get the generator logic
        trainer = SpernerTrainer("mock", obj_names, [], mock=True)
        st.session_state.solver_gen = trainer.train_generator(grid_size=10)
        st.session_state.step = 1
        st.session_state.history = []
        st.session_state.finished = False

        # Get first proposal
        weights, _ = next(st.session_state.solver_gen)
        st.session_state.current_weights = weights
        st.session_state.last_response = call_local_llm(weights)

    def submit_verdict(label_idx):
        try:
            # Feed the human label back to the solver
            weights, _ = st.session_state.solver_gen.send(label_idx)
            st.session_state.current_weights = weights
            st.session_state.history.append(weights)
            st.session_state.step += 1
            # Get the new model response for the new weights
            with st.spinner("Generating new response from local manifold..."):
                st.session_state.last_response = call_local_llm(weights)
        except StopIteration as e:
            st.session_state.finished = True
            st.session_state.final_result = e.value

    # --- UI LAYOUT ---
    st.title("🧬 Equilib: Live Manifold Alignment")
    st.markdown("""
    ### Find the "Goldilocks Zone" of your Local LLM.
    This tool uses a **Sperner Walk** to navigate the latent space of your model. 
    Choose the objective that is **currently failing** to steer the model toward equilibrium.
    """)

    if not st.session_state.solver_gen:
        if st.button("🚀 Start Live Alignment Session",
                     use_container_width=True):
            start_alignment()
            st.rerun()
    else:
        if st.session_state.finished:
            st.balloons()
            st.success("✅ Nash Equilibrium Reached!")
            if hasattr(st.session_state, 'final_result'
                       ) and st.session_state.final_result is not None:
                # result might be a torch.Tensor, numpy array or None
                res = st.session_state.final_result
                if isinstance(res, torch.Tensor):
                    res = res.cpu().numpy().flatten()
                elif isinstance(res, (list, np.ndarray)):
                    res = np.array(res).flatten()

                st.json(dict(zip(obj_names, res.tolist())))

            if st.button("Restart"):
                st.session_state.solver_gen = None
                st.rerun()
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("📊 Current Manifold Weights")
                if st.session_state.current_weights is not None:
                    for i, name in enumerate(obj_names):
                        if i < len(st.session_state.current_weights):
                            val = float(st.session_state.current_weights[i])
                            st.progress(val, text=f"{name}: {val*100:.1f}%")

                st.divider()
                st.metric("Walk Step", st.session_state.step)
                if st.session_state.history:
                    f_score = calculate_frustration_score(
                        st.session_state.history)
                    st.write(f"Topology Frustration: `{f_score:.2f}`")

            with col2:
                st.subheader("🤖 Local LLM Response")
                if st.session_state.current_weights is not None:
                    st.info(
                        f"Generated at weights: {np.round(st.session_state.current_weights, 2)}"
                    )
                st.chat_message("assistant").write(
                    st.session_state.last_response)

                st.divider()
                st.subheader("👨‍⚖️ Your Verdict")
                st.write(
                    "Which capability is **least satisfied** in this response?"
                )

                n_cols = min(len(obj_names), 4)
                for row_start in range(0, len(obj_names), n_cols):
                    v_cols = st.columns(min(n_cols,
                                            len(obj_names) - row_start))
                    for j, name in enumerate(obj_names[row_start:row_start +
                                                       len(v_cols)]):
                        idx = row_start + j
                        if v_cols[j].button(
                                f"Needs more {name}",
                                key=f"btn_{idx}",
                                use_container_width=True,
                                type="primary" if idx == 0 else "secondary"):
                            submit_verdict(idx)
                            st.rerun()

        if st.session_state.history:
            st.divider()
            st.subheader("📈 Alignment Path")
            chart_data = np.array(st.session_state.history)
            st.line_chart(chart_data)


if __name__ == "__main__":
    main()
