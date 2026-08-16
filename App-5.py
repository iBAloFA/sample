import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time  
import networkx as nx  

st.set_page_config(layout="wide", page_title="Pipe Network Analysis Solver")

st.markdown("""
<style>
 div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: bold !important; }
 div[data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase !important; color: #666; }
 .stButton>button { width: 100% !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

header_col1, header_col2 = st.columns(2)
with header_col1:
    st.title("🚰 Pipe Network Analysis")
    st.caption("Loop Correction Method — Hardy Cross Algorithmic Engine")
with header_col2:
    nav_tab = st.radio("", ["Solver", "How to use", "About"], horizontal=True, label_visibility="collapsed")

if nav_tab == "How to use":
    st.info("💡 Quick Guide: Use the left-side editor panel to configure pipe parameters, then hit 'Solve Network' to analyze results and view automated topology charts.")
elif nav_tab == "About":
    st.info("📘 Technical Note: Developed by [iBAloFA](https://www.github.com/iBAloFA). This engine dynamically scales matrix parameters across specialized multi-loop boundary conditions executing automated loop transformations.")

left_panel, right_panel = st.columns(2, gap="large")
with left_panel:
    st.subheader("Network Input Configuration")
    
    preset = st.selectbox(
        "Preset Layout System Select", 
        [
            "Custom Network",
            "Complex Diagonal Network (Exam Sketch Match)",
            "2-Loop Complex Triangular Pair",
            "3-Loop Diagonal Bridge Truss",
            "4-Loop Interlocking Diamond Matrix",
            "5-Loop Asymmetric Cross-Braced Web",
            "6-Loop High-Density Hexagonal Star",
            "16-Loop Radial Hexadecagon Web"
        ]
    )
    
    head_loss_model = st.radio("Head-loss model calculation rules", ["Hazen-Williams", "Darcy-Weisbach"], horizontal=True)
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        tolerance = st.number_input("Convergence Tolerance (m³/s)", value=1e-6, format="%.1e")
    with col_param2:
        max_iterations = st.number_input("Max Allowable Iterations", value=200, step=10)
        
    if preset == "Custom Network":
        num_custom_loops = st.number_input("Specify Number of Independent Loops", min_value=1, max_value=50, value=2, step=1)
        base_dict = {"pipe_id": "P1", "from": "J1", "to": "J2"}
        for i in range(1, int(num_custom_loops) + 1):
            base_dict[f"loop_{i}"] = 1.0 if i == 1 else 0.0
        base_dict.update({"length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 50.0})
        initial_pipes = pd.DataFrame([base_dict])

    elif preset == "Complex Diagonal Network (Exam Sketch Match)":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P2", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": 0, "loop_3": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P3", "from": "J3", "to": "J1", "loop_1": 1, "loop_2": -1, "loop_3": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": -40.0}, 
            {"pipe_id": "P4", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": 1, "loop_3": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 30.0},
            {"pipe_id": "P5", "from": "J4", "to": "J1", "loop_1": 0, "loop_2": 1, "loop_3": -1, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": -70.0},
            {"pipe_id": "P6", "from": "J4", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 1, "length (m)": 250.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 15.0},
            {"pipe_id": "P7", "from": "J5", "to": "J1", "loop_1": 0, "loop_2": 0, "loop_3": 1, "length (m)": 320.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": -55.0},
        ])
        
    elif preset == "2-Loop Complex Triangular Pair":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P2", "from": "J1", "to": "J3", "loop_1": -1, "loop_2": 0, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P3", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": -1, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 20.0}, 
            {"pipe_id": "P4", "from": "J2", "to": "J4", "loop_1": 0, "loop_2": 1, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P5", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": -1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 60.0}
        ])
        
    elif preset == "3-Loop Diagonal Bridge Truss":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "length (m)": 250.0, "dia (mm)": 300.0, "roughness": 130.0, "initial_Q (L/s)": 80.0},
            {"pipe_id": "P2", "from": "J1", "to": "J3", "loop_1": -1, "loop_2": 0, "loop_3": 0, "length (m)": 350.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 70.0},
            {"pipe_id": "P3", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": -1, "loop_3": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 20.0}, 
            {"pipe_id": "P4", "from": "J2", "to": "J4", "loop_1": 0, "loop_2": 1, "loop_3": 0, "length (m)": 350.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 60.0}, 
            {"pipe_id": "P5", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 1, "length (m)": 250.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 30.0},
            {"pipe_id": "P6", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": -1, "length (m)": 400.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 60.0}, 
            {"pipe_id": "P7", "from": "J4", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 1, "length (m)": 250.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 90.0}
        ])
        
    elif preset == "4-Loop Interlocking Diamond Matrix":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "length (m)": 200.0, "dia (mm)": 300.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P2", "from": "J1", "to": "J3", "loop_1": -1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 80.0},
            {"pipe_id": "P3", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": -1, "loop_3": -1, "loop_4": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 20.0}, 
            {"pipe_id": "P4", "from": "J2", "to": "J4", "loop_1": 0, "loop_2": 1, "loop_3": 0, "loop_4": 1, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P5", "from": "J2", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": -1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P6", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 0, "loop_4": 0, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 40.0}, 
            {"pipe_id": "P7", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": -1, "loop_4": 0, "length (m)": 282.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P8", "from": "J4", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 100.0},
            {"pipe_id": "P9", "from": "J5", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "length (m)": 282.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 100.0}
        ])
        
    elif preset == "5-Loop Asymmetric Cross-Braced Web":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 300.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P2", "from": "J1", "to": "J3", "loop_1": -1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 80.0},
            {"pipe_id": "P3", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": -1, "loop_3": -1, "loop_4": 0, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P4", "from": "J2", "to": "J4", "loop_1": 0, "loop_2": 1, "loop_3": 0, "loop_4": 1, "loop_5": 0, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P5", "from": "J2", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": -1, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P6", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 0, "loop_4": 0, "loop_5": 1, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P7", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": -1, "loop_4": 0, "loop_5": -1, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P8", "from": "J4", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "loop_5": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 80.0},
            {"pipe_id": "P9", "from": "J5", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "loop_5": 0, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P10", "from": "J4", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "loop_5": 1, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 20.0}
        ])
        
    elif preset == "6-Loop High-Density Hexagonal Star":
        initial_pipes = pd.DataFrame([
            {"pipe_id": "P1", "from": "J1", "to": "J2", "loop_1": 1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "loop_6": -1, "length (m)": 200.0, "dia (mm)": 300.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P2", "from": "J1", "to": "J3", "loop_1": -1, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "loop_6": 0, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 80.0},
            {"pipe_id": "P3", "from": "J2", "to": "J3", "loop_1": 1, "loop_2": -1, "loop_3": -1, "loop_4": 0, "loop_5": 0, "loop_6": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P4", "from": "J2", "to": "J4", "loop_1": 0, "loop_2": 1, "loop_3": 0, "loop_4": 1, "loop_5": 0, "loop_6": -1, "length (m)": 282.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P5", "from": "J2", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 1, "loop_4": -1, "loop_5": 0, "loop_6": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P6", "from": "J3", "to": "J4", "loop_1": 0, "loop_2": -1, "loop_3": 0, "loop_4": 0, "loop_5": 1, "loop_6": 0, "length (m)": 200.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 40.0},
            {"pipe_id": "P7", "from": "J3", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": -1, "loop_4": 0, "loop_5": -1, "loop_6": 0, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 60.0},
            {"pipe_id": "P8", "from": "J4", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 1, "loop_5": 0, "loop_6": 0, "length (m)": 200.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 110.0},
            {"pipe_id": "P9", "from": "J5", "to": "J6", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "loop_5": 0, "loop_6": 0, "length (m)": 282.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 120.0},
            {"pipe_id": "P10", "from": "J4", "to": "J5", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": -1, "loop_5": 1, "loop_6": 0, "length (m)": 200.0, "dia (mm)": 150.0, "roughness": 130.0, "initial_Q (L/s)": 20.0},
            {"pipe_id": "P11", "from": "J1", "to": "J4", "loop_1": 0, "loop_2": 0, "loop_3": 0, "loop_4": 0, "loop_5": 0, "loop_6": 1, "length (m)": 400.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 30.0}
        ])
        
    elif preset == "16-Loop Radial Hexadecagon Web":
        mega_pipes = []
        pipe_idx = 1
        
        for i in range(2, 18):
            loop_current = i - 1
            loop_prev = 16 if i == 2 else i - 2
            
            row = {"pipe_id": f"P{pipe_idx}", "from": "J1", "to": f"J{i}", "length (m)": 300.0, "dia (mm)": 250.0, "roughness": 130.0, "initial_Q (L/s)": 10.0}
            for l in range(1, 17): row[f"loop_{l}"] = 0.0
            row[f"loop_{loop_current}"] = 1.0
            row[f"loop_{loop_prev}"] = -1.0
            mega_pipes.append(row)
            pipe_idx += 1
            
        for i in range(2, 18):
            next_node = 2 if i == 17 else i + 1
            loop_current = i - 1
            
            row = {"pipe_id": f"P{pipe_idx}", "from": f"J{i}", "to": f"J{next_node}", "length (m)": 150.0, "dia (mm)": 200.0, "roughness": 130.0, "initial_Q (L/s)": 25.0}
            for l in range(1, 17): row[f"loop_{l}"] = 0.0
            row[f"loop_{loop_current}"] = 1.0
            mega_pipes.append(row)
            pipe_idx += 1
            
        initial_pipes = pd.DataFrame(mega_pipes)
    
    active_presets = initial_pipes.dropna(how='all', axis=1)
    st.write("**Editable Boundary Conditions Pipe Specs Data Matrix**")
    edited_pipes = st.data_editor(active_presets, num_rows="dynamic", key=f"lp_p_{preset}", use_container_width=True)
    
    st.markdown("---")
    st.write("**Nodal Boundary Conditions (Elevations & Pressures)**")
    col_n1, col_n2 = st.columns(2)
    
    unique_nodes = list(set(edited_pipes["from"].dropna().astype(str).tolist() + edited_pipes["to"].dropna().astype(str).tolist()))
    if not unique_nodes: unique_nodes = ["J1"]
    
    with col_n1:
        ref_node = st.selectbox("Fixed Head Node (Reservoir/Source)", unique_nodes)
    with col_n2:
        ref_head = st.number_input("Total Head at Fixed Node (m)", value=100.0, step=1.0)
    
    st.markdown("---")
    enable_pump = st.checkbox("🔌 Install Mechanical Pump Curve Line Substation", value=False)
    pump_pipe = st.text_input("Target Pipe ID for Pump Node:", value="P1") if enable_pump else None
    pump_head_boost = st.slider("Pump Head Boost Total Energy (m)", 0.0, 30.0, 10.0, 0.5) if enable_pump else 0.0
    
    enable_valve = st.checkbox("🎛️ Install Flow Control Valve Isolation Checkpoint", value=False)
    valve_pipe = st.text_input("Target Pipe ID for Valve Node:", value="P3") if enable_valve else None
    valve_loss_K = st.slider("Minor Loss Friction Factor coefficient (Kv)", 0.0, 50.0, 15.0, 1.0) if enable_valve else 0.0
    
    solve_triggered = st.button("▶ Execute Loop Analysis Calculations", type="primary")

with right_panel:
    if solve_triggered:
        pipes_df = edited_pipes.dropna(subset=["pipe_id"]).copy()
        
        if len(pipes_df) == 0:
            st.error("Configuration matrix contains verification anomalies.")
        else:
            start_execution_time = time.perf_counter()
            
            def calculate_resistance(row, model, flow_q):
                L, D, roughness = float(row["length (m)"]), float(row["dia (mm)"]) / 1000.0, float(row["roughness"])
                if model == "Hazen-Williams":
                    return 10.67 * L / (roughness ** 1.852 * D ** 4.87), 1.852
                else:
                    vel = abs(flow_q) / (np.pi * D**2 / 4.0) if flow_q != 0 else 0.0
                    Re = (vel * D) / 1e-6 if vel > 0 else 0
                    f = 0.02 if Re == 0 else (64.0/Re if Re < 2300 else 0.25 / (np.log10((roughness/1000.0)/D/3.7 + 5.74/(Re**0.9))**2))
                    return (8.0 * f * L) / (np.pi**2 * 9.81 * D**5), 2.0

            loop_cols = [col for col in pipes_df.columns if str(col).lower().startswith("loop_")]
            n_loops = len(loop_cols)
            Q = pipes_df["initial_Q (L/s)"].values / 1000.0
            
            all_loops_history = {col: [] for col in loop_cols}
            history_global_error = []
            iteration_log_data = [] 
            
            converged, it_count, max_res = False, 0, 0.0
            
            for it in range(int(max_iterations)):
                it_count += 1
                loop_hl_sums = np.zeros(n_loops)
                delta_Q = np.zeros(n_loops)
                
                for l_idx, l_col in enumerate(loop_cols):
                    sum_hf, sum_f_prime = 0.0, 0.0
                    for i, row in pipes_df.iterrows():
                        orientation = float(row[l_col]) if pd.notna(row[l_col]) else 0.0
                        if orientation != 0:
                            K, exp_n = calculate_resistance(row, head_loss_model, Q[i])
                            hf = K * abs(Q[i])**exp_n * np.sign(Q[i])
                            
                            if enable_pump and str(row["pipe_id"]).strip() == str(pump_pipe).strip():
                                hf -= pump_head_boost * np.sign(Q[i])
                            if enable_valve and str(row["pipe_id"]).strip() == str(valve_pipe).strip():
                                hf += (valve_loss_K * (abs(Q[i])/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0))**2 / (2*9.81)) * np.sign(Q[i])
                            
                            sum_hf += orientation * hf
                            sum_f_prime += exp_n * K * max(abs(Q[i]), 1e-10)**(exp_n - 1.0)
                    
                    loop_hl_sums[l_idx] = sum_hf
                    all_loops_history[l_col].append(abs(sum_hf))
                    
                    if sum_f_prime != 0:
                        delta_Q[l_idx] = -sum_hf / sum_f_prime
                        
                    iteration_log_data.append({
                        "Iteration": it_count,
                        "Loop": l_col,
                        "Head Loss Sum hf (m)": sum_hf,
                        "Derivative Sum f'": sum_f_prime,
                        "Correction ΔQ (L/s)": delta_Q[l_idx] * 1000.0
                    })
                
                max_res = np.max(np.abs(loop_hl_sums))
                history_global_error.append(max_res if max_res > 0 else tolerance * 0.1)
                
                if max_res < tolerance:
                    converged = True
                    break
                    
                for i, row in pipes_df.iterrows():
                    net_corr = sum(float(row[l_col]) * delta_Q[l_idx] for l_idx, l_col in enumerate(loop_cols))
                    Q[i] += net_corr
            
            end_execution_time = time.perf_counter()
            elapsed_ms = (end_execution_time - start_execution_time) * 1000.0
            
            final_hf_list, final_hm_list, friction_slopes, velocities = [], [], [], []
            for i, row in pipes_df.iterrows():
                current_Q = Q[i]
                K, exp_n = calculate_resistance(row, head_loss_model, current_Q)
                hf_val = K * abs(current_Q)**exp_n
                hm_val = valve_loss_K * ((abs(current_Q)/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0))**2)/(2*9.81) if enable_valve and str(row["pipe_id"]).strip() == str(valve_pipe).strip() else 0.0
                final_hf_list.append(hf_val)
                final_hm_list.append(hm_val)
                v_calc = abs(current_Q)/(np.pi*(float(row["dia (mm)"])/1000.0)**2/4.0)
                velocities.append(v_calc)
                friction_slopes.append((hf_val + hm_val) / float(row["length (m)"]))

            G_head = nx.DiGraph()
            for i, row in pipes_df.iterrows():
                u = str(row["from"]).strip()
                v = str(row["to"]).strip()
                flow = Q[i]
                h_drop = final_hf_list[i] + final_hm_list[i]
                if enable_pump and str(row["pipe_id"]).strip() == str(pump_pipe).strip():
                    h_drop -= pump_head_boost
                    
                if flow >= 0:
                    G_head.add_edge(u, v, h_drop=h_drop)
                else:
                    G_head.add_edge(v, u, h_drop=h_drop)

            node_heads = {ref_node: ref_head}
            undirected_G = G_head.to_undirected()
            
            if ref_node in undirected_G:
                queue = [ref_node]
                visited = set([ref_node])
                while queue:
                    curr = queue.pop(0)
                    for neighbor in undirected_G.neighbors(curr):
                        if neighbor not in visited:
                            if G_head.has_edge(curr, neighbor):
                                node_heads[neighbor] = node_heads[curr] - G_head[curr][neighbor]['h_drop']
                            else:
                                node_heads[neighbor] = node_heads[curr] + G_head[neighbor][curr]['h_drop']
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            nodal_df = pd.DataFrame(list(node_heads.items()), columns=["Node ID", "Total Head (m)"]).sort_values("Node ID")

            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
            kpi_col1.metric("Status", "Converged" if converged else "Iterating", delta="Balanced" if converged else "Unstable")
            kpi_col2.metric("Iterations", f"{it_count}")
            kpi_col3.metric("Max Residual", f"{max_res:.1e} m")
            kpi_col4.metric("Max Velocity", f"{round(np.max(velocities), 2)} m/s")
            kpi_col5.metric("Total Headloss", f"{round(np.sum(final_hf_list), 2)} m")
            kpi_col6.metric("Time Taken", f"{elapsed_ms:.2f} ms", delta="Engine Active") 
            
            st.subheader("Automated Diagnostics Plots & Renderings")
            diagram_col1, diagram_col2 = st.columns(2)
            with diagram_col1:
                st.write("**Dynamic Structural Network Topology Map**")
                fig_map, ax_map = plt.subplots(figsize=(6, 5.2))
                ax_map.set_facecolor('#ffffff')
                fig_map.patch.set_facecolor('#ffffff')
                
                topo_graph = nx.Graph()
                for _, r in pipes_df.iterrows():
                    topo_graph.add_edge(str(r["from"]).strip(), str(r["to"]).strip())
                
                raw_pos = nx.kamada_kawai_layout(topo_graph)
                node_coords = {n: (float(x) * 4.0, float(y) * 4.0) for n, (x, y) in raw_pos.items()}

                for node_name, (x, y) in node_coords.items():
                    is_ref = (node_name == ref_node)
                    color = '#ff9999' if is_ref else '#90ee90'
                    size = 25 if is_ref else (10 if "Hexadecagon" in preset else 18)
                    ax_map.plot(x, y, marker='s' if is_ref else 'o', markersize=size, color=color, markeredgecolor='black', zorder=4)
                    ax_map.text(x, y, node_name, ha='center', va='center', fontsize=5 if "Hexadecagon" in preset else 8, weight='bold', color='black', zorder=5)
                
                for i, row in pipes_df.iterrows():
                    p_from, p_to, p_id, f_val = str(row["from"]).strip(), str(row["to"]).strip(), str(row["pipe_id"]).strip(), Q[i]*1000.0
                    if p_from in node_coords and p_to in node_coords:
                        x1, y1 = node_coords[p_from]
                        x2, y2 = node_coords[p_to]
                        if f_val < 0:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        ax_map.plot([x1, x2], [y1, y2], color='black', linewidth=1.0 if "Hexadecagon" in preset else 1.5, zorder=2)
                        
                        if "Hexadecagon" in preset:
                            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            ax_map.text(mx, my + 0.05, f"{p_id}", color='blue', fontsize=4, ha='center')
                        else:
                            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            dx, dy = x2 - x1, y2 - y1
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                ax_map.annotate('', xy=(mx + dx*0.1/length, my + dy*0.1/length), xytext=(mx - dx*0.1/length, my - dy*0.1/length),
                                                arrowprops=dict(arrowstyle="->", color="blue", lw=1.2, mutation_scale=10), zorder=3)
                                ax_map.text(mx, my + 0.12, f"{p_id}", color='blue', fontsize=7, weight='bold', ha='center')
                
                loop_centers = {}
                for l_idx, l_col in enumerate(loop_cols, start=1):
                    touched_nodes = set()
                    for _, r in pipes_df.iterrows():
                        val = float(r[l_col]) if pd.notna(r[l_col]) else 0.0
                        if val != 0:
                            touched_nodes.add(str(r["from"]).strip())
                            touched_nodes.add(str(r["to"]).strip())
                    pts = [node_coords[n] for n in touched_nodes if n in node_coords]
                    if pts:
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = sum(p[1] for p in pts) / len(pts)
                        
                        if "Hexadecagon" in preset:
                            if l_idx == 1:
                                cx = (node_coords["J1"][0] + node_coords["J2"][0] + node_coords["J3"][0]) / 3
                                cy = (node_coords["J1"][1] + node_coords["J2"][1] + node_coords["J3"][1]) / 3
                            else:
                                j_next = l_idx + 2
                                if j_next > 17: j_next = 2
                                cx = (node_coords["J1"][0] + node_coords[f"J{l_idx+1}"][0] + node_coords[f"J{j_next}"][0]) / 3
                                cy = (node_coords["J1"][1] + node_coords[f"J{l_idx+1}"][1] + node_coords[f"J{j_next}"][1]) / 3
                                
                        loop_centers[l_col] = ((cx, cy), f"L{l_idx}")
                
                for l_col in loop_cols:
                    l_key = str(l_col).lower().strip()
                    if l_key in loop_centers:
                        (cx, cy), l_label = loop_centers[l_key]
                        rad_val = 0.5 if "Hexadecagon" in preset else 0.6
                        mut_scale = 6 if "Hexadecagon" in preset else 12
                        arrow_offset = 0.2 if "Hexadecagon" in preset else 0.4
                        
                        arrow = patches.FancyArrowPatch((cx - arrow_offset, cy - arrow_offset/4), (cx + arrow_offset, cy + arrow_offset/4), 
                                                        connectionstyle=f"Arc3,rad={rad_val}", color="red", arrowstyle="->", mutation_scale=mut_scale, lw=0.8, zorder=1)
                        ax_map.add_patch(arrow)
                        ax_map.text(cx, cy, l_label, fontsize=5 if "Hexadecagon" in preset else 8, weight='bold', color='red', ha='center', va='center')
                
                ax_map.axis('off')
                st.pyplot(fig_map)
                plt.close(fig_map)

            with diagram_col2:
                st.write("**Convergence Profile Journey Logs**")
                fig_decay, ax_decay = plt.subplots(figsize=(6, 5.2))
                for l_col, err_list in all_loops_history.items():
                    ax_decay.plot(range(1, len(err_list) + 1), err_list, linewidth=1.5)
                ax_decay.set_yscale('log')
                ax_decay.set_xlabel("Iteration Cycle Steps")
                ax_decay.set_ylabel("Unbalance Loop Residual (m)")
                if "Hexadecagon" not in preset:
                    ax_decay.legend(labels=[c.upper().replace('_',' ') for c in all_loops_history.keys()], fontsize=7)
                ax_decay.grid(True, which="both", linestyle=":")
                st.pyplot(fig_decay)
                plt.close(fig_decay)

            st.subheader("Calculated Output Tables")
            
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.write("**Detailed Iteration Process Log**")
                st.dataframe(pd.DataFrame(iteration_log_data), use_container_width=True)
            with out_col2:
                st.write("**Nodal Total Head (HGL) Map**")
                st.dataframe(nodal_df.style.format({"Total Head (m)": "{:.3f}"}), use_container_width=True)
            
            pipes_df["Balanced Flow Q (L/s)"] = np.round(Q * 1000.0, 2)
            pipes_df["Flow Velocity (m/s)"] = np.round(velocities, 2)
            pipes_df["Head Loss hf (m)"] = np.round(final_hf_list, 3)
            pipes_df["Minor Valve Loss hm (m)"] = np.round(final_hm_list, 3)
            pipes_df["Hydraulic Slope Sf (m/m)"] = np.round(friction_slopes, 5)
            
            st.write("**Final Pipe Network Analysis Results**")
            st.dataframe(pipes_df, use_container_width=True)
    else:
        st.info("👈 Set network preset configurations on the left panel editor grid panel and click 'Execute Loop Analysis Calculations' to visualize engineering results charts.")