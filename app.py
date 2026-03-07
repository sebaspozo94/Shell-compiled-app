import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import logic  # Ensure logic.so is in the same directory
import plotly.graph_objects as go
import io
import stl              
from stl import mesh 
from scipy.spatial import Delaunay
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Define the custom Matplotlib colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ['#cbd5e1', '#2563eb', '#08306b'])

st.set_page_config(page_title="Shell Topology Opt", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem; }
    .tag-container { display: flex; gap: 10px; margin-bottom: 1.5rem; }
    .tag { background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; border: 1px solid #e2e8f0; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PART 1: HEADER & OBJECTIVE
# ==========================================
st.markdown('<div class="main-header">Shell Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="tag-container"><span class="tag">Optimization</span><span class="tag">Shell</span><span class="tag">FEA Engine</span></div>', unsafe_allow_html=True)

with st.expander("🎯 App Objective", expanded=False):
    st.markdown("""
    **Objective:** Distribute a constant amount of material to maximize the stiffness of a shell-type structure 
    under external distributed load and self-weight.
    """)

# --- SETUP SESSION STATE ---
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
    st.session_state.history = None
    st.session_state.X = None
    st.session_state.Y = None

if "bc_df" not in st.session_state:
    st.session_state.bc_df = pd.DataFrame(
        [[48.0, 156.0, 4.0, 4.0, "Pinned"], [48.0, 36.0, 4.0, 4.0, "Pinned"], [192.0, 156.0, 4.0, 4.0, "Pinned"], [192.0, 36.0, 4.0, 4.0, "Pinned"]],
        columns=["X (in)", "Y (in)", "Width", "Height", "Type"]
    )

# --- SIDEBAR (Materials & Loads) ---
with st.sidebar:
    st.header("🧪 Material Properties")
    E = st.number_input("Elastic Modulus (psi)", value=1500000, step=100000)
    nu = st.slider("Poisson's Ratio (v)", 0.0, 0.5, 0.30)
    rho = st.number_input("Material Density (p)", value=0.010, format="%.3f")
    self_weight = st.checkbox("Include Self-Weight", value=True)

    st.header("⚖️ Loads")
    w_u = st.number_input("Distributed Load (w_u)", value=0.2778)

# ==========================================
# PART 2: MODEL CONFIGURATION (DROP MENUS)
# ==========================================
st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)
conf_col1, conf_col2, conf_col3 = st.columns(3)

with conf_col1:
    with st.expander("📏 Domain & Mesh", expanded=False):
        dimx = st.number_input("Domain X (in)", value=240, step=4, min_value=1)
        dimy = st.number_input("Domain Y (in)", value=192, step=4, min_value=1)
        nelx = st.number_input("Elements X", value=120, step=4, min_value=1, max_value=150)
        nely = st.number_input("Elements Y", value=96, step=4, min_value=1, max_value=150)

with conf_col2:
    with st.expander("🎯 Optimization Settings", expanded=False):
        vol_frac = st.slider("Volume Fraction", 0.05, 1.0, 0.3)
        rmin = st.number_input("Filter Radius (rmin)", value=5.0, step=1.0)
        itmax = st.number_input("Max Iterations", value=50, step=10)

with conf_col3:
    with st.expander("📐 Thickness Limits", expanded=False):
        tmin = st.number_input("Min Thickness (in)", value=2.0, step=0.5)
        tmax = st.number_input("Max Thickness (in)", value=12.0, step=0.5)

st.markdown("---")


# ==========================================
# PART 3: BOUNDARY CONDITIONS & RUN OPTIMIZATION (2 COLUMNS)
# ==========================================
st.markdown('<div class="section-header">🎛️ Boundary Conditions & Solver</div>', unsafe_allow_html=True)

col_bc, col_run = st.columns(2)

#
