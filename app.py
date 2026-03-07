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

# --- APP HEADER ---
st.markdown('<div class="main-header">Shell Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="tag-container"><span class="tag">Optimization</span><span class="tag">Shell</span><span class="tag">FEA Engine</span></div>', unsafe_allow_html=True)

with st.expander("🎯 App Objective", expanded=False):
    st.markdown("""
    **Objective:** Distribute a constant amount of material to maximize the stiffness of a shell-type structure 
    under external distributed load and self-weight.
    """)

# --- 1. SETUP SESSION STATE ---
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

# --- 2. SIDEBAR (Materials & Loads Only) ---
with st.sidebar:
    st.header("🧪 Material Properties")
    E = st.number_input("Elastic Modulus (psi)", value=1500000, step=100000)
    nu = st.slider("Poisson's Ratio (v)", 0.0, 0.5, 0.30)
    rho = st.number_input("Material Density (p)", value=0.010, format="%.3f")
    self_weight = st.checkbox("Include Self-Weight", value=True)

    st.header("⚖️ Loads")
    w_u = st.number_input("Distributed Load (w_u)", value=0.2778)

# ==========================================
# SECTION 1: MODEL CONFIGURATION (DROP MENUS)
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
        rmin = st.number_input("Filter Radius (rmin)", value=5.0)
        itmax = st.number_input("Max Iterations", value=50)

with conf_col3:
    with st.expander("📐 Thickness Limits", expanded=False):
        tmin = st.number_input("Min Thickness (in)", value=2.0)
        tmax = st.number_input("Max Thickness (in)", value=12.0)

st.markdown("---")

# ==========================================
# SECTION 2: INTERACTIVE BOUNDARY CONDITIONS
# ==========================================
st.markdown('<div class="section-header">🔍 Interactive Support Setup</div>', unsafe_allow_html=True)

col_t1, col_t2 = st.columns(2)
add_mode = col_t1.toggle("🖱️ Click to ADD Support", value=False, key="add_t")
del_mode = col_t2.toggle("🗑️ Click to DELETE Support", value=False, key="del_t")

# Ensure mutual exclusivity: If one is turned on, the other shouldn't interfere
if add_mode and del_mode:
    st.warning("Both modes active. System will prioritize the specific point selected.")

fig2d = go.Figure()

# 1. Draw the Design Domain
fig2d.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

# 2. Draw existing supports with Labels
for i, row in st.session_state.bc_df.iterrows():
    hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
    color = '#2563eb' if row['Type'] == "Pinned" else '#0f172a' 
    
    # Draw the support rectangle
    fig2d.add_shape(type="rect", 
                    x0=row['X (in)'] - hx, y0=row['Y (in)'] - hy, 
                    x1=row['X (in)'] + hx, y1=row['Y (in)'] + hy, 
                    line=dict(color=color, width=2), fillcolor=color, opacity=0.7)
    
    # Add the Identifier Label (S1, S2, etc.)
    fig2d.add_annotation(x=row['X (in)'], y=row['Y (in)'], text=f"S{i+1}", 
                         showarrow=False, font=dict(color="black", size=10, family="Arial Black"))

# 3. Show Grid for Interaction (Both Add and Delete now use the grid)
if add_mode or del_mode:
    grid_spacing = 12 # 12-inch grid for selection
    grid_x, grid_y = np.meshgrid(np.arange(0, dimx + grid_spacing, grid_spacing), 
                                 np.arange(0, dimy + grid_spacing, grid_spacing))
    
    grid_color = 'rgba(37, 99, 235, 0.2)' if add_mode else 'rgba(239, 68, 68, 0.2)'
    
    fig2d.add_trace(go.Scatter(
        x=grid_x.flatten(), y=grid_y.flatten(), 
        mode='markers', 
        marker=dict(size=8, color=grid_color, symbol='square'), 
        hoverinfo='text',
        text="Click to Add/Delete"
    ))

fig2d.update_layout(
    xaxis=dict(title="X (in)", range=[-10, dimx+10], constrain="domain", gridcolor='#f1f5f9'),
    yaxis=dict(title="Y (in)", range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain="domain", gridcolor='#f1f5f9'),
    margin=dict(l=0, r=0, t=20, b=0), height=500, showlegend=False, clickmode='event+select', plot_bgcolor='white'
)

event = st.plotly_chart(fig2d, on_select="rerun", selection_mode="points", key="bc_map", use_container_width=True)

# 4. Handle Logic for Add/Delete via Grid Selection
if event and event.get("selection") and len(event["selection"]["points"]) > 0:
    clicked_pt = event["selection"]["points"][0]
    cx, cy = clicked_pt["x"], clicked_pt["y"]
    
    if add_mode and not del_mode:
        # Check for duplicates at the exact grid point
        duplicate = st.session_state.bc_df[(st.session_state.bc_df['X (in)'] == cx) & (st.session_state.bc_df['Y (in)'] == cy)]
        if duplicate.empty:
            new_row = pd.DataFrame([[float(cx), float(cy), 4.0, 4.0, "Pinned"]], 
                                   columns=["X (in)", "Y (in)", "Width", "Height", "Type"])
            st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
            st.rerun()
            
    elif del_mode:
        # HIT-TEST: Check if clicked grid point is inside ANY existing support area
        to_drop = []
        for i, row in st.session_state.bc_df.iterrows():
            hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
            # If click point is within the rectangle bounds
            if (row['X (in)'] - hx <= cx <= row['X (in)'] + hx) and \
               (row['Y (in)'] - hy <= cy <= row['Y (in)'] + hy):
                to_drop.append(i)
        
        if to_drop:
            st.session_state.bc_df = st.session_state.bc_df.drop(to_drop).reset_index(drop=True)
            st.rerun()

with st.expander("📋 View/Edit Support Coordinates", expanded=False):
    # Nested Drop Menu for Instructions
    with st.expander("📖 How to read this table?", expanded=False):
        st.markdown("""
        | Column | Description |
        | :--- | :--- |
        | **X / Y (in)** | The center coordinates of the support on the domain. |
        | **Width / Height** | The physical size of the support "patch" in inches. |
        | **Type** | **Pinned:** Restricts movement (XYZ). <br> **Fixed:** Restricts movement and rotation (Moment). |
        
        **Pro Tips:**
        * You can manually type coordinates here to be laser-accurate.
        * Use the `Delete` key on your keyboard to remove a row.
        * Click the `+` at the bottom of the table to add a support manually.
        """)
    display_df = st.session_state.bc_df.copy()
    display_df.insert(0, "ID", [f"S{i+1}" for i in range(len(display_df))])
    
    edited_bc_df = st.data_editor(
        display_df, 
        num_rows="dynamic", use_container_width=True, hide_index=True, 
        column_config={"ID": st.column_config.TextColumn(disabled=True), "Type": st.column_config.SelectboxColumn("Type", options=["Pinned", "Fixed"])}
    )
    # Sync back (excluding the ID column)
    if not edited_bc_df.drop(columns=["ID"]).equals(st.session_state.bc_df):
        st.session_state.bc_df = edited_bc_df.drop(columns=["ID"])
        st.rerun()
        
# Solver prep
solver_df = st.session_state.bc_df.copy()
solver_df["Type"] = solver_df["Type"].map({"Pinned": 0, "Fixed": 1})
BCMatrix = solver_df.to_numpy()

# ==========================================
# SECTION 3: SOLVER & RESULTS
# ==========================================
st.markdown("---")
col_btn_l, col_btn_mid, col_btn_r = st.columns([1, 2, 1])
with col_btn_mid:
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if len(BCMatrix) == 0:
            st.error("Please add at least one support!")
        else:
            total_area = dimx * dimy
            target_volume = (total_area * tmin) + (vol_frac * total_area * (tmax - tmin))
            status_text = st.empty()
            live_plot_spot = st.empty()
            
            def update_live_view(current_it, current_ch, current_Z):
                fig_live, ax_live = plt.subplots(figsize=(10, 4))
                fig_live.patch.set_alpha(0.0)
                ax_live.axis('off') 
                im = ax_live.imshow(current_Z, cmap=custom_cmap, vmin=0, vmax=tmax, extent=[0, dimx, 0, dimy], origin='upper')
                live_plot_spot.pyplot(fig_live)
                plt.close(fig_live)
                status_text.info(f"⚙️ Optimizing... Iteration: {current_it}")

            with st.spinner("Optimizing..."):
                SW_val = 1 if self_weight else 0
                X, Y, Thickness, history = logic.run_topology_optimization(
                    float(dimx), float(dimy), float(E), float(nu), float(rho), int(SW_val), 
                    BCMatrix, float(w_u), int(nelx), int(nely), float(target_volume), 
                    float(rmin), float(tmin), float(tmax), int(itmax), progress_callback=update_live_view
                )
                st.session_state.history, st.session_state.X, st.session_state.Y, st.session_state.run_finished = history, X, Y, True
                st.rerun()

if st.session_state.run_finished:
    st.markdown('<div class="section-header">🕒 Interactive 3D Results</div>', unsafe_allow_html=True)
    steps = len(st.session_state.history)
    
    if "cam_eye" not in st.session_state: st.session_state.cam_eye = dict(x=1.2, y=-1.5, z=-0.8) 
    if "cam_up" not in st.session_state: st.session_state.cam_up = dict(x=0, y=0, z=1) 
    if "z_scale_val" not in st.session_state: st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
    if "view_rev" not in st.session_state: st.session_state.view_rev = 0

    view_cols = st.columns(5)
    if view_cols[0].button("⬇️ Bottom (XY)"):
        st.session_state.cam_eye, st.session_state.cam_up = dict(x=0, y=0, z=-2.5), dict(x=0, y=1, z=0)
        st.session_state.view_rev += 1
    if view_cols[1].button("➡️ Front (XZ)"):
        st.session_state.cam_eye, st.session_state.cam_up = dict(x=0, y=-2.5, z=0), dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
    if view_cols[2].button("↗️ Side (YZ)"):
        st.session_state.cam_eye, st.session_state.cam_up = dict(x=-2.5, y=0, z=0), dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
    if view_cols[3].button("🔄 Reset View"):
        st.session_state.cam_eye, st.session_state.cam_up = dict(x=1.2, y=-1.5, z=-0.8), dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
    if view_cols[4].button("📏 True Scale (Z)"):
        st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
        st.session_state.view_rev += 1

    col_slider, col_scale = st.columns([2, 1])
    with col_slider: idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
    with col_scale: z_scale_pct = st.slider("Visual Z-Scale (%)", 0, 100, key="z_scale_val")
    
    Z_plot = st.session_state.history[idx]
    x_1d, y_1d = np.unique(st.session_state.X), np.unique(st.session_state.Y)
    if len(x_1d) == Z_plot.shape[1] + 1: x_1d = (x_1d[:-1] + x_1d[1:]) / 2.0
    if len(y_1d) == Z_plot.shape[0] + 1: y_1d = (y_1d[:-1] + y_1d[1:]) / 2.0
    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d)

    Z_plot_neg = -Z_plot 
    custom_colorscale = [[0.0, '#08306b'], [0.4, '#2563eb'], [1.0, '#cbd5e1']]

    roof_surface = go.Surface(z=np.zeros_like(Z_plot_neg), x=X_mesh, y=Y_mesh, colorscale=[[0, '#cbd5e1'], [1, '#cbd5e1']], showscale=False, hoverinfo='skip')
    bottom_surface = go.Surface(z=Z_plot_neg, x=X_mesh, y=Y_mesh, colorscale=custom_colorscale, cmin=-tmax, cmax=0, colorbar=dict(title='Thickness (in)'))

    fig = go.Figure(data=[roof_surface, bottom_surface])

    # 4. Add Supports to 3D Plot
    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        # Create a "Block" at the support location
        fig.add_trace(go.Mesh3d(
            x=[row['X (in)']-hx, row['X (in)']+hx, row['X (in)']+hx, row['X (in)']-hx, row['X (in)']-hx, row['X (in)']+hx, row['X (in)']+hx, row['X (in)']-hx],
            y=[row['Y (in)']-hy, row['Y (in)']-hy, row['Y (in)']+hy, row['Y (in)']+hy, row['Y (in)']-hy, row['Y (in)']-hy, row['Y (in)']+hy, row['Y (in)']+hy],
            z=[0, 0, 0, 0, -tmax*1.2, -tmax*1.2, -tmax*1.2, -tmax*1.2],
            color='red', opacity=0.5, name=f"Support S{i+1}"
        ))
    fig.update_layout(
        uirevision=st.session_state.view_rev, 
        scene=dict(
            xaxis=dict(range=[0, dimx], title='X (in)'),
            yaxis=dict(range=[0, dimy], title='Y (in)'),
            zaxis=dict(range=[-tmax, 0], title='Z (in)'),
            aspectratio=dict(x=dimx/max(dimx, dimy), y=dimy/max(dimx, dimy), z=z_scale_pct/100.0),
            camera=dict(eye=st.session_state.cam_eye, up=st.session_state.cam_up)
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # STL Export
    st.subheader("💾 Export Geometry")
    def generate_stl(X, Y, Z):
        points2D = np.column_stack([X.flatten(), Y.flatten()])
        tri = Delaunay(points2D)
        slab_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(tri.simplices):
            for j in range(3):
                slab_mesh.vectors[i][j] = [points2D[f[j], 0], points2D[f[j], 1], Z.flatten()[f[j]]]
        buf = io.BytesIO()
        slab_mesh.save('slab.stl', fh=buf)
        return buf.getvalue()

    stl_data = generate_stl(X_mesh, Y_mesh, Z_plot_neg)
    st.download_button(label="📥 Download as .STL File", data=stl_data, file_name=f"Optimized_Slab_Iter{idx}.stl", mime="model/stl", type="primary")





