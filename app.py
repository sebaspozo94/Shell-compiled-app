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

# Split the layout into two columns
col_bc, col_run = st.columns(2)

# --- 3A. BOUNDARY CONDITIONS COLUMN ---
with col_bc:
    # Initialize session state keys for toggles
    if 'add_t' not in st.session_state: st.session_state.add_t = False
    if 'del_t' not in st.session_state: st.session_state.del_t = False

    def on_add_toggle():
        if st.session_state.add_t:
            st.session_state.del_t = False

    def on_del_toggle():
        if st.session_state.del_t:
            st.session_state.add_t = False

    col_t1, col_t2 = st.columns(2)
    add_mode = col_t1.toggle("🖱️ Click to ADD Support", key="add_t", on_change=on_add_toggle)
    del_mode = col_t2.toggle("🗑️ Click to DELETE Support", key="del_t", on_change=on_del_toggle)

    fig2d = go.Figure()

    fig2d.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                    line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        color = '#2563eb'
        fig2d.add_shape(type="rect", x0=row['X (in)']-hx, y0=row['Y (in)']-hy, x1=row['X (in)']+hx, y1=row['Y (in)']+hy, 
                        line=dict(color=color, width=2), fillcolor=color, opacity=0.6)
        fig2d.add_annotation(x=row['X (in)'], y=row['Y (in)'], text=f"S{i+1}", showarrow=False, 
                             font=dict(color="black", size=11, family="Arial Black"))

    grid_spacing = 12
    grid_x, grid_y = np.meshgrid(np.arange(0, dimx + 1, grid_spacing), np.arange(0, dimy + 1, grid_spacing))
    gx, gy = grid_x.flatten(), grid_y.flatten()

    grid_opacity = 0.3 if (add_mode or del_mode) else 0.0
    grid_color = 'blue' if add_mode else 'red'

    fig2d.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=12, color=grid_color, opacity=grid_opacity, symbol='square'),
        hoverinfo='text', text="Click here", name="Grid"
    ))

    # Incorporating the domain constraint fix so resizing works nicely in columns
    fig2d.update_layout(
        autosize=True,
        xaxis=dict(range=[-10, dimx+10], constrain='domain'), 
        yaxis=dict(range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain='domain'), 
        clickmode='event+select', 
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False
    )

    event = st.plotly_chart(fig2d, on_select="rerun", key="bc_map", use_container_width=True)

    if event and "selection" in event and len(event["selection"]["points"]) > 0:
        pt = event["selection"]["points"][0]
        cx, cy = pt['x'], pt['y']
        
        if add_mode:
            if not ((st.session_state.bc_df['X (in)'] == cx) & (st.session_state.bc_df['Y (in)'] == cy)).any():
                new_row = pd.DataFrame([[float(cx), float(cy), 4.0, 4.0, "Pinned"]], 
                                       columns=["X (in)", "Y (in)", "Width", "Height", "Type"])
                st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
                st.rerun()
                
        elif del_mode:
            to_drop = []
            for i, row in st.session_state.bc_df.iterrows():
                hx, hy = row['Width']/2, row['Height']/2
                if (row['X (in)']-hx <= cx <= row['X (in)']+hx) and (row['Y (in)']-hy <= cy <= row['Y (in)']+hy):
                    to_drop.append(i)
            
            if to_drop:
                st.session_state.bc_df = st.session_state.bc_df.drop(to_drop).reset_index(drop=True)
                st.rerun()
                
    with st.expander("📋 View/Edit Support Coordinates", expanded=False):
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
        if not edited_bc_df.drop(columns=["ID"]).equals(st.session_state.bc_df):
            st.session_state.bc_df = edited_bc_df.drop(columns=["ID"])
            st.rerun()

# --- 3B. SOLVER / RUN COLUMN ---
with col_run:
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little vertical padding
    
    # Solver prep
    solver_df = st.session_state.bc_df.copy()
    solver_df["Type"] = solver_df["Type"].map({"Pinned": 0, "Fixed": 1})
    BCMatrix = solver_df.to_numpy()

    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if len(BCMatrix) == 0:
            st.error("Please add at least one support!")
        else:
            total_area = dimx * dimy
            target_volume = (total_area * tmin) + (vol_frac * total_area * (tmax - tmin))
            status_text = st.empty()
            live_plot_spot = st.empty()
            
            def update_live_view(current_it, current_ch, current_Z):
                fig_live, ax_live = plt.subplots(figsize=(6, 4)) # Reduced figsize slightly to fit the column better
                fig_live.patch.set_alpha(0.0)
                ax_live.axis('off') 
                
                im = ax_live.imshow(current_Z, cmap=custom_cmap, vmin=0, vmax=tmax, 
                                    extent=[0, dimx, 0, dimy], origin='upper')
                
                for i, row in st.session_state.bc_df.iterrows():
                    hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
                    rect = plt.Rectangle((row['X (in)'] - hx, row['Y (in)'] - hy), 
                                         row['Width'], row['Height'], 
                                         color='black', alpha=0.3)
                    ax_live.add_patch(rect)
                    
                    ax_live.text(row['X (in)'], row['Y (in)'], f"S{i+1}", 
                                 color='black', ha='center', va='center', 
                                 fontsize=9, fontweight='bold')
                
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

# ==========================================
# PART 4: INTERACTIVE 3D RESULTS
# ==========================================
if st.session_state.run_finished:
    st.markdown("---")
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
    with col_slider: 
        idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
    with col_scale: 
        z_scale_pct = st.slider("Visual Z-Scale (%)", 0, 100, key="z_scale_val")
    
    Z_raw = st.session_state.history[idx]
    
    # FIX MIRRORING: Flip up-down and transpose
    Z_final = np.flipud(Z_raw).T 
    
    x_coords = np.linspace(0, dimx, Z_final.shape[1])
    y_coords = np.linspace(0, dimy, Z_final.shape[0])
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)

    Z_plot_neg = -Z_final 
    custom_colorscale = [[0.0, '#08306b'], [0.4, '#2563eb'], [1.0, '#cbd5e1']]

    roof_surface = go.Surface(z=np.zeros_like(Z_plot_neg), x=X_mesh, y=Y_mesh, colorscale=[[0, '#cbd5e1'], [1, '#cbd5e1']], showscale=False, hoverinfo='skip')
    bottom_surface = go.Surface(z=Z_plot_neg, x=X_mesh, y=Y_mesh, colorscale=custom_colorscale, cmin=-tmax, cmax=0, colorbar=dict(title='Thickness (in)'))

    fig = go.Figure(data=[roof_surface, bottom_surface])

    support_depth = -tmax * 1.2
    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        x_min, x_max = row['X (in)'] - hx, row['X (in)'] + hx
        y_min, y_max = row['Y (in)'] - hy, row['Y (in)'] + hy
        
        fig.add_trace(go.Mesh3d(
            x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            z=[support_depth, support_depth, support_depth, support_depth, tmax * 0.1, tmax * 0.1, tmax * 0.1, tmax * 0.1],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='red', 
            opacity=0.8, 
            flatshading=True,
            name=f"Support S{i+1}",
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=st.session_state.bc_df['X (in)'],
        y=st.session_state.bc_df['Y (in)'],
        z=[tmax * 0.1] * len(st.session_state.bc_df),
        mode='text',
        text=[f"S{i+1}" for i in range(len(st.session_state.bc_df))],
        textfont=dict(color="black", size=14, family="Arial Black"),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        uirevision=st.session_state.view_rev, 
        scene=dict(
            xaxis=dict(range=[-0.05 * dimx, 1.05 * dimx], title='X (in)'),
            yaxis=dict(range=[-0.05 * dimy, 1.05 * dimy], title='Y (in)'),
            zaxis=dict(range=[support_depth, tmax * 0.2], title='Z (in)'),
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
