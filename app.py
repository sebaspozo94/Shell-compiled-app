import streamlit as st
import numpy as np
import logic  # Ensure logic.so is in the same directory
import plotly.graph_objects as go
import io
import stl              
from stl import mesh 
from scipy.spatial import Delaunay
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Shell Dashboard", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem; }
    .tag-container { display: flex; gap: 10px; margin-bottom: 1rem; }
    .tag { background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; border: 1px solid #e2e8f0; }
    .section-header { font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">Shell Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="tag-container"><span class="tag">Dashboard Layout</span><span class="tag">Shell</span><span class="tag">FEA Engine</span></div>', unsafe_allow_html=True)
st.markdown("---")

# --- SETUP SESSION STATE ---
if 'run_finished' not in st.session_state: st.session_state.run_finished = False
if 'history' not in st.session_state: st.session_state.history = None
if 'X' not in st.session_state: st.session_state.X = None
if 'Y' not in st.session_state: st.session_state.Y = None
if "bc_df" not in st.session_state:
    st.session_state.bc_df = pd.DataFrame(
        [[48.0, 156.0, 4.0, 4.0, "Pinned"], [48.0, 36.0, 4.0, 4.0, "Pinned"], [192.0, 156.0, 4.0, 4.0, "Pinned"], [192.0, 36.0, 4.0, 4.0, "Pinned"]],
        columns=["X (in)", "Y (in)", "Width", "Height", "Type"]
    )
if "run_bc_df" not in st.session_state: st.session_state.run_bc_df = st.session_state.bc_df.copy()
if "show_labels" not in st.session_state: st.session_state.show_labels = False

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
# DASHBOARD GRID SETUP
# ==========================================
# ROW 1
r1c1, r1c2, r1c3 = st.columns(3)
# ROW 2
r2c1, r2c2, r2c3 = st.columns(3)
# ROW 3
r3c1, r3c2, r3c3 = st.columns(3)

# ==========================================
# ROW 1, COLUMN 1: App Objective & Model Config
# ==========================================
with r1c1:
    st.markdown('<div class="section-header">🎯 Objective & Config</div>', unsafe_allow_html=True)
    
    with st.expander("App Objective", expanded=False):
        st.markdown("**Objective:** Distribute material to maximize stiffness under load & self-weight.")
        
    with st.expander("📏 Domain & Mesh", expanded=False):
        dimx = st.number_input("Domain X (in)", value=240, step=4, min_value=1)
        dimy = st.number_input("Domain Y (in)", value=192, step=4, min_value=1)
        nelx = st.number_input("Elements X", value=120, step=4, min_value=1)
        nely = st.number_input("Elements Y", value=96, step=4, min_value=1)

    with st.expander("🎯 Optimization Settings", expanded=False):
        vol_frac = st.slider("Volume Fraction", 0.05, 1.0, 0.3)
        rmin = st.number_input("Filter Radius (rmin)", value=5.0, step=1.0)
        itmax = st.number_input("Max Iterations", value=50, step=10)

    with st.expander("📐 Thickness Limits", expanded=False):
        tmin = st.number_input("Min Thickness (in)", value=2.0, step=0.5)
        tmax = st.number_input("Max Thickness (in)", value=12.0, step=0.5)

# ==========================================
# ROW 2, COLUMN 1: Boundary Conditions
# ==========================================
with r2c1:
    st.markdown('<div class="section-header">🎛️ Boundary Conditions</div>', unsafe_allow_html=True)
    
    if 'add_t' not in st.session_state: st.session_state.add_t = False
    if 'del_t' not in st.session_state: st.session_state.del_t = False
    def on_add_toggle():
        if st.session_state.add_t: st.session_state.del_t = False
    def on_del_toggle():
        if st.session_state.del_t: st.session_state.add_t = False

    c_t1, c_t2 = st.columns(2)
    add_mode = c_t1.toggle("➕ ADD", key="add_t", on_change=on_add_toggle)
    del_mode = c_t2.toggle("➖ DELETE", key="del_t", on_change=on_del_toggle)

    fig2d = go.Figure()
    fig2d.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        fig2d.add_shape(type="rect", x0=row['X (in)']-hx, y0=row['Y (in)']-hy, x1=row['X (in)']+hx, y1=row['Y (in)']+hy, 
                        line=dict(color='red', width=2), fillcolor='red', opacity=0.6)
        if st.session_state.show_labels:
            fig2d.add_annotation(x=row['X (in)'], y=row['Y (in)'], text=f"S{i+1}", showarrow=False, font=dict(color="black", size=11, family="Arial Black"))

    grid_spacing = 12
    grid_x, grid_y = np.meshgrid(np.arange(0, dimx + 1, grid_spacing), np.arange(0, dimy + 1, grid_spacing))
    gx, gy = grid_x.flatten(), grid_y.flatten()
    grid_opacity = 0.3 if (add_mode or del_mode) else 0.0
    grid_color = 'blue' if add_mode else 'red'

    fig2d.add_trace(go.Scatter(x=gx, y=gy, mode='markers', marker=dict(size=10, color=grid_color, opacity=grid_opacity, symbol='square'), hoverinfo='text', text="Click here", name="Grid"))

    fig2d.update_layout(autosize=True, xaxis=dict(range=[-10, dimx+10], constrain='domain'), yaxis=dict(range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain='domain'), clickmode='event+select', margin=dict(l=0, r=0, t=0, b=0), showlegend=False)

    event = st.plotly_chart(fig2d, on_select="rerun", key="bc_map", use_container_width=True)

    if event and "selection" in event and len(event["selection"]["points"]) > 0:
        pt = event["selection"]["points"][0]
        cx, cy = pt['x'], pt['y']
        if add_mode:
            if not ((st.session_state.bc_df['X (in)'] == cx) & (st.session_state.bc_df['Y (in)'] == cy)).any():
                new_row = pd.DataFrame([[float(cx), float(cy), 4.0, 4.0, "Pinned"]], columns=["X (in)", "Y (in)", "Width", "Height", "Type"])
                st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
                st.rerun()
        elif del_mode:
            to_drop = []
            for i, row in st.session_state.bc_df.iterrows():
                hx, hy = row['Width']/2, row['Height']/2
                if (row['X (in)']-hx <= cx <= row['X (in)']+hx) and (row['Y (in)']-hy <= cy <= row['Y (in)']+hy): to_drop.append(i)
            if to_drop:
                st.session_state.bc_df = st.session_state.bc_df.drop(to_drop).reset_index(drop=True)
                st.rerun()
                
    with st.expander("📋 Edit Coordinates", expanded=False):
        st.checkbox("🏷️ Show Identifiers", key="show_labels")
        display_df = st.session_state.bc_df.copy()
        display_df.insert(0, "ID", [f"S{i+1}" for i in range(len(display_df))])
        edited_bc_df = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"ID": st.column_config.TextColumn(disabled=True), "Type": st.column_config.SelectboxColumn("Type", options=["Pinned", "Fixed"])})
        if not edited_bc_df.drop(columns=["ID"]).equals(st.session_state.bc_df):
            st.session_state.bc_df = edited_bc_df.drop(columns=["ID"])
            st.rerun()

# ==========================================
# ROW 2, COLUMN 2: Solver Output
# ==========================================
with r2c2:
    st.markdown('<div class="section-header">🚀 Solver</div>', unsafe_allow_html=True)
    
    solver_df = st.session_state.bc_df.copy()
    solver_df["Type"] = solver_df["Type"].map({"Pinned": 0, "Fixed": 1})
    BCMatrix = solver_df.to_numpy()

    run_pressed = st.button("Run Optimization", type="primary", use_container_width=True)
    live_plot_spot = st.empty()
    status_text = st.empty()

    def plot_2d_thickness_mpl(Z_matrix):
        x_range, y_range = dimx + 20, dimy + 20
        fig = plt.figure(figsize=(5, 5 * (y_range / x_range)), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue", ['#cbd5e1', '#2563eb', '#08306b'])
        ax.imshow(np.flipud(Z_matrix), cmap=custom_cmap, extent=[0, dimx, 0, dimy], vmin=0, vmax=tmax, interpolation='nearest')
        ax.add_patch(patches.Rectangle((0, 0), dimx, dimy, linewidth=2, edgecolor='#0f172a', facecolor='none', linestyle='--'))
        for _, row in st.session_state.run_bc_df.iterrows():
            ax.add_patch(patches.Rectangle((row['X (in)'] - row['Width']/2, row['Y (in)'] - row['Height']/2), row['Width'], row['Height'], linewidth=1, edgecolor='darkred', facecolor='red', alpha=0.5))
        ax.set_xlim(-10, dimx + 10)
        ax.set_ylim(-10, dimy + 10)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf

    def plot_2d_thickness_plotly(Z_matrix):
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=np.flipud(Z_matrix), x=np.linspace(0, dimx, Z_matrix.shape[1]), y=np.linspace(0, dimy, Z_matrix.shape[0]), colorscale=[[0.0, '#cbd5e1'], [0.5, '#2563eb'], [1.0, '#08306b']], zmin=0, zmax=tmax, showscale=False, hoverinfo='skip'))
        fig.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
        for _, row in st.session_state.run_bc_df.iterrows():
            hx, hy = row['Width']/2.0, row['Height']/2.0
            fig.add_shape(type="rect", x0=row['X (in)']-hx, y0=row['Y (in)']-hy, x1=row['X (in)']+hx, y1=row['Y (in)']+hy, line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.4)')
        fig.update_layout(autosize=True, xaxis=dict(range=[-10, dimx+10], constrain='domain'), yaxis=dict(range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain='domain'), margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        return fig

    if run_pressed:
        if len(BCMatrix) == 0:
            st.error("Please add supports!")
        else:
            st.session_state.run_bc_df = st.session_state.bc_df.copy()
            target_volume = (dimx * dimy * tmin) + (vol_frac * dimx * dimy * (tmax - tmin))
            def update_live_view(current_it, current_ch, current_Z):
                live_plot_spot.image(plot_2d_thickness_mpl(current_Z), use_container_width=True) 
                status_text.info(f"⚙️ Iteration: {current_it}")

            with st.spinner("Optimizing..."):
                SW_val = 1 if self_weight else 0
                X, Y, Thickness, history = logic.run_topology_optimization(float(dimx), float(dimy), float(E), float(nu), float(rho), int(SW_val), BCMatrix, float(w_u), int(nelx), int(nely), float(target_volume), float(rmin), float(tmin), float(tmax), int(itmax), progress_callback=update_live_view)
                st.session_state.history, st.session_state.X, st.session_state.Y, st.session_state.run_finished = history, X, Y, True
                st.rerun()

    if st.session_state.run_finished and st.session_state.history is not None:
        live_plot_spot.plotly_chart(plot_2d_thickness_plotly(st.session_state.history[-1]), use_container_width=True, key="final_result_plot")
        status_text.success(f"✅ Complete! Iterations: {len(st.session_state.history)}")

# ==========================================
# ROW 2, COLUMN 3: 3D Interactive Results
# ==========================================
with r2c3:
    st.markdown('<div class="section-header">🕒 3D View</div>', unsafe_allow_html=True)
    if st.session_state.run_finished:
        steps = len(st.session_state.history)
        
        # Squeezed controls for the 1/3 column
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            view_choice = st.selectbox("🎥 Camera View", ["Default", "Bottom (XY)", "Front (XZ)", "Side (YZ)"])
        with v_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_true_scale = st.checkbox("📏 True Z-Scale")

        # Set Camera Angles
        if view_choice == "Bottom (XY)": cam_eye, cam_up = dict(x=0, y=0, z=-2.5), dict(x=0, y=1, z=0)
        elif view_choice == "Front (XZ)": cam_eye, cam_up = dict(x=0, y=-2.5, z=0), dict(x=0, y=0, z=1)
        elif view_choice == "Side (YZ)": cam_eye, cam_up = dict(x=-2.5, y=0, z=0), dict(x=0, y=0, z=1)
        else: cam_eye, cam_up = dict(x=1.2, y=-1.5, z=-0.8), dict(x=0, y=0, z=1)

        idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
        
        if "z_scale_val" not in st.session_state: st.session_state.z_scale_val = 100
        if use_true_scale:
            z_scale_pct = int(100*tmax/max(dimx, dimy))
        else:
            z_scale_pct = st.slider("Visual Z-Scale (%)", 0, 100, st.session_state.z_scale_val)
            st.session_state.z_scale_val = z_scale_pct
        
        Z_final = np.flipud(st.session_state.history[idx]).T 
        X_mesh, Y_mesh = np.meshgrid(np.linspace(0, dimx, Z_final.shape[1]), np.linspace(0, dimy, Z_final.shape[0]))
        Z_plot_neg = -Z_final 

        roof_surface = go.Surface(z=np.zeros_like(Z_plot_neg), x=X_mesh, y=Y_mesh, colorscale=[[0, '#cbd5e1'], [1, '#cbd5e1']], showscale=False, hoverinfo='skip')
        bottom_surface = go.Surface(z=Z_plot_neg, x=X_mesh, y=Y_mesh, colorscale=[[0.0, '#08306b'], [0.4, '#2563eb'], [1.0, '#cbd5e1']], cmin=-tmax, cmax=0, 
                                    colorbar=dict(title='Thick. (in)', orientation='h', x=0.5, y=1.05, xanchor='center', yanchor='bottom', thickness=10, len=0.7))

        fig3d = go.Figure(data=[roof_surface, bottom_surface])
        support_depth = -tmax * 1.2
        
        for _, row in st.session_state.run_bc_df.iterrows():
            hx, hy = row['Width']/2.0, row['Height']/2.0
            x_min, x_max = row['X (in)']-hx, row['X (in)']+hx
            y_min, y_max = row['Y (in)']-hy, row['Y (in)']+hy
            fig3d.add_trace(go.Mesh3d(x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min], y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max], z=[support_depth, support_depth, support_depth, support_depth, tmax*0.1, tmax*0.1, tmax*0.1, tmax*0.1], i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color='red', opacity=0.8, flatshading=True, showlegend=False))

        fig3d.update_layout(
            scene=dict(xaxis=dict(range=[-0.05*dimx, 1.05*dimx], title='X'), yaxis=dict(range=[-0.05*dimy, 1.05*dimy], title='Y'), zaxis=dict(range=[support_depth, tmax*0.2], title='Z'), aspectratio=dict(x=dimx/max(dimx, dimy), y=dimy/max(dimx, dimy), z=z_scale_pct/100.0), camera=dict(eye=cam_eye, up=cam_up)),
            margin=dict(l=0, r=0, b=0, t=50), height=450
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("Run the solver to generate 3D results.")

# ==========================================
# ROW 3, COLUMN 1: STL Export
# ==========================================
with r3c1:
    if st.session_state.run_finished:
        st.markdown('<div class="section-header">💾 Export</div>', unsafe_allow_html=True)
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

        # Generates STL based on the currently selected iteration slider in the 3D block
        stl_data = generate_stl(X_mesh, Y_mesh, Z_plot_neg)
        st.download_button(label="📥 Download as .STL File", data=stl_data, file_name=f"Optimized_Slab_Iter{idx}.stl", mime="model/stl", type="primary", use_container_width=True)
