import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logic  # This imports your compiled logic.so file
import plotly.graph_objects as go
import io
import stl              # <-- Add this line
from stl import mesh 
from scipy.spatial import Delaunay
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Define the custom Matplotlib colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ['#cbd5e1', '#2563eb', '#08306b'])

st.set_page_config(page_title="Shell Topology Opt", layout="wide")

# --- CUSTOM CSS FOR PORTFOLIO MATCHING ---
st.markdown("""
<style>
    /* Style the main title to look like the portfolio headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0rem;
    }
    /* Style the pill tags */
    .tag-container {
        display: flex;
        gap: 10px;
        margin-bottom: 2rem;
    }
    .tag {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 12px;
        border-radius: 9999px; /* Fully rounded */
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<div class="main-header">Shell Topology Optimization</div>', unsafe_allow_html=True)
# Add portfolio-style pill tags
st.markdown("""
<div class="tag-container">
    <span class="tag">Optimization</span>
    <span class="tag">Shell</span>
    <span class="tag">FEA Engine</span>
</div>
""", unsafe_allow_html=True)

# --- NEW: App Objective Expander ---
with st.expander("🎯 App Objective"):
    st.markdown("""
    **Objective:** Distribute a constant amount of material to maximize the stiffness of a shell-type structure under external distributed load and self-weight.
    """)

# --- 1. SETUP SESSION STATE ---
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
    st.session_state.history = None
    st.session_state.X = None
    st.session_state.Y = None
if 'mesh_data' not in st.session_state:
    st.session_state.mesh_data = None

with st.sidebar:
    st.header("📐 Geometry & Mesh")
    # Group Dimensions
    col1, col2 = st.sidebar.columns(2)
    dimx = col1.number_input("Domain X (in)", value=240, step=4, min_value=1)
    dimy = col2.number_input("Domain Y (in)", value=192, step=4, min_value=1)
    
    # Group Elements
    col3, col4 = st.sidebar.columns(2)
    elemx = col3.number_input("Elements X", value=120, step=4, min_value=1,max_value=150)
    elemy = col4.number_input("Elements Y", value=96, step=4, min_value=1,max_value=150)

    # Hide Materials in an Expander (Closed by default)
    with st.sidebar.expander("🧪 Material Properties"):
        E = st.number_input("Elastic Modulus (psi)", value=1500000, step=100000)
        nu = st.slider("Poisson's Ratio (v)", 0.0, 0.5, 0.30)
        rho = st.number_input("Material Density (p)", value=0.010, format="%.3f")
        self_weight = st.checkbox("Include Self-Weight", value=True)

    st.header("⚖️ Loads & Constraints")
    w_u = st.number_input("Distributed Load (w_u)", value=0.2778)
        
    st.subheader("Boundary Conditions")
    
    with st.expander("ℹ️ How to define supports"):
        st.markdown("""
        Each row represents a boundary condition zone. **X** and **Y** define the center coordinate of a rectangle with dimensions **Width** (dx) and **Height** (dy). All nodes that fall inside this rectangular area will have the boundary condition specified in the **Type** column.
        """)
    
    # 1. Initialize the DataFrame into Session State ONLY IF it doesn't exist yet
    if "bc_df" not in st.session_state:
        st.session_state.bc_df = pd.DataFrame(
            [
                [48.0, 156.0, 4.0, 4.0, "Pinned"], 
                [48.0, 36.0, 4.0, 4.0, "Pinned"],  
                [192.0, 156.0, 4.0, 4.0, "Pinned"], 
                [192.0, 36.0, 4.0, 4.0, "Pinned"]
            ],
            columns=["X (in)", "Y (in)", "Width", "Height", "Type"]
        )
    
    # 2. Setup the Interactive Map
    add_mode = st.toggle("🖱️ Click on Map to Add Pinned Support", value=False)
    
    fig2d = go.Figure()
    
    # Draw the main slab boundary (using your dimx and dimy variables)
    fig2d.add_shape(
        type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
        line=dict(color="RoyalBlue", width=2), fillcolor="rgba(65, 105, 225, 0.1)"
    )
    
    # Draw existing supports from the session state dataframe
    for _, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        color = "#ef4444" if row['Type'] == "Pinned" else "#22c55e" # Red for Pin, Green for Fix
        
        fig2d.add_shape(
            type="rect", x0=row['X (in)'] - hx, y0=row['Y (in)'] - hy, x1=row['X (in)'] + hx, y1=row['Y (in)'] + hy,
            line=dict(color=color, width=2), fillcolor=color
        )
    
    # If Add Mode is ON, create a clickable grid (spacing every 12 inches)
    if add_mode:
        grid_x, grid_y = np.meshgrid(np.arange(0, dimx + 12, 12), np.arange(0, dimy + 12, 12))
        fig2d.add_trace(go.Scatter(
            x=grid_x.flatten(), y=grid_y.flatten(), mode='markers',
            marker=dict(size=6, color='rgba(100, 100, 100, 0.4)'),
            hoverinfo='none'
        ))
    
    # Format the plot
    fig2d.update_layout(
        xaxis=dict(title="X (in)", range=[-20, dimx+20], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (in)", range=[-20, dimy+20]),
        height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, clickmode='event+select'
    )
    
    # Render the plot and catch click events
    event = st.plotly_chart(fig2d, on_select="rerun", selection_mode="points", key="bc_map")
    
    # 3. Process the click to add a new support
    if add_mode and event and len(event.selection["points"]) > 0:
        clicked_pt = event.selection["points"][0]
        new_x, new_y = clicked_pt["x"], clicked_pt["y"]
        
        # Check for duplicates to prevent stacking them on the exact same spot
        duplicate = st.session_state.bc_df[(st.session_state.bc_df['X (in)'] == new_x) & (st.session_state.bc_df['Y (in)'] == new_y)]
        if duplicate.empty:
            new_row = pd.DataFrame([[new_x, new_y, 4.0, 4.0, "Pinned"]], columns=["X (in)", "Y (in)", "Width", "Height", "Type"])
            st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
            st.rerun() # Force a refresh to show the new support instantly
    
    # 4. Your existing Data Editor setup (now reading from session_state)
    edited_bc_df = st.data_editor(
        st.session_state.bc_df, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Support Type",
                help="Select between Pinned or Fixed",
                options=["Pinned", "Fixed"],
                required=True,
            )
        }
    )
    
    # 5. Two-Way Sync: If the user types in the table, update the session state and refresh the map
    if not edited_bc_df.equals(st.session_state.bc_df):
        st.session_state.bc_df = edited_bc_df
        st.rerun()
    
    # 6. Convert for the solver (your original logic)
    solver_df = edited_bc_df.copy()
    solver_df["Type"] = solver_df["Type"].map({"Pinned": 0, "Fixed": 1})
    BCMatrix = solver_df.to_numpy()

    st.header("🎯 Optimization Params")
    vol_frac = st.slider("Volume Fraction", 0.05, 1.0, 0.3)
    rmin = st.number_input("Filter Radius (rmin)", value=5.0)
    tmin = st.number_input("Min Thickness", value=2.0)
    tmax = st.number_input("Max Thickness", value=12.0)
    itmax = st.number_input("Max Iterations", value=50)

# --- 2. THE MAIN COLUMNS ---
col1, col2 = st.columns(2)

with col1:
    # Notice we changed 'primary' to standard so it's a clean white button
    if st.button("🔍 Preview Geometry", use_container_width=True):
        with st.spinner("Generating preview..."):
            fig_pre, ax_pre = plt.subplots(figsize=(10, 6))
            # Make the plot background transparent to match the app
            fig_pre.patch.set_alpha(0.0)
            ax_pre.patch.set_alpha(0.0)
            
            domain = patches.Rectangle((0, 0), dimx, dimy, linewidth=2, edgecolor='#0f172a', facecolor='none', linestyle='--')
            ax_pre.add_patch(domain)
            
            for i in range(BCMatrix.shape[0]):
                x0, y0, bdx, bdy, cond = BCMatrix[i]
                lower_left_x, lower_left_y = x0 - bdx/2, y0 - bdy/2
                # Change colors to match the slate/blue theme
                color = '#0f172a' if cond == 1 else '#2563eb' 
                label = 'Fixed' if cond == 1 else 'Pinned'
                
                rect = patches.Rectangle(
                    (lower_left_x, lower_left_y), bdx, bdy, 
                    linewidth=1, edgecolor=color, facecolor=color, alpha=0.7,
                    label=f"Support {i+1} ({label})"
                )
                ax_pre.add_patch(rect)
            
            ax_pre.set_xlim(-10, dimx + 10); ax_pre.set_ylim(-10, dimy + 10)
            ax_pre.set_aspect('equal')
            ax_pre.set_title(f"Geometry Preview: {dimx}x{dimy} Domain", color="#0f172a")
            ax_pre.set_xlabel("X (in)", color="#475569"); ax_pre.set_ylabel("Y (in)", color="#475569")
            ax_pre.grid(True, linestyle=':', alpha=0.3)
            # Assuming nx and ny are your number of elements in X and Y
            ax_pre.set_xticks(np.linspace(0, dimx, nelx + 1))
            ax_pre.set_yticks(np.linspace(0, dimy, nely + 1))
            ax_pre.grid(color='#94a3b8', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Optional: Hide the tick labels so it doesn't look cluttered, but keep the grid
            ax_pre.set_xticklabels([])
            ax_pre.set_yticklabels([])
            
            # Hide top and right spines for a cleaner look
            ax_pre.spines['top'].set_visible(False)
            ax_pre.spines['right'].set_visible(False)
            ax_pre.spines['bottom'].set_color('#cbd5e1')
            ax_pre.spines['left'].set_color('#cbd5e1')
            
            handles, labels = ax_pre.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label: ax_pre.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            st.pyplot(fig_pre)
            st.session_state.mesh_data = "Ready"

with col2:
    # Type="primary" will now pick up the blue color from your config.toml
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if st.session_state.mesh_data is None:
            st.error("Please click 'Preview Geometry' first to initialize the mesh!")
        else:
            total_area = dimx * dimy
            target_volume = (total_area * tmin) + (vol_frac * total_area * (tmax - tmin))
            
            status_text = st.empty()
            live_plot_spot = st.empty()
            
            def update_live_view(current_it, current_ch, current_Z):
                fig_live, ax_live = plt.subplots(figsize=(10, 4))
                fig_live.patch.set_alpha(0.0)
                ax_live.axis('off') # Turn off axes for a cleaner live view
                
                ext = [0, dimx, 0, dimy] 
                im = ax_live.imshow(current_Z, cmap=custom_cmap, vmin=0, vmax=tmax, extent=ext, origin='upper')
                plt.colorbar(im, ax=ax_live, label='Thickness (in)')
                
                live_plot_spot.pyplot(fig_live)
                plt.close(fig_live)

                status_text.info(f"⚙️ Optimizing... Iteration: {current_it} | Max Displacement: {current_ch:.4f} in")

            with st.spinner("Crunching the numbers..."):
                X, Y, Thickness, history = logic.run_topology_optimization(
                    dimx, dimy, E, nu, rho, SW, BCMatrix, w_u, 
                    int(nelx), int(nely), target_volume, rmin, tmin, tmax, int(itmax),
                    progress_callback=update_live_view  
                )
                
                status_text.success("✅ Optimization Complete!")
                st.session_state.history = history
                st.session_state.X = X
                st.session_state.Y = Y
                st.session_state.run_finished = True
                st.rerun()

# --- 3. THE RESULTS EXPLORER (INTERACTIVE 3D) ---
if st.session_state.run_finished:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🕒 Interactive 3D Results</div>', unsafe_allow_html=True)

    steps = len(st.session_state.history)
    
# Initialize session states for the 3D viewer
    if "cam_eye" not in st.session_state:
        st.session_state.cam_eye = dict(x=1.2, y=-1.5, z=-0.8) 
    if "cam_up" not in st.session_state:
        st.session_state.cam_up = dict(x=0, y=0, z=1) 
    if "z_scale_val" not in st.session_state:
        st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
    if "view_rev" not in st.session_state:
        st.session_state.view_rev = 0

    # Create the buttons in a row
    view_cols = st.columns(5)
    if view_cols[0].button("⬇️ Bottom (XY)"):
        st.session_state.cam_eye = dict(x=0, y=0, z=-2.5)
        st.session_state.cam_up = dict(x=0, y=1, z=0) 
        st.session_state.view_rev += 1
        
    if view_cols[1].button("➡️ Front (XZ)"):
        st.session_state.cam_eye = dict(x=0, y=-2.5, z=0)
        st.session_state.cam_up = dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
        
    if view_cols[2].button("↗️ Side (YZ)"):
        st.session_state.cam_eye = dict(x=-2.5, y=0, z=0)
        st.session_state.cam_up = dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
        
    if view_cols[3].button("🔄 Reset View"):
        st.session_state.cam_eye = dict(x=1.2, y=-1.5, z=-0.8)
        st.session_state.cam_up = dict(x=0, y=0, z=1)
        st.session_state.view_rev += 1
        
    if view_cols[4].button("📏 True Scale (Z)"):
        st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
        st.session_state.view_rev += 1

    # UI Controls
    col_slider, col_scale = st.columns([2, 1])
    with col_slider:
        idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
    with col_scale:
        z_scale_pct = st.slider("Visual Z-Scale (%)", 0, 100, key="z_scale_val")
    Z_plot = st.session_state.history[idx]
    
    # Node to Element Center Fix
    x_1d = np.unique(st.session_state.X)
    y_1d = np.unique(st.session_state.Y)
    if len(x_1d) == Z_plot.shape[1] + 1:
        x_1d = (x_1d[:-1] + x_1d[1:]) / 2.0
    if len(y_1d) == Z_plot.shape[0] + 1:
        y_1d = (y_1d[:-1] + y_1d[1:]) / 2.0
    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d)

    # Invert Z to hang from the base
    Z_plot_neg = -Z_plot 

    # Custom Colormap
    custom_colorscale = [
        [0.0, '#08306b'], 
        [0.4, '#2563eb'],
        [1.0, '#cbd5e1'] 
    ]

    # Flat Roof Surface
    roof_surface = go.Surface(
        z=np.zeros_like(Z_plot_neg),
        x=X_mesh,
        y=Y_mesh,
        colorscale=[[0, '#cbd5e1'], [1, '#cbd5e1']],
        showscale=False,
        hoverinfo='skip' 
    )

    # Main bottom optimized surface
    bottom_surface = go.Surface(
        z=Z_plot_neg, 
        x=X_mesh, 
        y=Y_mesh, 
        colorscale=custom_colorscale, 
        cmin=-tmax, 
        cmax=0,
        colorbar=dict(title='Thickness (in)', outlinewidth=0, tickfont=dict(color='#475569'))
    )

    fig = go.Figure(data=[roof_surface, bottom_surface])

    # 1. Calculate ratios
    max_dim = max(dimx, dimy)
    z_ratio = z_scale_pct / 100.0

    # 2. Apply layout ALWAYS including both uirevision and the camera
    fig.update_layout(
        uirevision=st.session_state.view_rev, 
        scene=dict(
            xaxis=dict(range=[0, dimx], title='X (in)', backgroundcolor='white', gridcolor='#e2e8f0', showbackground=True),
            yaxis=dict(range=[0, dimy], title='Y (in)', backgroundcolor='white', gridcolor='#e2e8f0', showbackground=True),
            zaxis=dict(range=[-tmax, 0], title='Z (in)', backgroundcolor='white', gridcolor='#e2e8f0', showbackground=True),
            aspectratio=dict(x=dimx/max_dim, y=dimy/max_dim, z=z_ratio),
            camera=dict(
                eye=st.session_state.cam_eye,
                up=st.session_state.cam_up
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Request #5: STL Export Generation ---
    st.subheader("💾 Export Geometry")
    
    def generate_stl(X, Y, Z):
        # Flatten the arrays to create a point cloud
        points2D = np.column_stack([X.flatten(), Y.flatten()])
        Z_flat = Z.flatten()
        
        # Triangulate the 2D grid to create faces
        tri = Delaunay(points2D)
        
        # Create the 3D mesh
        slab_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(tri.simplices):
            for j in range(3):
                slab_mesh.vectors[i][j] = [points2D[f[j], 0], points2D[f[j], 1], Z_flat[f[j]]]
                
        # Save to an in-memory bytes buffer
        buf = io.BytesIO()
        slab_mesh.save('slab.stl', fh=buf)
        return buf.getvalue()

    # Generate the STL using the inverted Z data
    stl_data = generate_stl(X_mesh, Y_mesh, Z_plot_neg)
    
    st.download_button(
        label="📥 Download as .STL File",
        data=stl_data,
        file_name=f"Optimized_Slab_Iter{idx}.stl",
        mime="model/stl",
        type="primary"

    )




























