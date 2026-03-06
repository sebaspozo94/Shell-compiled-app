import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logic  # This imports your compiled logic.so file

st.set_page_config(page_title="Shell Topology Opt", layout="wide")
st.title("Shell Topology Optimization")

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
    dimx = st.number_input("Domain X (mm)", value=240)
    dimy = st.number_input("Domain Y (mm)", value=192)
    nelx = st.slider("Elements X", 20, 480, 120)
    nely = st.slider("Elements Y", 20, 384, 96)

    st.header("🧪 Material Properties")
    E = st.number_input("Elastic Modulus (E)", value=1500000)
    nu = st.slider("Poisson's Ratio (ν)", 0.0, 0.49, 0.3)
    rho = st.number_input("Material Density (ρ)", value=0.0145)
    sw_toggle = st.checkbox("Include Self-Weight", value=True)
    SW = 1 if sw_toggle else 0

    st.header("⚖️ Loads & Constraints")
    w_u = st.number_input("Distributed Load (w_u)", value=0.2778)
    
    st.subheader("Boundary Conditions")
    st.caption("Rows: [x, y, width, height, type (0=Pinned, 1=Fixed)]")
    default_bc = [
        [48, 156, 4, 4, 0], [48, 36, 4, 4, 0], 
        [192, 156, 4, 4, 0], [192, 36, 4, 4, 0]
    ]
    edited_bc = st.data_editor(default_bc, num_rows="dynamic")
    BCMatrix = np.array(edited_bc)

    st.header("🎯 Optimization Params")
    vol_frac = st.slider("Volume Fraction", 0.05, 1.0, 0.3)
    rmin = st.number_input("Filter Radius (rmin)", value=5.0)
    tmin = st.number_input("Min Thickness", value=2.0)
    tmax = st.number_input("Max Thickness", value=12.0)
    itmax = st.number_input("Max Iterations", value=50)

# --- 2. THE MAIN COLUMNS ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Preview Geometry", use_container_width=True):
        with st.spinner("Generating preview..."):
            fig_pre, ax_pre = plt.subplots(figsize=(10, 6))
            domain = patches.Rectangle((0, 0), dimx, dimy, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
            ax_pre.add_patch(domain)
            
            for i in range(BCMatrix.shape[0]):
                x0, y0, bdx, bdy, cond = BCMatrix[i]
                lower_left_x, lower_left_y = x0 - bdx/2, y0 - bdy/2
                color = 'red' if cond == 1 else 'blue'
                label = 'Fixed' if cond == 1 else 'Pinned'
                
                rect = patches.Rectangle(
                    (lower_left_x, lower_left_y), bdx, bdy, 
                    linewidth=1, edgecolor=color, facecolor=color, alpha=0.5,
                    label=f"Support {i+1} ({label})"
                )
                ax_pre.add_patch(rect)
            
            ax_pre.set_xlim(-10, dimx + 10); ax_pre.set_ylim(-10, dimy + 10)
            ax_pre.set_aspect('equal')
            ax_pre.set_title(f"Geometry Preview: {dimx}x{dimy} Domain")
            ax_pre.set_xlabel("X (mm)"); ax_pre.set_ylabel("Y (mm)")
            ax_pre.grid(True, linestyle=':', alpha=0.6)
            
            handles, labels = ax_pre.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label: ax_pre.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            st.pyplot(fig_pre)
            st.session_state.mesh_data = "Ready"

# --- Inside app.py ---

with col2:
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if st.session_state.mesh_data is None:
            st.error("Please click 'Preview Geometry' first to initialize the mesh!")
        else:
            total_area = dimx * dimy
            target_volume = (total_area * tmin) + (vol_frac * total_area * (tmax - tmin))
            
            # 1. Create empty spots for the live feed
            status_text = st.empty()
            live_plot_spot = st.empty()
            
            # 2. Define the callback function that Streamlit will run
            def update_live_view(current_it, current_ch, current_Z):
                status_text.info(f"⚙️ Optimizing... Iteration: {current_it} | Max Change: {current_ch:.4f}")
                
                fig_live, ax_live = plt.subplots(figsize=(10, 4))
                ext = [0, dimx, 0, dimy] # Rough extent for live preview
                im = ax_live.imshow(current_Z, cmap='jet', vmin=0, vmax=tmax, extent=ext, origin='upper')
                plt.colorbar(im, ax=ax_live, label='Thickness (mm)')
                
                # Push to UI and immediately close to save memory
                live_plot_spot.pyplot(fig_live)
                plt.close(fig_live)

            with st.spinner("Crunching the numbers..."):
                # 3. Pass the function into the compiled logic
                X, Y, Thickness, history = logic.run_topology_optimization(
                    dimx, dimy, E, nu, rho, SW, BCMatrix, w_u, 
                    int(nelx), int(nely), target_volume, rmin, tmin, tmax, int(itmax),
                    progress_callback=update_live_view  # <-- PASS IT HERE
                )
                
                status_text.success("✅ Optimization Complete!")
                st.session_state.history = history
                st.session_state.X = X
                st.session_state.Y = Y
                st.session_state.run_finished = True
                st.rerun()

# --- 3. THE RESULTS EXPLORER ---
if st.session_state.run_finished:
    st.divider()
    st.header("🕒 Results Explorer")

    steps = len(st.session_state.history)
    idx = st.slider("Iteration History", 0, steps - 1, steps - 1)

    Z_plot = st.session_state.history[idx]
    fig_res, ax_res = plt.subplots(figsize=(10, 4))

    ext = [st.session_state.X.min(), st.session_state.X.max(),
           st.session_state.Y.min(), st.session_state.Y.max()]

    im = ax_res.imshow(Z_plot, cmap='jet', vmin=0, vmax=tmax, extent=ext, origin='upper')
    plt.colorbar(im, ax=ax_res, label='Thickness (mm)')
    ax_res.set_title(f"Snapshot Index: {idx} of {steps-1}")
    ax_res.set_xlabel("X (mm)"); ax_res.set_ylabel("Y (mm)")
    
    st.pyplot(fig_res)