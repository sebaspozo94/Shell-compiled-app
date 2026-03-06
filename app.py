import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import logic  # This imports your compiled logic.so file

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
    dimx = st.number_input("Domain X (in)", value=240, min_value=1)
    dimy = st.number_input("Domain Y (in)", value=192, min_value=1)
    nelx = st.number_input("Elements X", value=120, min_value=1,max_value=180)
    nely = st.number_input("Elements Y", value=96, min_value=1,max_value=180)

    st.header("🧪 Material Properties")
    E = st.number_input("Elastic Modulus (psi)", value=1500000)
    nu = st.slider("Poisson's Ratio (ν)", 0.0, 0.49, 0.3)
    rho = st.number_input("Material Density (ρ)", value=0.0145)
    sw_toggle = st.checkbox("Include Self-Weight", value=True)
    SW = 1 if sw_toggle else 0

    st.header("⚖️ Loads & Constraints")
    w_u = st.number_input("Distributed Load (w_u)", value=0.2778)
    
    st.subheader("Boundary Conditions")
    st.caption("Rows: [x, y, width, height, type (0=Pin, 1=Fix)]")
    default_bc = [
        [48, 156, 4, 4, 0], [48, 36, 4, 4, 0], 
        [192, 156, 4, 4, 0], [192, 36, 4, 4, 0]
    ]
    edited_bc = st.data_editor(default_bc, num_rows="dynamic", use_container_width=True)
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
            ax_pre.set_xlabel("X (mm)", color="#475569"); ax_pre.set_ylabel("Y (mm)", color="#475569")
            ax_pre.grid(True, linestyle=':', alpha=0.3)
            
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
                status_text.info(f"⚙️ Optimizing... Iteration: {current_it} | Max Change: {current_ch:.4f}")
                
                fig_live, ax_live = plt.subplots(figsize=(10, 4))
                fig_live.patch.set_alpha(0.0)
                ax_live.axis('off') # Turn off axes for a cleaner live view
                
                ext = [0, dimx, 0, dimy] 
                im = ax_live.imshow(current_Z, cmap='jet', vmin=0, vmax=tmax, extent=ext, origin='upper')
                plt.colorbar(im, ax=ax_live, label='Thickness (mm)')
                
                live_plot_spot.pyplot(fig_live)
                plt.close(fig_live)

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

# --- 3. THE RESULTS EXPLORER ---
if st.session_state.run_finished:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🕒 Results Explorer</div>', unsafe_allow_html=True)

    steps = len(st.session_state.history)
    idx = st.slider("Iteration History", 0, steps - 1, steps - 1)

    Z_plot = st.session_state.history[idx]
    fig_res, ax_res = plt.subplots(figsize=(10, 4))
    fig_res.patch.set_alpha(0.0) # Transparent background

    ext = [st.session_state.X.min(), st.session_state.X.max(),
           st.session_state.Y.min(), st.session_state.Y.max()]

    im = ax_res.imshow(Z_plot, cmap='jet', vmin=0, vmax=tmax, extent=ext, origin='upper')
    
    # Style the colorbar
    cbar = plt.colorbar(im, ax=ax_res, label='Thickness (mm)')
    cbar.outline.set_visible(False)
    
    ax_res.set_title(f"Snapshot Index: {idx} of {steps-1}", color="#0f172a")
    ax_res.set_xlabel("X (mm)", color="#475569"); ax_res.set_ylabel("Y (mm)", color="#475569")
    
    # Clean up spines
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False)
    ax_res.spines['bottom'].set_color('#cbd5e1')
    ax_res.spines['left'].set_color('#cbd5e1')
    
    st.pyplot(fig_res)