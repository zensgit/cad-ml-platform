import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import httpx
import json
import time

# Configuration
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="CAD AI Platform (L4)",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .warning-box {
        border-left: 5px solid #ff4b4b;
        background-color: #ffeaea;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏭 Intelligent CAD Analysis Platform")
    st.markdown("**L4 Capability Demo**: Recognition • DFM Analysis • Process Selection • Cost Estimation")

    # --- Sidebar: Controls ---
    with st.sidebar:
        st.header("1. Upload & Configure")
        uploaded_file = st.file_uploader("Upload 3D Part", type=["step", "stp", "dxf"])
        
        st.subheader("Manufacturing Specs")
        material = st.selectbox("Material", ["Steel", "Aluminum", "Titanium", "Plastic"])
        batch_size = st.number_input("Batch Size", min_value=1, max_value=10000, value=100)
        
        run_btn = st.button("🚀 Run AI Analysis", type="primary")

        st.markdown("---")
        st.info("System Status: **Online (L4-Ready)**")

    # --- Main Area ---
    if not uploaded_file:
        render_landing_page()
        return

    if run_btn and uploaded_file:
        with st.spinner("🚀 Uploading geometry... Parsing B-Rep... Running UV-Net..."):
            # Call API
            try:
                # Prepare form data
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                options = {
                    "extract_features": True,
                    "classify_parts": True,
                    "quality_check": True,
                    "process_recommendation": True,
                    "estimate_cost": True
                }
                data = {
                    "options": json.dumps(options),
                    "material": material.lower()
                }
                
                # Mock API call if server not running (Graceful fallback for demo)
                try:
                    res = httpx.post(f"{API_URL}/analyze/", files=files, data=data, timeout=30.0)
                    if res.status_code == 200:
                        result = res.json()
                        render_results(result, batch_size)
                    else:
                        st.error(f"Analysis failed: {res.text}")
                except httpx.ConnectError:
                    st.warning("⚠️ Backend API not reachable. Showing MOCK data for demonstration.")
                    mock_result = generate_mock_result(uploaded_file.name, material, batch_size)
                    time.sleep(1.5) # Simulate latency
                    render_results(mock_result, batch_size)
                    
            except Exception as e:
                st.error(f"Error: {e}")

def render_landing_page():
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://img.freepik.com/free-vector/3d-printing-industry-isometric-composition_1284-22467.jpg?w=800", caption="Generative Design & DFM")
    with col2:
        st.markdown("""
        ### Why this Platform?
        
        Traditional CAD software tells you dimensions. **We tell you feasibility.**
        
        *   🧠 **AI Recognition**: Instantly identifies "Shaft", "Gear", "Housing".
        *   🛡️ **DFM Guardian**: Detects thin walls, undercuts, and unmachinable features.
        *   💰 **Instant Quoting**: Real-time cost estimation based on geometry & material.
        
        **Ready to start? Upload a STEP file in the sidebar.**
        """)

def render_results(data, batch_size):
    results = data["results"]
    classification = results.get("classification", {})
    quality = results.get("quality", {})
    process = results.get("process", {})
    cost = results.get("cost_estimation", {})
    
    # --- Top Row: Identity & Cost ---
    st.markdown("### 🔍 Analysis Report")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Part Type", classification.get("part_type", "Unknown").title(), 
                  f"{classification.get('confidence', 0)*100:.0f}% Confidence")
    with col2:
        st.metric("Manufacturability", quality.get("manufacturability", "Unknown").title(),
                  f"Score: {quality.get('score', 0)}")
    with col3:
        unit_cost = cost.get("total_unit_cost", 0)
        st.metric("Unit Cost", f"${unit_cost}", f"Batch: {batch_size}")
    with col4:
        total_order = unit_cost * batch_size
        st.metric("Total Order", f"${total_order:,.2f}")

    st.markdown("---")

    # --- Middle Row: 3D & DFM ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("🎨 Geometry & Features")
        # Placeholder 3D Viz
        fig = go.Figure(data=[go.Mesh3d(
            x=[0, 1, 0, 0],
            y=[0, 0, 1, 0],
            z=[0, 0, 0, 1],
            color='lightblue',
            opacity=0.50
        )])
        fig.update_layout(scene=dict(aspectmode='data'), height=300, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        feats = results.get("features_3d", {})
        if feats:
            st.json({k:v for k,v in feats.items() if k not in ["surface_types", "embedding_vector"]}, expanded=False)

    with c2:
        st.subheader("🛠️ DFM & Process")
        
        # Process Card
        prim = process.get("primary_recommendation", {})
        st.success(f"**Recommended Process**: {prim.get('process', 'N/A').replace('_', ' ').title()}")
        st.markdown(f"*{prim.get('reason', '')}*")
        
        # DFM Issues
        issues = quality.get("issues", [])
        if issues:
            st.markdown("#### ⚠️ Manufacturing Risks")
            for issue in issues:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>{issue.get('code')}</strong>: {issue.get('message')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("✅ No DFM issues detected. Ready for production.")

    # --- Bottom: Cost Breakdown ---
    st.subheader("💰 Cost Breakdown")
    breakdown = cost.get("breakdown", {})
    if breakdown:
        b_df = pd.DataFrame([breakdown])
        st.dataframe(b_df, use_container_width=True)
        
        # Pie chart
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig_pie.update_layout(height=300, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Manufacturing Decision ---
    decision = results.get("manufacturing_decision", {})
    if decision:
        st.subheader("🧭 Manufacturing Decision")
        st.json(decision, expanded=False)

def generate_mock_result(filename, material, batch):
    """Fallback if API is down"""
    return {
        "results": {
            "classification": {"part_type": "bracket", "confidence": 0.92},
            "quality": {
                "manufacturability": "medium", 
                "score": 75, 
                "issues": [{"code": "THIN_WALL", "message": "Detected wall thickness < 0.8mm at feature #42."}]
            },
            "process": {
                "primary_recommendation": {
                    "process": "cnc_milling",
                    "reason": "Complex prismatic geometry requires 5-axis milling."
                }
            },
            "cost_estimation": {
                "total_unit_cost": 42.50,
                "breakdown": {
                    "material_cost": 12.00,
                    "machining_cost": 25.50,
                    "setup_amortized": 5.00
                }
            },
            "features_3d": {"volume": 45000, "surface_area": 12000}
        }
    }

if __name__ == "__main__":
    main()
