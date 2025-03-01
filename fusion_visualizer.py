import streamlit as st
import folium
from streamlit_folium import folium_static
import cv2
import numpy as np
from sensor_fusion import LocalizationFusion
import time
from datetime import datetime
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add plotly to requirements
# Update requirements.txt with: plotly==5.18.0

# Page configuration
st.set_page_config(
    page_title="Enhanced Sensor Fusion Visualizer",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1v0mbdj {
        width: 100%;
    }
    .plot-container {
        height: 400px;
    }
    </style>
    """, unsafe_allow_html=True)

class EnhancedFusionVisualizer:
    def __init__(self):
        self.fusion = None
        self.is_running = False
        self.current_frame = None
        self.current_depth = None
        self.trajectory = []  # Store position history
        self.lock = threading.Lock()
        
    def start(self, gps_port: str, gps_baudrate: int) -> bool:
        """Start the fusion system"""
        self.fusion = LocalizationFusion(gps_port, gps_baudrate)
        if not self.fusion.start():
            return False
            
        self.is_running = True
        self.frame_thread = threading.Thread(target=self._update_frame)
        self.frame_thread.start()
        return True
        
    def stop(self):
        """Stop the fusion system"""
        self.is_running = False
        if self.fusion:
            self.fusion.stop()
        if hasattr(self, 'frame_thread'):
            self.frame_thread.join()
            
    def _colorize_depth(self, depth_image):
        """Convert depth image to colormap for visualization"""
        if depth_image is None:
            return None
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return depth_colormap
            
    def _update_frame(self):
        """Update the camera frame with visualization overlays"""
        while self.is_running:
            rgbd_data = self.fusion._get_rgbd_data()
            if rgbd_data is None:
                continue
                
            # Get color and depth frames
            color_frame = rgbd_data.color_frame.copy()
            depth_frame = rgbd_data.depth_frame.copy()
            
            # Get current pose
            pose = self.fusion.get_current_pose()
            
            # Store trajectory
            self.trajectory.append(pose.position)
            if len(self.trajectory) > 100:  # Keep last 100 positions
                self.trajectory.pop(0)
            
            # Draw pose information on frame
            cv2.putText(color_frame, f"Confidence: {pose.confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw feature points if available
            if hasattr(self.fusion, 'prev_rgbd_data'):
                prev_frame = cv2.cvtColor(self.fusion.prev_rgbd_data.color_frame, 
                                        cv2.COLOR_BGR2GRAY)
                kp, _ = self.fusion.orb.detectAndCompute(prev_frame, None)
                color_frame = cv2.drawKeypoints(color_frame, kp, None, 
                                             color=(0, 255, 0), 
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            with self.lock:
                self.current_frame = color_frame
                self.current_depth = depth_frame
            
            time.sleep(0.03)  # ~30 FPS
            
    def get_current_data(self):
        """Get the latest frame and depth data"""
        with self.lock:
            if self.current_frame is not None and self.current_depth is not None:
                return {
                    'color': self.current_frame.copy(),
                    'depth': self.current_depth.copy(),
                    'depth_colored': self._colorize_depth(self.current_depth)
                }
            return None

def create_map(latitude: float, longitude: float, trajectory=None, zoom: int = 15):
    """Create a folium map with trajectory"""
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
    
    # Add current position marker
    folium.Marker(
        [latitude, longitude],
        popup="Current Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add trajectory if available
    if trajectory and len(trajectory) > 1:
        trajectory_coords = []
        for pos in trajectory:
            lat = np.degrees(pos[1] / 6371000)  # Earth radius in meters
            lon = np.degrees(pos[0] / (6371000 * np.cos(np.radians(lat))))
            trajectory_coords.append([lat, lon])
        
        folium.PolyLine(
            trajectory_coords,
            weight=2,
            color='blue',
            opacity=0.8
        ).add_to(m)
    
    return m

def create_3d_trajectory(trajectory):
    """Create a 3D trajectory plot using plotly"""
    if not trajectory:
        return None
        
    trajectory = np.array(trajectory)
    fig = go.Figure(data=[go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines+markers',
        marker=dict(
            size=2,
            color=np.linspace(0, 1, len(trajectory)),
            colorscale='Viridis',
        ),
        line=dict(
            color='blue',
            width=2
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=400
    )
    
    return fig

def main():
    st.title("🤖 Enhanced Sensor Fusion Visualizer")
    
    # Initialize session state
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedFusionVisualizer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        gps_port = st.text_input("GPS Port", value="/dev/ttyUSB0")
        gps_baudrate = st.number_input("GPS Baudrate", value=9600, min_value=1200)
        
        if not st.session_state.visualizer.is_running:
            if st.button("Start System"):
                if st.session_state.visualizer.start(gps_port, gps_baudrate):
                    st.success("System started successfully!")
                else:
                    st.error("Failed to start system")
        else:
            if st.button("Stop System"):
                st.session_state.visualizer.stop()
                st.warning("System stopped")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera feeds
        st.subheader("Sensor Data Visualization")
        cam_col1, cam_col2 = st.columns(2)
        
        with cam_col1:
            st.markdown("**RGB Camera Feed**")
            rgb_placeholder = st.empty()
        
        with cam_col2:
            st.markdown("**Depth Visualization**")
            depth_placeholder = st.empty()
        
        # 3D Trajectory
        st.subheader("3D Trajectory")
        trajectory_placeholder = st.empty()
        
        # Map view
        st.subheader("GPS Location")
        map_placeholder = st.empty()
    
    with col2:
        # Sensor fusion data
        st.subheader("Fusion Data")
        data_placeholder = st.empty()
        
        # Feature matching stats
        st.subheader("Visual Odometry Stats")
        vo_stats_placeholder = st.empty()
    
    # Main update loop
    while st.session_state.visualizer.is_running:
        # Get current data
        data = st.session_state.visualizer.get_current_data()
        if data:
            # Update camera feeds
            rgb_placeholder.image(data['color'], channels="BGR", use_column_width=True)
            depth_placeholder.image(data['depth_colored'], channels="BGR", use_column_width=True)
        
        # Get current pose
        pose = st.session_state.visualizer.fusion.get_current_pose()
        
        # Update fusion data display
        with data_placeholder:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Position X", f"{pose.position[0]:.2f}m")
                st.metric("Position Y", f"{pose.position[1]:.2f}m")
                st.metric("Position Z", f"{pose.position[2]:.2f}m")
            with col2:
                st.metric("Orientation", f"[{', '.join([f'{x:.2f}' for x in pose.orientation])}]")
                st.metric("Confidence", f"{pose.confidence:.2f}")
                st.text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Update visual odometry stats
        with vo_stats_placeholder:
            if hasattr(st.session_state.visualizer.fusion, 'prev_rgbd_data'):
                prev_frame = cv2.cvtColor(
                    st.session_state.visualizer.fusion.prev_rgbd_data.color_frame,
                    cv2.COLOR_BGR2GRAY
                )
                kp, _ = st.session_state.visualizer.fusion.orb.detectAndCompute(prev_frame, None)
                st.metric("Feature Points", len(kp) if kp else 0)
        
        # Update 3D trajectory
        trajectory_fig = create_3d_trajectory(st.session_state.visualizer.trajectory)
        if trajectory_fig:
            trajectory_placeholder.plotly_chart(trajectory_fig, use_container_width=True)
        
        # Update map
        # Convert position to lat/lon (simplified)
        EARTH_RADIUS = 6371000  # meters
        lat = np.degrees(pose.position[1] / EARTH_RADIUS)
        lon = np.degrees(pose.position[0] / (EARTH_RADIUS * np.cos(np.radians(lat))))
        
        m = create_map(lat, lon, st.session_state.visualizer.trajectory)
        with map_placeholder:
            folium_static(m)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main() 