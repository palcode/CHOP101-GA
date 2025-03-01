import streamlit as st
import folium
from streamlit_folium import folium_static
import cv2
import numpy as np
from sensor_fusion import LocalizationFusion
import time
from datetime import datetime
import threading

# Page configuration
st.set_page_config(
    page_title="Sensor Fusion Visualizer",
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
    </style>
    """, unsafe_allow_html=True)

class FusionVisualizer:
    def __init__(self):
        self.fusion = None
        self.is_running = False
        self.current_frame = None
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
            
    def _update_frame(self):
        """Update the camera frame with visualization overlays"""
        while self.is_running:
            rgbd_data = self.fusion._get_rgbd_data()
            if rgbd_data is None:
                continue
                
            # Get color frame
            frame = rgbd_data.color_frame.copy()
            
            # Get current pose
            pose = self.fusion.get_current_pose()
            
            # Draw pose information on frame
            cv2.putText(frame, f"Confidence: {pose.confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw feature points if available
            if hasattr(self.fusion, 'prev_rgbd_data'):
                prev_frame = cv2.cvtColor(self.fusion.prev_rgbd_data.color_frame, 
                                        cv2.COLOR_BGR2GRAY)
                kp, _ = self.fusion.orb.detectAndCompute(prev_frame, None)
                frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
            
            with self.lock:
                self.current_frame = frame
            
            time.sleep(0.03)  # ~30 FPS
            
    def get_current_frame(self):
        """Get the latest frame with visualizations"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

def create_map(latitude: float, longitude: float, zoom: int = 15):
    """Create a folium map centered at the given coordinates"""
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
    folium.Marker(
        [latitude, longitude],
        popup="Current Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    return m

def main():
    st.title("🤖 Sensor Fusion Visualizer")
    
    # Initialize session state
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = FusionVisualizer()
    
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
        # Camera feed and visualization
        st.subheader("Camera Feed & Visual Odometry")
        camera_placeholder = st.empty()
        
        # Map view
        st.subheader("Location Map")
        map_placeholder = st.empty()
    
    with col2:
        # Sensor data
        st.subheader("Fusion Data")
        data_placeholder = st.empty()
    
    # Main update loop
    while st.session_state.visualizer.is_running:
        # Update camera feed
        frame = st.session_state.visualizer.get_current_frame()
        if frame is not None:
            camera_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        # Get current pose
        pose = st.session_state.visualizer.fusion.get_current_pose()
        
        # Update data display
        with data_placeholder:
            st.metric("Position X", f"{pose.position[0]:.2f}m")
            st.metric("Position Y", f"{pose.position[1]:.2f}m")
            st.metric("Position Z", f"{pose.position[2]:.2f}m")
            st.metric("Orientation", f"[{', '.join([f'{x:.2f}' for x in pose.orientation])}]")
            st.metric("Confidence", f"{pose.confidence:.2f}")
            st.text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Update map
        # Convert position to lat/lon (simplified)
        EARTH_RADIUS = 6371000  # meters
        lat = np.degrees(pose.position[1] / EARTH_RADIUS)
        lon = np.degrees(pose.position[0] / (EARTH_RADIUS * np.cos(np.radians(lat))))
        
        m = create_map(lat, lon)
        with map_placeholder:
            folium_static(m)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main() 