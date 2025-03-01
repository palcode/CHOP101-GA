"""
RGBD-GPS Sensor Fusion Visualizer

This module provides a real-time visualization interface for the sensor fusion system
that combines RealSense RGBD camera data with GPS information. It uses Streamlit
for the web interface and provides multiple visualization components including:
- RGB and depth camera feeds
- 3D trajectory visualization
- GPS position mapping
- Sensor fusion statistics

The visualization system runs in a multi-threaded environment to ensure smooth
real-time updates while maintaining responsive user interface.

Key Components:
    - EnhancedFusionVisualizer: Main class handling visualization and data processing
    - create_map: Creates an interactive map with trajectory
    - create_3d_trajectory: Generates 3D trajectory visualization
    - main: Entry point and UI layout manager

Dependencies:
    - streamlit: Web interface framework
    - folium: Map visualization
    - opencv-python: Image processing
    - plotly: 3D trajectory visualization
    - numpy: Numerical computations
    - sensor_fusion: Core sensor fusion functionality
"""

import streamlit as st
import folium
from streamlit_folium import folium_static
import cv2
import numpy as np
from sensor_fusion import LocalizationFusion, PoseEstimate
import time
from datetime import datetime
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualizer.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_TRAJECTORY_POINTS = 100
MIN_UPDATE_INTERVAL = 0.01  # seconds
MAX_UPDATE_INTERVAL = 1.0   # seconds
DEFAULT_MAP_ZOOM = 15
EARTH_RADIUS_METERS = 6371000
DEFAULT_PLOT_HEIGHT = 400
MAX_RETRIES = 3

class VisualizerError(Exception):
    """Custom exception for visualizer errors."""
    pass

class EnhancedFusionVisualizer:
    """
    A class that handles the visualization of sensor fusion data.
    
    This class manages the real-time visualization of RGBD camera feeds,
    GPS data, and fusion results. It runs in a separate thread to ensure
    smooth updates and responsive UI.
    
    Attributes:
        fusion (LocalizationFusion): The sensor fusion system instance
        is_running (bool): Flag indicating if the system is active
        current_frame (np.ndarray): Latest RGB frame from camera
        current_depth (np.ndarray): Latest depth frame from camera
        trajectory (list): History of positions for trajectory visualization
        lock (threading.Lock): Thread synchronization lock
    """
    
    def __init__(self):
        """Initialize the visualization system with safety checks."""
        self.fusion: Optional[LocalizationFusion] = None
        self.is_running: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None
        self.trajectory: List[np.ndarray] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_update = time.time()
        
    def validate_gps_params(self, port: str, baudrate: int) -> bool:
        """
        Validate GPS parameters for security.
        
        Args:
            port (str): Serial port name
            baudrate (int): Communication baudrate
            
        Returns:
            bool: True if parameters are valid
        """
        # Validate port name
        if not isinstance(port, str) or not port.strip():
            logger.error("Invalid port name")
            return False
            
        # Validate baudrate
        if not isinstance(baudrate, int) or baudrate <= 0:
            logger.error("Invalid baudrate")
            return False
            
        return True
        
    def start(self, gps_port: str, gps_baudrate: int) -> bool:
        """
        Start the fusion and visualization system.
        
        Args:
            gps_port (str): Serial port for GPS module
            gps_baudrate (int): Baud rate for GPS communication
            
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_running:
            logger.warning("System is already running")
            return False
            
        if not self.validate_gps_params(gps_port, gps_baudrate):
            return False
            
        try:
            self.fusion = LocalizationFusion(gps_port, gps_baudrate)
            if not self.fusion.start():
                return False
                
            self.is_running = True
            self._stop_event.clear()
            self.frame_thread = threading.Thread(
                target=self._update_frame,
                name="VisualizerThread"
            )
            self.frame_thread.daemon = True
            self.frame_thread.start()
            
            logger.info("Visualization system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting visualization: {e}")
            self._cleanup()
            return False
        
    def stop(self):
        """
        Stop the visualization system and cleanup resources.
        
        This method ensures proper shutdown of all components including
        the fusion system and processing thread.
        """
        if not self.is_running:
            return
            
        try:
            self._stop_event.set()
            self.is_running = False
            
            if hasattr(self, 'frame_thread'):
                self.frame_thread.join(timeout=5.0)
                if self.frame_thread.is_alive():
                    logger.warning("Frame thread did not terminate properly")
            
            self._cleanup()
            logger.info("Visualization system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping visualization: {e}")
            
    def _cleanup(self):
        """Clean up resources safely."""
        try:
            if self.fusion:
                self.fusion.stop()
            self.fusion = None
            self.current_frame = None
            self.current_depth = None
            self.trajectory.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def _colorize_depth(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert depth image to colormap for better visualization.
        
        Args:
            depth_image (np.ndarray): Raw depth image
            
        Returns:
            np.ndarray: Colorized depth image using JET colormap
        """
        if depth_image is None or not isinstance(depth_image, np.ndarray):
            return None
            
        try:
            normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            normalized_depth = normalized_depth.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            return depth_colormap
        except Exception as e:
            logger.error(f"Error colorizing depth image: {e}")
            return None
            
    def _update_frame(self):
        """
        Main processing loop for frame updates.
        
        This method runs in a separate thread and continuously updates
        the visualization data including camera feeds, pose information,
        and trajectory history.
        """
        retry_count = 0
        
        while not self._stop_event.is_set():
            try:
                # Rate limiting
                current_time = time.time()
                elapsed = current_time - self._last_update
                if elapsed < MIN_UPDATE_INTERVAL:
                    time.sleep(MIN_UPDATE_INTERVAL - elapsed)
                
                rgbd_data = self.fusion._get_rgbd_data()
                if rgbd_data is None:
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        logger.error("Failed to get RGBD data after maximum retries")
                        break
                    continue
                
                retry_count = 0  # Reset counter on successful read
                
                # Thread-safe data update
                with self.lock:
                    self._update_visualization_data(rgbd_data)
                
                self._last_update = time.time()
                
            except Exception as e:
                logger.error(f"Error in visualization update: {e}")
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    break
                    
    def _update_visualization_data(self, rgbd_data):
        """Update visualization data thread-safely."""
        try:
            color_frame = rgbd_data.color_frame.copy()
            depth_frame = rgbd_data.depth_frame.copy()
            
            pose = self.fusion.get_current_pose()
            
            # Update trajectory with bounds checking
            self.trajectory.append(pose.position)
            if len(self.trajectory) > MAX_TRAJECTORY_POINTS:
                self.trajectory.pop(0)
            
            # Update frame with safety checks
            if color_frame.size > 0 and depth_frame.size > 0:
                self.current_frame = color_frame
                self.current_depth = depth_frame
                
        except Exception as e:
            logger.error(f"Error updating visualization data: {e}")
            
    def get_current_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the latest sensor data for visualization.
        
        Returns:
            dict: Dictionary containing:
                - 'color': RGB camera frame
                - 'depth': Raw depth frame
                - 'depth_colored': Colorized depth visualization
        """
        with self.lock:
            if self.current_frame is None or self.current_depth is None:
                return None
                
            try:
                return {
                    'color': self.current_frame.copy(),
                    'depth': self.current_depth.copy(),
                    'depth_colored': self._colorize_depth(self.current_depth)
                }
            except Exception as e:
                logger.error(f"Error getting current data: {e}")
                return None

def create_map(latitude: float, longitude: float, 
              trajectory: Optional[List[np.ndarray]] = None,
              zoom: int = DEFAULT_MAP_ZOOM) -> folium.Map:
    """
    Create an interactive map with current position and trajectory.
    
    Args:
        latitude (float): Current latitude
        longitude (float): Current longitude
        trajectory (list, optional): List of previous positions
        zoom (int, optional): Initial map zoom level
        
    Returns:
        folium.Map: Interactive map with markers and trajectory
    """
    try:
        # Validate inputs
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            raise ValueError("Invalid coordinates")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Coordinates out of range")
            
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
        
        # Add current position
        folium.Marker(
            [latitude, longitude],
            popup="Current Location",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add trajectory with validation
        if trajectory and len(trajectory) > 1:
            trajectory_coords = []
            for pos in trajectory:
                if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                    logger.warning("Invalid trajectory point, skipping")
                    continue
                    
                lat = np.degrees(pos[1] / EARTH_RADIUS_METERS)
                lon = np.degrees(pos[0] / (EARTH_RADIUS_METERS * np.cos(np.radians(lat))))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    trajectory_coords.append([lat, lon])
            
            if trajectory_coords:
                folium.PolyLine(
                    trajectory_coords,
                    weight=2,
                    color='blue',
                    opacity=0.8
                ).add_to(m)
        
        return m
        
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        # Return a default map centered at (0, 0)
        return folium.Map(location=[0, 0], zoom_start=2)

def create_3d_trajectory(trajectory: List[np.ndarray]) -> Optional[go.Figure]:
    """
    Create an interactive 3D plot of the movement trajectory.
    
    Args:
        trajectory (list): List of position vectors
        
    Returns:
        go.Figure: Plotly figure with 3D trajectory visualization
    """
    if not trajectory or not all(isinstance(p, np.ndarray) and p.shape == (3,) 
                               for p in trajectory):
        return None
        
    try:
        trajectory_array = np.array(trajectory)
        fig = go.Figure(data=[go.Scatter3d(
            x=trajectory_array[:, 0],
            y=trajectory_array[:, 1],
            z=trajectory_array[:, 2],
            mode='lines+markers',
            marker=dict(
                size=2,
                color=np.linspace(0, 1, len(trajectory_array)),
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
            height=DEFAULT_PLOT_HEIGHT
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating 3D trajectory: {e}")
        return None

def main():
    """
    Main entry point for the visualization application.
    
    This function sets up the Streamlit interface and manages the
    real-time updates of all visualization components. It handles:
    - User interface layout
    - Configuration options
    - Real-time data updates
    - Visualization components
    """
    try:
        st.title("🤖 Enhanced Sensor Fusion Visualizer")
        
        # Initialize session state safely
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = EnhancedFusionVisualizer()
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            gps_port = st.text_input("GPS Port", value="/dev/ttyUSB0")
            gps_baudrate = st.number_input("GPS Baudrate", value=9600, min_value=1200)
            
            if not st.session_state.visualizer.is_running:
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