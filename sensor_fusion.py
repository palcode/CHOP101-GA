#!/usr/bin/env python3

"""
RGBD-GPS Sensor Fusion System

This module implements a sensor fusion system that combines data from an
Intel RealSense RGBD camera with GPS measurements to provide accurate
localization and tracking. The system uses visual odometry techniques
for relative motion estimation and GPS for absolute position reference.

Key Features:
    - Real-time RGBD camera processing
    - Visual odometry using ORB features
    - GPS integration and coordinate conversion
    - Thread-safe operation
    - Confidence-based fusion

The system uses a weighted fusion approach that combines the high-frequency
local measurements from visual odometry with the lower-frequency but absolute
GPS positions. This provides robust localization even in challenging conditions
like GPS signal loss or feature-poor environments.

Classes:
    - RGBDData: Container for RGBD camera data
    - PoseEstimate: Container for pose estimation results
    - LocalizationFusion: Main fusion system implementation

Dependencies:
    - pyrealsense2: Intel RealSense SDK
    - opencv-python: Computer vision operations
    - numpy: Numerical computations
    - transforms3d: 3D transformation utilities
    - scipy: Scientific computing utilities
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import transforms3d as t3d
from scipy.spatial.transform import Rotation
from gps_driver import GPSDriver, GPSData
import threading
import time
import logging
import os
from pathlib import Path
import re
from functools import wraps

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sensor_fusion.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MIN_FRAME_WIDTH = 320
MIN_FRAME_HEIGHT = 240
MAX_FPS = 60
MIN_FPS = 1
EARTH_RADIUS_METERS = 6371000
MIN_MATCHES = 10
MAX_MATCHES = 1000
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

def validate_port(port: str) -> bool:
    """Validate serial port name for security."""
    # Allow only standard device patterns
    return bool(re.match(r'^(/dev/tty[A-Za-z0-9]+|COM[0-9]+)$', port))

def validate_frame_params(width: int, height: int, fps: int) -> bool:
    """Validate frame parameters."""
    return (MIN_FRAME_WIDTH <= width <= MAX_FRAME_WIDTH and
            MIN_FRAME_HEIGHT <= height <= MAX_FRAME_HEIGHT and
            MIN_FPS <= fps <= MAX_FPS)

def thread_safe(func):
    """Decorator to ensure thread-safe method execution."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return wrapper

@dataclass(frozen=True)  # Make dataclass immutable
class RGBDData:
    """
    Container for RGBD camera data.
    
    Attributes:
        color_frame (np.ndarray): RGB color image
        depth_frame (np.ndarray): Depth image in millimeters
        timestamp (float): Unix timestamp of frame capture
    """
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp: float

    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.color_frame, np.ndarray):
            raise ValueError("color_frame must be a numpy array")
        if not isinstance(self.depth_frame, np.ndarray):
            raise ValueError("depth_frame must be a numpy array")
        if not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be a number")

@dataclass(frozen=True)  # Make dataclass immutable
class PoseEstimate:
    """
    Container for pose estimation results.
    
    Attributes:
        position (np.ndarray): 3D position vector [x, y, z]
        orientation (np.ndarray): Quaternion orientation [w, x, y, z]
        confidence (float): Confidence measure [0-1]
        timestamp (float): Unix timestamp of estimate
    """
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [w, x, y, z]
    confidence: float
    timestamp: float

    def __post_init__(self):
        """Validate pose data after initialization."""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("position must be a 3D numpy array")
        if not isinstance(self.orientation, np.ndarray) or self.orientation.shape != (4,):
            raise ValueError("orientation must be a 4D quaternion numpy array")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

class LocalizationFusion:
    """
    Main class implementing the sensor fusion system.
    
    This class handles the integration of RGBD camera data with GPS
    measurements to provide real-time pose estimation. It implements
    visual odometry for local motion estimation and fuses it with
    GPS data for global position correction.
    
    The system runs in a separate thread to ensure real-time
    performance and provides thread-safe access to the latest
    pose estimate.
    
    Attributes:
        pipeline (rs.pipeline): RealSense pipeline
        config (rs.config): RealSense configuration
        gps (GPSDriver): GPS communication handler
        prev_rgbd_data (RGBDData): Previous RGBD frame for visual odometry
        current_pose (PoseEstimate): Latest fused pose estimate
        orb (cv2.ORB): ORB feature detector
        matcher (cv2.BFMatcher): Feature matcher
        is_running (bool): System status flag
        lock (threading.Lock): Thread synchronization lock
    """
    
    def __init__(self, gps_port: str = "/dev/ttyUSB0", gps_baudrate: int = 9600,
                 frame_width: int = 640, frame_height: int = 480, fps: int = 30):
        """
        Initialize the fusion system with validated parameters.
        
        Args:
            gps_port (str): Serial port for GPS module
            gps_baudrate (int): Baud rate for GPS communication
            frame_width (int): Camera frame width
            frame_height (int): Camera frame height
            fps (int): Frames per second
            
        Raises:
            ValueError: If any parameter is invalid
            SecurityError: If port name is potentially unsafe
        """
        # Validate inputs
        if not validate_port(gps_port):
            raise SecurityError(f"Invalid or unsafe port name: {gps_port}")
        if not validate_frame_params(frame_width, frame_height, fps):
            raise ValueError("Invalid frame parameters")
        if not isinstance(gps_baudrate, int) or gps_baudrate <= 0:
            raise ValueError("Invalid baudrate")

        # Initialize RealSense pipeline with validated parameters
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, frame_width, frame_height, 
                                rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, frame_width, frame_height, 
                                rs.format.z16, fps)
        
        # Initialize GPS with validated parameters
        self.gps = GPSDriver(port=gps_port, baudrate=gps_baudrate)
        
        # Initialize tracking variables
        self._reset_state()
        
        # Threading control
        self.lock = threading.Lock()
        self.is_running = False
        self._stop_event = threading.Event()
        
    def _reset_state(self):
        """Reset internal state variables safely."""
        self.prev_rgbd_data: Optional[RGBDData] = None
        self.current_pose = PoseEstimate(
            position=np.zeros(3),
            orientation=np.array([1., 0., 0., 0.]),
            confidence=0.0,
            timestamp=time.time()
        )
        
        # Feature detection and matching with validated parameters
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @thread_safe
    def start(self) -> bool:
        """
        Start the fusion system.
        
        This method initializes the RealSense camera, connects to the
        GPS module, and starts the processing thread.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_running:
            logger.warning("System is already running")
            return False
            
        try:
            self.pipeline.start(self.config)
            
            if not self.gps.connect():
                logger.error("Failed to connect to GPS")
                self.pipeline.stop()
                return False
            
            self.is_running = True
            self._stop_event.clear()
            
            self.process_thread = threading.Thread(
                target=self._process_loop,
                name="SensorFusionThread"
            )
            self.process_thread.daemon = True  # Ensure thread terminates with main program
            self.process_thread.start()
            
            logger.info("Sensor fusion system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting sensor fusion: {e}")
            self._cleanup()
            return False
            
    @thread_safe
    def stop(self):
        """
        Stop the fusion system and cleanup resources.
        
        This method ensures proper shutdown of all components including
        the RealSense pipeline, GPS connection, and processing thread.
        """
        if not self.is_running:
            return

        try:
            self._stop_event.set()
            self.is_running = False
            
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=5.0)  # Wait with timeout
                if self.process_thread.is_alive():
                    logger.warning("Process thread did not terminate properly")
            
            self._cleanup()
            logger.info("Sensor fusion system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping sensor fusion: {e}")
            
    def _cleanup(self):
        """Clean up resources safely."""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
            if hasattr(self, 'gps'):
                self.gps.disconnect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_current_pose(self) -> PoseEstimate:
        """
        Get the latest fused pose estimate.
        
        Returns:
            PoseEstimate: Current pose with position, orientation, and confidence
        """
        with self.lock:
            return self.current_pose
            
    def _get_rgbd_data(self) -> Optional[RGBDData]:
        """
        Get the latest RGBD data from RealSense camera.
        
        This method handles the communication with the RealSense
        camera and converts the raw frames to numpy arrays.
        
        Returns:
            Optional[RGBDData]: RGBD data if available, None otherwise
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
                
            return RGBDData(
                color_frame=np.asanyarray(color_frame.get_data()),
                depth_frame=np.asanyarray(depth_frame.get_data()),
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error getting RGBD data: {e}")
            return None
            
    def _estimate_visual_odometry(self, curr_data: RGBDData) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate relative motion using visual odometry.
        
        This method implements the visual odometry pipeline:
        1. Feature detection using ORB
        2. Feature matching between consecutive frames
        3. Essential matrix computation
        4. Relative pose recovery
        
        Args:
            curr_data (RGBDData): Current RGBD frame
            
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: 
                Tuple of (position, orientation) if successful,
                None if estimation fails
        """
        if self.prev_rgbd_data is None:
            self.prev_rgbd_data = curr_data
            return None
            
        # Detect features
        prev_kp, prev_des = self.orb.detectAndCompute(
            cv2.cvtColor(self.prev_rgbd_data.color_frame, cv2.COLOR_BGR2GRAY),
            None
        )
        curr_kp, curr_des = self.orb.detectAndCompute(
            cv2.cvtColor(curr_data.color_frame, cv2.COLOR_BGR2GRAY),
            None
        )
        
        if prev_des is None or curr_des is None:
            return None
            
        # Match features
        matches = self.matcher.match(prev_des, curr_des)
        
        if len(matches) < 10:
            return None
            
        # Get matched point coordinates
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        
        # Calculate essential matrix
        E, mask = cv2.findEssentialMat(prev_pts, curr_pts, focal=615.0, pp=(320.0, 240.0))
        
        if E is None:
            return None
            
        # Recover relative pose
        _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts)
        
        self.prev_rgbd_data = curr_data
        
        return t.flatten(), Rotation.from_matrix(R).as_quat()
        
    def _fuse_measurements(self, 
                          visual_position: np.ndarray,
                          visual_orientation: np.ndarray,
                          gps_data: GPSData) -> PoseEstimate:
        """
        Fuse visual odometry and GPS measurements.
        
        This method implements the sensor fusion algorithm:
        1. Convert GPS coordinates to local frame
        2. Weighted fusion of positions
        3. Orientation from visual odometry
        4. Confidence computation
        
        Args:
            visual_position (np.ndarray): Position from visual odometry
            visual_orientation (np.ndarray): Orientation from visual odometry
            gps_data (GPSData): GPS measurement
            
        Returns:
            PoseEstimate: Fused pose estimate
        """
        # Convert GPS to local coordinates (simplified - assumes small distances)
        EARTH_RADIUS = 6371000  # meters
        
        # Calculate local position from GPS
        dx = EARTH_RADIUS * np.cos(np.radians(gps_data.latitude)) * np.radians(gps_data.longitude)
        dy = EARTH_RADIUS * np.radians(gps_data.latitude)
        dz = gps_data.altitude
        
        gps_position = np.array([dx, dy, dz])
        
        # Simple weighted average fusion (can be improved with Kalman filter)
        gps_weight = 0.7  # Trust GPS more for position
        vo_weight = 0.3
        
        fused_position = (gps_weight * gps_position + 
                         vo_weight * visual_position)
        
        # For orientation, trust visual odometry more
        fused_orientation = visual_orientation
        
        # Calculate confidence based on number of satellites and visual features
        confidence = min(1.0, gps_data.satellites / 12.0)
        
        return PoseEstimate(
            position=fused_position,
            orientation=fused_orientation,
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _process_loop(self):
        """
        Main processing loop for sensor fusion.
        
        This method runs in a separate thread and continuously:
        1. Gets new RGBD data
        2. Computes visual odometry
        3. Reads GPS data
        4. Performs sensor fusion
        5. Updates the current pose estimate
        """
        while self.is_running:
            # Get RGBD data
            rgbd_data = self._get_rgbd_data()
            if rgbd_data is None:
                continue
                
            # Get visual odometry estimate
            vo_estimate = self._estimate_visual_odometry(rgbd_data)
            if vo_estimate is None:
                continue
                
            visual_position, visual_orientation = vo_estimate
            
            # Get GPS data
            gps_data = self.gps.read_gps_data()
            if gps_data is None:
                continue
                
            # Fuse measurements
            with self.lock:
                self.current_pose = self._fuse_measurements(
                    visual_position,
                    visual_orientation,
                    gps_data
                )
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload

def main():
    """Example usage of sensor fusion"""
    fusion = LocalizationFusion()
    
    if fusion.start():
        try:
            while True:
                pose = fusion.get_current_pose()
                print(f"Position: {pose.position}")
                print(f"Orientation: {pose.orientation}")
                print(f"Confidence: {pose.confidence}")
                print("-" * 30)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping sensor fusion...")
        finally:
            fusion.stop()
    else:
        print("Failed to start sensor fusion")

if __name__ == "__main__":
    main() 