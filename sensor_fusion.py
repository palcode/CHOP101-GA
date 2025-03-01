#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List
import transforms3d as t3d
from scipy.spatial.transform import Rotation
from gps_driver import GPSDriver, GPSData
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RGBDData:
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp: float

@dataclass
class PoseEstimate:
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [w, x, y, z]
    confidence: float
    timestamp: float

class LocalizationFusion:
    def __init__(self, gps_port: str = "/dev/ttyUSB0", gps_baudrate: int = 9600):
        """
        Initialize the sensor fusion system
        
        Args:
            gps_port: Serial port for GPS module
            gps_baudrate: Baud rate for GPS communication
        """
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Initialize GPS
        self.gps = GPSDriver(port=gps_port, baudrate=gps_baudrate)
        
        # Initialize tracking variables
        self.prev_rgbd_data: Optional[RGBDData] = None
        self.current_pose = PoseEstimate(
            position=np.zeros(3),
            orientation=np.array([1., 0., 0., 0.]),  # Identity quaternion
            confidence=0.0,
            timestamp=0.0
        )
        
        # Feature detection and matching
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Threading control
        self.is_running = False
        self.lock = threading.Lock()
        
    def start(self) -> bool:
        """
        Start the sensor fusion system
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        try:
            # Start RealSense pipeline
            self.pipeline.start(self.config)
            
            # Connect GPS
            if not self.gps.connect():
                logger.error("Failed to connect to GPS")
                self.pipeline.stop()
                return False
            
            self.is_running = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_loop)
            self.process_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting sensor fusion: {e}")
            return False
            
    def stop(self):
        """Stop the sensor fusion system"""
        self.is_running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self.pipeline.stop()
        self.gps.disconnect()
        
    def get_current_pose(self) -> PoseEstimate:
        """Get the latest fused pose estimate"""
        with self.lock:
            return self.current_pose
            
    def _get_rgbd_data(self) -> Optional[RGBDData]:
        """Get the latest RGBD data from RealSense"""
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
        Estimate relative motion using visual odometry
        
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: (relative_position, relative_orientation)
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
        Fuse visual odometry and GPS measurements
        
        Args:
            visual_position: Position estimate from visual odometry
            visual_orientation: Orientation estimate from visual odometry
            gps_data: GPS measurement
            
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
        """Main processing loop for sensor fusion"""
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