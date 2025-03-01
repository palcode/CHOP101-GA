# RGBD-GPS Sensor Fusion System

A real-time sensor fusion system that combines RealSense RGBD camera data with GPS information for improved localization and tracking.

## Features

- Real-time fusion of RGBD camera and GPS data
- Visual odometry using ORB feature detection
- 3D trajectory visualization
- GPS position tracking with map overlay
- Depth visualization and processing
- Interactive web-based interface

## System Components

1. **GPS Driver (`gps_driver.py`)**
   - Handles GPS device communication
   - Parses NMEA sentences
   - Provides real-time GPS data

2. **Sensor Fusion (`sensor_fusion.py`)**
   - Combines visual odometry with GPS data
   - Implements feature detection and matching
   - Provides real-time pose estimation
   - Handles sensor synchronization

3. **Visualization (`fusion_visualizer.py`)**
   - Real-time data visualization
   - Interactive 3D trajectory plotting
   - GPS position mapping
   - RGBD camera feed display

## Requirements

- Python 3.8+
- Intel RealSense Camera
- GPS Module (NMEA compatible)
- Required Python packages (see `requirements.txt`)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Connect your hardware:
   - Intel RealSense camera via USB
   - GPS module via serial port

## Usage

1. Start the visualization system:
```bash
streamlit run fusion_visualizer.py
```

2. In the web interface:
   - Configure GPS port and baudrate
   - Click "Start System" to begin
   - Monitor real-time sensor fusion results

## System Architecture

### Data Flow
```
RealSense Camera → Visual Odometry → Sensor Fusion ←→ GPS Data
         ↓             ↓                ↓
    RGB Image    Feature Detection   Pose Estimation
    Depth Map    Motion Tracking     Position Fusion
         ↓             ↓                ↓
                 Visualization
```

### Key Components

1. **Visual Odometry**
   - ORB feature detection and matching
   - Essential matrix computation
   - Relative pose estimation

2. **GPS Integration**
   - NMEA sentence parsing
   - Position conversion
   - Data synchronization

3. **Sensor Fusion**
   - Weighted position averaging
   - Confidence estimation
   - Trajectory tracking

4. **Visualization**
   - Real-time camera feeds
   - 3D trajectory plotting
   - Interactive map display
   - Sensor statistics

## Configuration

The system can be configured through the web interface:

- GPS Port: Serial port for GPS module (default: /dev/ttyUSB0)
- Baudrate: Communication speed (default: 9600)
- Visualization options in the interface

## Performance Considerations

- Visual odometry requires good lighting and feature-rich environments
- GPS accuracy depends on satellite visibility and signal quality
- System performance may vary based on hardware capabilities

## License

MIT License - Feel free to use and modify as needed.