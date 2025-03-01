# RGBD-GPS Sensor Fusion System

A comprehensive sensor fusion system that integrates RGBD camera data with GPS information for accurate localization and tracking. The system uses visual odometry techniques for relative motion estimation and GPS for absolute position reference.

## Features

- Real-time RGBD camera processing using Intel RealSense
- Visual odometry using ORB features
- GPS integration and coordinate conversion
- Thread-safe operation
- Confidence-based sensor fusion
- Interactive visualization dashboard
- Docker support for easy deployment

## Prerequisites

- Python 3.8 or higher
- Intel RealSense SDK 2.0
- GPS module (compatible with NMEA protocol)
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rgbd-gps-fusion.git
cd rgbd-gps-fusion
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. Build the Docker image:
```bash
docker build -t rgbd-gps-fusion .
```

2. Run the container:
```bash
docker run --privileged \
    -v /dev:/dev \
    -p 8501:8501 \
    --device=/dev/ttyUSB0:/dev/ttyUSB0 \
    rgbd-gps-fusion
```

Note: The `--privileged` flag and device mapping are required for accessing the RealSense camera and GPS module.

## Usage

### Running the Sensor Fusion System

1. Start the main sensor fusion system:
```bash
python sensor_fusion.py
```

2. Launch the visualization dashboard:
```bash
streamlit run fusion_visualizer.py
```

3. Access the dashboard at `http://localhost:8501`

### Configuration

The system can be configured through the web interface or by modifying the following parameters:

- GPS Settings:
  - Port: Default `/dev/ttyUSB0`
  - Baudrate: Default `9600`

- Camera Settings:
  - Resolution: 640x480 (default)
  - FPS: 30 (default)

### Visualization Features

The dashboard provides:
- Real-time RGB and depth camera feeds
- 3D trajectory visualization
- GPS position on interactive map
- Sensor fusion statistics
- Visual odometry metrics

## Project Structure

```
rgbd-gps-fusion/
├── sensor_fusion.py      # Core sensor fusion implementation
├── fusion_visualizer.py  # Visualization dashboard
├── gps_driver.py        # GPS communication module
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md          # Documentation
```

## Docker Support

The system is containerized using Docker for easy deployment. The Dockerfile includes:
- Base image with Python and required system libraries
- RealSense SDK installation
- Application code and dependencies
- Exposed ports for the web interface
- Device mappings for hardware access

## Troubleshooting

1. GPS Connection Issues:
   - Verify the correct port in device manager
   - Check USB connections
   - Ensure proper permissions

2. Camera Issues:
   - Verify RealSense SDK installation
   - Check USB 3.0 connection
   - Update firmware if needed

3. Visualization Issues:
   - Check browser compatibility
   - Verify port accessibility
   - Check system resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel RealSense Team
- OpenCV Community
- Streamlit Team