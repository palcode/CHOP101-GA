import streamlit as st
import folium
from streamlit_folium import folium_static
import time
from gps_driver import GPSDriver
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="GPS Tracker",
    page_icon="🛰️",
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

class GPSTracker:
    def __init__(self):
        self.gps = None
        self.is_tracking = False
        
    def initialize_gps(self, port, baudrate):
        try:
            self.gps = GPSDriver(port=port, baudrate=baudrate)
            return self.gps.connect()
        except Exception as e:
            st.error(f"Error initializing GPS: {e}")
            return False
            
    def stop_tracking(self):
        if self.gps:
            self.gps.disconnect()
        self.is_tracking = False

def create_map(latitude, longitude, zoom=15):
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
    folium.Marker(
        [latitude, longitude],
        popup="Current Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    return m

def main():
    st.title("🛰️ GPS Tracker")
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = GPSTracker()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        port = st.text_input("Port", value="/dev/ttyUSB0")
        baudrate = st.number_input("Baudrate", value=9600, min_value=1200)
        
        if not st.session_state.tracker.is_tracking:
            if st.button("Start Tracking"):
                if st.session_state.tracker.initialize_gps(port, baudrate):
                    st.session_state.tracker.is_tracking = True
                    st.success("GPS initialized successfully!")
                else:
                    st.error("Failed to initialize GPS")
        else:
            if st.button("Stop Tracking"):
                st.session_state.tracker.stop_tracking()
                st.warning("GPS tracking stopped")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("GPS Data")
        data_container = st.container()
    
    with col1:
        st.subheader("Live Location")
        map_container = st.container()
    
    # Update loop
    while st.session_state.tracker.is_tracking:
        gps_data = st.session_state.tracker.gps.read_gps_data()
        
        if gps_data:
            with data_container:
                st.metric("Latitude", f"{gps_data.latitude:.6f}°")
                st.metric("Longitude", f"{gps_data.longitude:.6f}°")
                st.metric("Altitude", f"{gps_data.altitude:.1f}m")
                st.metric("Satellites", gps_data.satellites)
                st.metric("Fix Quality", gps_data.fix_quality)
                st.text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            
            with map_container:
                m = create_map(gps_data.latitude, gps_data.longitude)
                folium_static(m, width=800)
        
        time.sleep(1)

if __name__ == "__main__":
    main() 