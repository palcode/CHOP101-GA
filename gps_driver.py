#!/usr/bin/env python3

import serial
import pynmea2
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GPSData:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    timestamp: str = ""
    satellites: int = 0
    fix_quality: int = 0

class GPSDriver:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        """
        Initialize GPS driver
        
        Args:
            port (str): Serial port for GPS module
            baudrate (int): Baud rate for serial communication
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.gps_data = GPSData()
        
    def connect(self) -> bool:
        """
        Establish connection with GPS module
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            return True
        except serial.SerialException as e:
            print(f"Error connecting to GPS: {e}")
            return False
            
    def disconnect(self):
        """Close the serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            
    def read_gps_data(self) -> Optional[GPSData]:
        """
        Read and parse GPS data
        
        Returns:
            Optional[GPSData]: GPS data if successfully parsed, None otherwise
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
            
        try:
            line = self.serial_connection.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                self.gps_data.latitude = msg.latitude
                self.gps_data.longitude = msg.longitude
                self.gps_data.altitude = msg.altitude
                self.gps_data.timestamp = msg.timestamp.strftime("%H:%M:%S")
                self.gps_data.satellites = msg.num_sats
                self.gps_data.fix_quality = msg.gps_qual
                return self.gps_data
                
        except (serial.SerialException, pynmea2.ParseError, UnicodeDecodeError) as e:
            print(f"Error reading GPS data: {e}")
        return None

def main():
    """Example usage of GPS driver"""
    gps = GPSDriver()
    
    if gps.connect():
        print("GPS connected successfully!")
        try:
            while True:
                gps_data = gps.read_gps_data()
                if gps_data:
                    print(f"Latitude: {gps_data.latitude}°")
                    print(f"Longitude: {gps_data.longitude}°")
                    print(f"Altitude: {gps_data.altitude}m")
                    print(f"Time: {gps_data.timestamp}")
                    print(f"Satellites: {gps_data.satellites}")
                    print(f"Fix Quality: {gps_data.fix_quality}")
                    print("-" * 30)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping GPS reading...")
        finally:
            gps.disconnect()
    else:
        print("Failed to connect to GPS")

if __name__ == "__main__":
    main() 