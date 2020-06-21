"""Preprocess data for use in simulation.

The following files should be placed in data/raw:
- gps_01-10.zip
- gps_11-20.zip
- gps_21-30.zip
- order_01-30.zip
"""
import csv
import sys
from dataclasses import asdict, dataclass
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

ORDER_FILE = 'order_01-30.zip'
GPS_FILES = ['gps_01-10.zip', 'gps_11-20.zip', 'gps_21-30.zip']


@dataclass
class DriverSchedule:
    driver_id: str
    start_lat: float = 0.0
    start_lng: float = 0.0
    start_time: int = sys.maxsize
    end_time: int = -1


def process_trajectory_file(f, output_file_dir):
    """Build driver schedules from trajectory file and save to disk."""
    ds = f.name.split('_')[1]

    drivers = dict()

    for line in f:
        driver_id, order_id, timestamp, lng, lat = line.decode().split(',')
        timestamp = int(timestamp)

        # Get driver
        if driver_id not in drivers:
            drivers[driver_id] = DriverSchedule(driver_id)
        d = drivers[driver_id]

        # Update driver schedule
        if d.start_time is None or d.start_time > timestamp:
            d.start_time = timestamp
            d.start_lat = float(lat)
            d.start_lng = float(lng)
        d.end_time = max(d.end_time, timestamp)

    # Data filters
    print(f'{len(drivers)} drivers processed for {ds}')
    drivers = {driver_id: driver_schedule for driver_id, driver_schedule in drivers.items() if
               driver_schedule.end_time > driver_schedule.start_time}
    print(f'{len(drivers)} after filtering zero length shifts for {ds}')

    output_file_full_path = output_file_dir / 'drivers'
    output_file_full_path.mkdir(exist_ok=True)

    df = pd.DataFrame([asdict(d) for d in drivers.values()])
    df = df.sort_values(by='start_time', ignore_index=True)

    df.to_csv(output_file_full_path / f'{ds}.csv', index=False)


def main():
    data_path = Path(__file__).parent.parent / 'data'

    raw_data_path = data_path / 'raw'
    interim_data_path = data_path / 'interim'
    processed_data_path = data_path / 'processed'

    interim_data_path.mkdir(exist_ok=True)
    processed_data_path.mkdir(exist_ok=True)

    # Extract orders zip
    if not (interim_data_path / 'hexagon_grid_table.csv').exists():
        with ZipFile(raw_data_path / ORDER_FILE) as z:
            z.extractall(interim_data_path)

    # Process trajectories
    for gps_file in GPS_FILES:
        with ZipFile(raw_data_path / gps_file) as z:
            for name in z.namelist():
                with z.open(name, 'r') as f:
                    process_trajectory_file(f, processed_data_path)


if __name__ == '__main__':
    main()
