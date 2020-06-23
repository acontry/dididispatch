import heapq
from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import KDTree as KDTree

from model.agent import Agent
from sim.geo import METERS_PER_DEG_LAT, METERS_PER_DEG_LNG, great_circle_distance, intermediate_point
from sim.grid import Grid

STEP_SEC = 2
MAX_PICKUP_RADIUS_METERS = 2000
DRIVER_SPEED_SEC_PER_METER = 1.0 / 3

CANCEL_PROB_DISTANCES = list(range(200, 2001, 200))


PROCESSED_DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed'


class DriverState(Enum):
    FREE = auto()
    IDLE_MOVING = auto()
    IN_RIDE = auto()


@dataclass
class Driver:
    """Driver.

    Ordered by end time.
    """
    driver_id: str
    lat: float
    lng: float
    start_time: int
    end_time: int  # Shift end time
    available_time: int = 0  # Next time that driver is available (not in a ride)
    destination_lat: float = 0  # Destination used for idle movement
    destination_lng: float = 0
    idle_duration: int = 0
    state: DriverState = DriverState.FREE
    score: float = 0

    def __lt__(self, other):
        return self.end_time < other.end_time

    def assign_ride(self, lat: float, lng: float, available_time: int, score: float):
        """Assign a ride to driver.

        Store location as ride destination. We don't need to track the driver's location until
        they arrive.
        """
        self.lat = lat
        self.lng = lng
        self.available_time = available_time
        self.idle_duration = 0
        self.state = DriverState.IN_RIDE
        self.score += score


@dataclass
class Order:
    order_id: str
    start_time: int
    end_time: int
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    reward: float
    cancel_prob: List[float]

    def create_match(self, driver: Driver, day_of_week: int):
        """Create a potential match with a driver."""
        distance = great_circle_distance(driver.lat, driver.lng, self.pickup_lat, self.pickup_lng)
        return {
            'order_id': self.order_id,
            'driver_id': driver.driver_id,
            'order_driver_distance': distance,
            'order_start_location': [self.pickup_lng, self.pickup_lat],
            'order_finish_location': [self.dropoff_lng, self.dropoff_lat],
            'driver_location': [driver.lng, driver.lat],
            'timestamp': self.start_time,
            'order_finish_timestamp': self.end_time,
            'day_of_week': day_of_week,
            'reward_units': self.reward,
            'pick_up_eta': distance * DRIVER_SPEED_SEC_PER_METER
        }


class Simulator:
    def __init__(self, agent: Agent, ds: str, num_repositionable_drivers: int):
        self.agent = agent
        self.ds = ds
        self.num_repositionable_drivers = num_repositionable_drivers

        # Spatial info
        self.grid = Grid()

        #########
        # Drivers
        self.drivers_queued: deque = self.init_drivers(ds)  # Before go online. Natural order: start_time
        self.drivers_online = dict()  # All online drivers, keyed by driver_id.
        self.drivers_done = []  # Offline drivers.

        # These are secondary data structures to enable fast lookup for driver updates.
        #
        # Online but not in ride. Keyed by driver_id
        self.drivers_available = OrderedDict()
        # Online and in ride. Natural order: available_time. Min-heap of tuples of (available_time, driver_id)
        self.drivers_busy = []

        ########
        # Orders
        self.orders: deque = self.init_orders(ds)
        self.orders_active = OrderedDict()

        ##################
        # Simulation state
        self.day_of_week = datetime.strptime(self.ds, '%Y%m%d').weekday()  # Monday = 0, Sunday = 6
        self.time = min(self.drivers_queued[0].start_time, self.orders[0].start_time)
        self.steps = 0

        self.num_completed = 0
        self.num_cancelled = 0
        self.num_unfulfilled = 0
        self.score = 0
        self.score_cancelled = 0
        self.score_unfulfilled = 0

    @staticmethod
    def init_drivers(ds):
        print('Initializing drivers')
        df = pd.read_parquet(PROCESSED_DATA_PATH / 'drivers' / f'{ds}.parquet')
        drivers = deque()
        for _, row in df.iterrows():
            driver = Driver(row.driver_id, row.start_lat, row.start_lng, row.start_time, row.end_time)
            drivers.append(driver)
        print(f'{len(drivers)} drivers initialized')
        return drivers

    @staticmethod
    def init_orders(ds):
        print('Initializing orders')
        df = pd.read_parquet(PROCESSED_DATA_PATH / 'orders' / f'{ds}.parquet')
        orders = deque()
        for _, row in df.iterrows():
            order = Order(
                row.order_id,
                row.start_time,
                row.stop_time,
                row.pickup_lat,
                row.pickup_lng,
                row.dropoff_lat,
                row.dropoff_lng,
                row.reward,
                row.cancel_prob)
            orders.append(order)
        print('Orders initialized')
        return orders

    def run(self):
        while self.orders:
            self.step()

    def step(self):
        self.time += STEP_SEC
        self.steps += 1

        if (self.steps % 100) == 0:
            print(f'Time: {self.time} | Drivers online: {len(self.drivers_online)} | '
                  f'Drivers available: {len(self.drivers_available)} | Drivers busy: {len(self.drivers_busy)}')

        # Complete routes, moving finished drivers back to available.
        self.complete_routes()
        # Remove drivers that went offline
        self.remove_offline_drivers()
        # Add new drivers that went online
        self.add_online_drivers()

        # Add orders that were made
        self.add_new_orders()

        # Build candidate order-driver pairs
        candidates = self.build_candidates()
        matched = self.agent.dispatch(candidates)
        # TODO: Driver repositioning

        self.process_new_matches(matched)
        self.process_unfulfilled_orders()
        self.process_idle_drivers()

    def complete_routes(self):
        while self.drivers_busy and self.drivers_busy[0][0] <= self.time:
            _, driver_id = heapq.heappop(self.drivers_busy)
            driver = self.drivers_online[driver_id]
            driver.state = DriverState.FREE
            self.drivers_available[driver.driver_id] = driver

    def remove_offline_drivers(self):
        remove = []
        for driver_id, driver in self.drivers_available.items():
            if driver.end_time <= self.time:
                remove.append(driver_id)  # Track which drivers to remove from available
                driver = self.drivers_online.pop(driver_id)  # Pop from online dict
                self.drivers_done.append(driver)  # Append to list
        for driver_id in remove:
            del self.drivers_available[driver_id]

    def add_online_drivers(self):
        while self.drivers_queued and self.drivers_queued[0].start_time <= self.time:
            driver = self.drivers_queued.popleft()
            self.drivers_online[driver.driver_id] = driver  # Main data structure
            self.drivers_available[driver.driver_id] = driver  # Secondary data structure

    def add_new_orders(self):
        while self.orders and self.orders[0].start_time <= self.time:
            order = self.orders.popleft()
            self.orders_active[order.order_id] = order

    def build_candidates(self):
        drivers = [(METERS_PER_DEG_LAT * d.lat, METERS_PER_DEG_LNG * d.lng) for d in self.drivers_available.values()]
        orders = [(METERS_PER_DEG_LAT * order.pickup_lat, METERS_PER_DEG_LNG * order.pickup_lng)
                  for order in self.orders_active.values()]
        order_tree = KDTree(orders)
        driver_tree = KDTree(drivers)

        all_order_matches = order_tree.query_ball_tree(driver_tree, r=MAX_PICKUP_RADIUS_METERS)

        candidates = []
        for order_matches, order in zip(all_order_matches, self.orders_active.values()):
            for driver_idx in order_matches:
                driver = self.drivers_available[driver_idx]
                candidates.append(order.create_match(driver, self.day_of_week))
        return candidates

    def process_new_matches(self, matched):
        for match in matched:
            driver_id = match['driver_id']
            order_id = match['order_id']

            driver = self.drivers_available[driver_id]
            order = self.orders_active[order_id]

            pickup_distance = great_circle_distance(driver.lat, driver.lng, order.pickup_lat, order.pickup_lng)
            if pickup_distance > 2000:
                print('Pickup distance > 2km')
            cancel_prob = np.interp(pickup_distance, CANCEL_PROB_DISTANCES, order.cancel_prob)
            if cancel_prob > np.random.rand():  # Canceled
                self.num_cancelled += 1
                self.score_cancelled += order.reward
            else:  # Not cancelled
                self.num_completed += 1
                self.score += order.reward

                # Update driver
                # Ride duration calculation:
                # - P2 is calculated by great circle distance and 3 m/s moving speed
                # - P3 comes directly from order duration
                available_time = (
                        self.time +
                        int(pickup_distance * DRIVER_SPEED_SEC_PER_METER) +
                        (order.end_time - order.start_time))
                driver.assign_ride(order.dropoff_lat, order.dropoff_lng, available_time, order.reward)

                del self.drivers_available[driver_id]
                del self.orders_active[order_id]
                heapq.heappush(self.drivers_busy, (driver.available_time, driver.driver_id))

    def process_unfulfilled_orders(self):
        """Remaining orders are unfulfilled, so remove them and track stats."""
        self.num_unfulfilled += len(self.orders_active)
        self.score_unfulfilled += np.sum([order.reward for order in self.orders_active.values()])
        self.orders_active.clear()

    def process_idle_drivers(self):
        """Idle driver movement."""
        for driver in self.drivers_available.values():
            driver.idle_duration += STEP_SEC
            # Driver who just idly reached their idle destination are now (temporarily) free
            if driver.state == DriverState.IDLE_MOVING and driver.available_time <= self.time:
                driver.lat = driver.destination_lat
                driver.lng = driver.destination_lng
                driver.state = DriverState.FREE

            # Assign new idle destinations to free driver
            if driver.state == DriverState.FREE:
                # Choose idle transition
                transitions = self.grid.idle_transitions(self.time, self.grid.lookup_grid_id(driver.lat, driver.lng))
                grid_id = np.random.choice(list(transitions.keys()), p=list(transitions.values()))
                lat, lng = self.grid.lookup_coord(grid_id)
                # Update driver
                driver.destination_lat = lat
                driver.destination_lng = lng

                dist = great_circle_distance(driver.lat, driver.lng, driver.destination_lat, driver.destination_lng)
                driver.available_time = self.time + int(DRIVER_SPEED_SEC_PER_METER * dist)
                driver.state = DriverState.IDLE_MOVING

            # Advance driver towards their idle destinations
            if driver.state == DriverState.IDLE_MOVING:
                dist = great_circle_distance(driver.lat, driver.lng, driver.destination_lat, driver.destination_lng)
                frac = min(STEP_SEC / (DRIVER_SPEED_SEC_PER_METER * dist), 1.0)
                lat, lng = intermediate_point(driver.lat, driver.lng, driver.destination_lat, driver.destination_lng, frac)
                driver.lat = lat
                driver.lng = lng
            else:
                raise ValueError(f'Idle driver encountered in state {driver.state}')






if __name__ == '__main__':
    sim = Simulator(Agent(), '20161101', 5)
    sim.run()
