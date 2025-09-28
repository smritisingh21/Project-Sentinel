import matplotlib.pyplot as plt
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary
from math import sqrt

def get_inputs():
    shape = input("Enter the shape of the area (square/rectangle): ").lower()
    if shape == "square":
        side = int(input("Enter the side length of the square: "))
        width = height = side
    elif shape == "rectangle":
        width = int(input("Enter the width of the rectangle: "))
        height = int(input("Enter the height of the rectangle: "))
    else:
        raise ValueError("Invalid shape. Only square or rectangle allowed.")
    
    num_sensors = int(input("Enter the number of sensors: "))
    sensor_range = float(input("Enter the range of each sensor: "))
    
    return width, height, num_sensors, sensor_range

def is_within_range(sensor, point, r):
    dist = sqrt((sensor[0] - point[0])**2 + (sensor[1] - point[1])**2)
    return dist <= r

def optimize_sensor_placement(width, height, num_sensors, sensor_range):
    grid_points = [(x, y) for x in range(width) for y in range(height)]
    possible_sensor_locations = grid_points.copy()

    # Create the problem
    prob = LpProblem("Sensor_Coverage", LpMaximize)

    # Decision variables
    sensor_vars = {
        loc: LpVariable(f"sensor_{loc[0]}_{loc[1]}", cat=LpBinary)
        for loc in possible_sensor_locations
    }
    point_covered = {
        pt: LpVariable(f"covered_{pt[0]}_{pt[1]}", cat=LpBinary)
        for pt in grid_points
    }

    # Objective: Maximize number of points covered
    prob += lpSum(point_covered[pt] for pt in grid_points)

    # Constraints:
    # Each point is covered by at least one sensor within range
    for pt in grid_points:
        covering_sensors = [sensor_vars[loc] for loc in possible_sensor_locations if is_within_range(loc, pt, sensor_range)]
        if covering_sensors:
            prob += point_covered[pt] <= lpSum(covering_sensors)

    # Limit number of sensors
    prob += lpSum(sensor_vars[loc] for loc in possible_sensor_locations) == num_sensors

    prob.solve()

    selected_sensors = [loc for loc in possible_sensor_locations if sensor_vars[loc].varValue == 1.0]
    return selected_sensors, width, height, sensor_range

def plot_result(sensors, width, height, sensor_range):
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title("Sensor Placement Optimization")

    for sensor in sensors:
        circle = plt.Circle(sensor, sensor_range, color='blue', alpha=0.2)
        ax.add_patch(circle)
        ax.plot(sensor[0], sensor[1], 'ro')  # sensor point

    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
    width, height, num_sensors, sensor_range = get_inputs()
    sensors, width, height, sensor_range = optimize_sensor_placement(width, height, num_sensors, sensor_range)
    print("Sensors placed at:", sensors)
    plot_result(sensors, width, height, sensor_range)
