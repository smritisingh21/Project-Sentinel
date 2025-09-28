import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ortools.sat.python import cp_model

def get_user_input_with_areas():
    """Gathers parameters from the user, focusing on priority areas."""
    params = {}
    print("--- Area-Based Sensor Optimizer ---")
    print("This version focuses on covering priority ZONES, not just points.\n")

    # 1. Area Dimensions
    print("--- 1. Define the Rectangular Area ---")
    params['width'] = float(input("Enter the width of your area (e.g., 50.0): "))
    params['height'] = float(input("Enter the height of your area (e.g., 50.0): "))
    
    # 2. Priority Areas
    print("\n--- 2. Enter Priority Areas ---")
    print("Enter 'x y radius' for each circular priority area (e.g., '10 20 5').")
    print("Type 'done' when finished.")
    priority_areas = []
    while True:
        try:
            entry = input(f"Priority Area #{len(priority_areas) + 1} (or 'done'): ")
            if entry.lower() == 'done':
                if not priority_areas:
                    print("You must enter at least one priority area.")
                    continue
                break
            x, y, r = map(float, entry.split())
            if r <= 0:
                print("Error: Radius must be a positive number.")
                continue
            priority_areas.append({'center': (x, y), 'radius': r})
        except ValueError:
            print("Invalid format. Please enter as 'x y radius'.")
    params['priority_areas'] = priority_areas

    # 3. Sampling Density
    print("\n--- 3. Define Coverage Quality ---")
    params['density'] = float(input("Enter sampling density for areas (e.g., 1.0 is high, 2.0 is lower): "))

    # 4. Sensor Parameters
    print("\n--- 4. Sensor Specifications ---")
    params['max_sensors'] = int(input("Enter the total number of sensors you have available: "))
    params['sensor_range'] = float(input("Enter the coverage radius of each sensor (e.g., 5.0): "))

    # 5. Fixed Optimization Weights
    params['w_coverage'] = 1000
    params['w_overlap'] = 50
    params['w_cost'] = 10
    
    return params

def generate_sub_targets(priority_areas, density):
    """Generates a set of discrete points within the given priority areas."""
    sub_targets = set()
    for area in priority_areas:
        center_x, center_y = area['center']
        radius = area['radius']
        
        # Create a grid of points around the circle's bounding box
        for x in np.arange(center_x - radius, center_x + radius + density, density):
            for y in np.arange(center_y - radius, center_y + radius + density, density):
                # Check if the point is inside the circle
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius:
                    sub_targets.add((x, y))
    
    if not sub_targets:
         print("\nWarning: Could not generate any sub-targets. Your density might be too large for the given radii.")
         return []
         
    return list(sub_targets)

def plot_results_with_areas(params, placed_sensors_coords, all_sub_targets, covered_sub_targets):
    """Visualizes the optimization result, showing priority areas."""
    width = params['width']
    height = params['height']
    sensor_range = params['sensor_range']
    
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    ax.set_facecolor('#F0F0F0')

    # 1. Plot Priority Areas
    for area in params['priority_areas']:
        center_x, center_y = area['center']
        radius = area['radius']
        priority_zone = Circle((center_x, center_y), radius, color='red', alpha=0.1, ec='red', lw=1, linestyle='--', label='Priority Area' if 'Priority Area' not in [p.get_label() for p in ax.patches] else "")
        ax.add_patch(priority_zone)

    # 2. Plot Sub-Targets
    all_sub_targets_np = np.array(all_sub_targets)
    covered_sub_targets_set = set(covered_sub_targets)
    
    uncovered_mask = [pt not in covered_sub_targets_set for pt in all_sub_targets]
    
    ax.scatter(all_sub_targets_np[uncovered_mask, 0], all_sub_targets_np[uncovered_mask, 1], c='#ff9999', marker='.', s=10, label='Uncovered Sub-Target')
    ax.scatter(all_sub_targets_np[~np.array(uncovered_mask), 0], all_sub_targets_np[~np.array(uncovered_mask), 1], c='green', marker='.', s=10, label='Covered Sub-Target')

    # 3. Plot Deployed Sensors
    for (x, y) in placed_sensors_coords:
        ax.scatter(x, y, c='blue', marker='^', s=150, edgecolors='black', linewidths=1, zorder=10, label='Deployed Sensor' if 'Deployed Sensor' not in [c.get_label() for c in ax.collections] else "")
        coverage_circle = Circle((x, y), sensor_range, color='blue', alpha=0.15, ec='blue', lw=0.5, linestyle='--')
        ax.add_patch(coverage_circle)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    plt.title('Optimal Sensor Deployment (Area-Based)', fontsize=16)
    plt.xlabel(f'Width ({width:.0f} units)', fontsize=12)
    plt.ylabel(f'Height ({height:.0f} units)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()


def main():
    params = get_user_input_with_areas()
    
    print("\nGenerating sub-target points within priority areas...")
    sub_targets = generate_sub_targets(params['priority_areas'], params['density'])
    print(f"Generated {len(sub_targets)} unique sub-target points.")

    if not sub_targets:
        print("Cannot proceed with optimization without any targets.")
        return

    # --- The rest of the logic is the same, just using 'sub_targets' ---
    locations = []
    grid_step = 1.0 # Potential sensor locations every 1 unit
    for x in np.arange(0, params['width'] + grid_step/2, grid_step):
        for y in np.arange(0, params['height'] + grid_step/2, grid_step):
            locations.append((x,y))

    num_locations = len(locations)
    num_targets = len(sub_targets)

    # Pre-calculate which locations can cover which sub-targets
    covers = {}
    for i in range(num_locations):
        for j in range(num_targets):
            dist = np.sqrt((locations[i][0] - sub_targets[j][0])**2 + (locations[i][1] - sub_targets[j][1])**2)
            if dist <= params['sensor_range']:
                covers[(i, j)] = 1
            else:
                covers[(i, j)] = 0
                
    # ILP Model (same as before)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x_{i}') for i in range(num_locations)]
    y = [model.NewBoolVar(f'y_{j}') for j in range(num_targets)]
    num_covers = [model.NewIntVar(0, params['max_sensors'], f'num_covers_{j}') for j in range(num_targets)]

    for j in range(num_targets):
        sensors_covering_target_j = [x[i] for i in range(num_locations) if covers.get((i, j), 0) == 1]
        model.Add(cp_model.LinearExpr.Sum(sensors_covering_target_j) == num_covers[j])

    for j in range(num_targets):
        model.Add(num_covers[j] > 0).OnlyEnforceIf(y[j])
        model.Add(num_covers[j] == 0).OnlyEnforceIf(y[j].Not())

    model.Add(cp_model.LinearExpr.Sum(x) <= params['max_sensors'])

    total_objective_expression = []
    for j in range(num_targets):
        total_objective_expression.append(y[j] * params['w_coverage'])
    for i in range(num_locations):
        total_objective_expression.append(x[i] * -params['w_cost'])
    for j in range(num_targets):
        actual_overlap_count = model.NewIntVar(0, params['max_sensors'], f'actual_overlap_{j}')
        model.Add(actual_overlap_count == num_covers[j] - 1).OnlyEnforceIf(y[j])
        model.Add(actual_overlap_count == 0).OnlyEnforceIf(y[j].Not())
        total_objective_expression.append(actual_overlap_count * -params['w_overlap'])

    model.Maximize(cp_model.LinearExpr.Sum(total_objective_expression))

    # Solve and process results
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60.0 # Increased time for potentially harder problems
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n--- Optimization Complete ---")
        placed_sensors_coords = [locations[i] for i in range(num_locations) if solver.Value(x[i]) == 1]
        
        covered_sub_targets = [sub_targets[j] for j in range(num_targets) if solver.Value(y[j]) == 1]
        
        print(f"Optimal / Feasible solution found!")
        print(f"  - Sensors deployed: {len(placed_sensors_coords)} / {params['max_sensors']} available")
        print(f"  - Sub-targets covered: {len(covered_sub_targets)} / {num_targets}")

        plot_results_with_areas(params, placed_sensors_coords, sub_targets, covered_sub_targets)
    else:
        print("\n--- Optimization Result ---")
        print(f"Solver Status: {solver.StatusName(status)}")
        print("No solution found. Try increasing sensor range, number of sensors, or solver time limit.")

if __name__ == '__main__':
    main()