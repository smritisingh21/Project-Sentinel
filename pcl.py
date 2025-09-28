import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ortools.sat.python import cp_model

def get_user_simple_input():
    """Gathers essential parameters from the user interactively for sensor placement."""
    params = {}
    print("--- Simplified Sensor Placement Optimizer ---")
    print("Let's define your area, targets, and sensors.\n")

    # 1. Area Dimensions
    print("--- 1. Define the Rectangular Area ---")
    params['width'] = float(input("Enter the width of your area (e.g., 50.0): "))
    params['height'] = float(input("Enter the height of your area (e.g., 50.0): "))
    
    # Grid for potential sensor locations - fixed for simplicity
    grid_step = 1.0 # Every 1 unit is a potential sensor location
    
    # 2. Target Locations
    print("\n--- 2. Enter Important Target Coordinates ---")
    print("Enter 'x y' for each target (e.g., '10 20'). Type 'done' when finished.")
    targets = []
    while True:
        try:
            entry = input(f"Target #{len(targets) + 1} (or 'done'): ")
            if entry.lower() == 'done':
                if not targets:
                    print("You must enter at least one target.")
                    continue
                break
            x, y = map(float, entry.split())
            if 0 <= x <= params['width'] and 0 <= y <= params['height']:
                targets.append((x, y))
            else:
                print(f"Warning: Target ({x},{y}) is outside the {params['width']}x{params['height']} area. Please re-enter or ensure it's within bounds.")
        except ValueError:
            print("Invalid format. Please enter coordinates as 'x y'.")
    params['targets'] = targets

    # 3. Sensor Parameters
    print("\n--- 3. Sensor Specifications ---")
    params['max_sensors'] = int(input("Enter the total number of sensors you have available: "))
    params['sensor_range'] = float(input("Enter the coverage radius of each sensor (e.g., 5.0): "))

    # 4. Fixed Optimization Weights (for simplicity, we'll use reasonable defaults)
    # These prioritize coverage, then minimize overlap, then minimize total sensors.
    params['w_coverage'] = 1000  # High reward for covering a target
    params['w_overlap'] = 50     # Moderate penalty for each instance of overlap
    params['w_cost'] = 10        # Small penalty for using each sensor
    
    print(f"\nOptimization will prioritize covering targets, minimizing overlap, and using fewer sensors.")

    # Generate potential sensor locations based on grid step
    potential_locations = []
    # Using small epsilon to ensure max_width/height are included if grid_step is small
    for x in np.arange(0, params['width'] + grid_step/2, grid_step):
        for y in np.arange(0, params['height'] + grid_step/2, grid_step):
            potential_locations.append((x, y))
    params['potential_locations'] = potential_locations
    
    return params

def plot_optimization_results(params, placed_sensors_coords, covered_targets_indices):
    """Visualizes the optimization result using matplotlib."""
    width = params['width']
    height = params['height']
    targets = np.array(params['targets'])
    sensor_range = params['sensor_range']
    
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    ax.set_facecolor('#F0F0F0') # Light grey background

    # Plot targets
    all_targets_x = targets[:, 0]
    all_targets_y = targets[:, 1]
    
    # Mark uncovered targets red, covered targets green
    uncovered_mask = np.ones(len(targets), dtype=bool)
    if covered_targets_indices: # Ensure there are covered targets to mark
        uncovered_mask[covered_targets_indices] = False
    
    ax.scatter(all_targets_x[uncovered_mask], all_targets_y[uncovered_mask], 
               c='red', marker='x', s=100, linewidths=2, label='Uncovered Target')
    ax.scatter(all_targets_x[~uncovered_mask], all_targets_y[~uncovered_mask], 
               c='green', marker='o', s=100, edgecolors='black', linewidths=1, label='Covered Target')

    # Plot sensors and their ranges
    for (x, y) in placed_sensors_coords:
        ax.scatter(x, y, c='blue', marker='^', s=150, edgecolors='black', linewidths=1, label='Deployed Sensor' if 'Deployed Sensor' not in [c.get_label() for c in ax.collections] else "")
        coverage_circle = Circle((x, y), sensor_range, color='blue', alpha=0.15, ec='blue', lw=0.5, linestyle='--')
        ax.add_patch(coverage_circle)

    ax.set_xlim(-0.1 * width, width * 1.1) # Add a small buffer around the area
    ax.set_ylim(-0.1 * height, height * 1.1)
    ax.set_aspect('equal', adjustable='box')
    plt.title('Optimal Sensor Deployment (Minimizing Overlap & Cost)', fontsize=16)
    plt.xlabel(f'Width ({width:.0f} units)', fontsize=12)
    plt.ylabel(f'Height ({height:.0f} units)', fontsize=12)
    plt.legend(loc='lower left', bbox_to_anchor=(1.02, 0)) # Move legend outside
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust layout to make space for legend
    plt.show()

def main():
    params = get_user_simple_input()

    locations = params['potential_locations']
    targets = params['targets']
    sensor_range = params['sensor_range']
    
    num_locations = len(locations)
    num_targets = len(targets)

    # Pre-calculate which locations can cover which targets
    covers = {}
    for i in range(num_locations):
        for j in range(num_targets):
            dist = np.sqrt((locations[i][0] - targets[j][0])**2 + (locations[i][1] - targets[j][1])**2)
            if dist <= sensor_range:
                covers[(i, j)] = 1
            else:
                covers[(i, j)] = 0
                
    # --- ILP Model using Google OR-Tools CP-SAT Solver ---
    model = cp_model.CpModel()

    # --- Decision Variables ---
    x = [model.NewBoolVar(f'x_{i}') for i in range(num_locations)]
    y = [model.NewBoolVar(f'y_{j}') for j in range(num_targets)]
    num_covers = [model.NewIntVar(0, params['max_sensors'], f'num_covers_{j}') for j in range(num_targets)]

    # --- Constraints ---
    for j in range(num_targets):
        sensors_covering_target_j = [x[i] for i in range(num_locations) if covers.get((i, j), 0) == 1]
        model.Add(cp_model.LinearExpr.Sum(sensors_covering_target_j) == num_covers[j])

    for j in range(num_targets):
        model.Add(num_covers[j] > 0).OnlyEnforceIf(y[j])
        model.Add(num_covers[j] == 0).OnlyEnforceIf(y[j].Not())

    model.Add(cp_model.LinearExpr.Sum(x) <= params['max_sensors'])

    # --- Objective Function ---
    total_objective_expression = []
    for j in range(num_targets):
        total_objective_expression.append(y[j] * params['w_coverage'])

    for i in range(num_locations):
        total_objective_expression.append(x[i] * -params['w_cost'])

    for j in range(num_targets):
        # *** THIS IS THE CORRECTED SECTION ***
        # Create a variable for the overlap count (num_covers - 1)
        # It's only calculated if the target is covered (y[j] is true)
        actual_overlap_count = model.NewIntVar(0, params['max_sensors'], f'actual_overlap_{j}')
        
        # If target j is covered, then overlap = num_covers[j] - 1
        model.Add(actual_overlap_count == num_covers[j] - 1).OnlyEnforceIf(y[j])
        
        # If target j is NOT covered, then overlap must be 0
        model.Add(actual_overlap_count == 0).OnlyEnforceIf(y[j].Not())

        total_objective_expression.append(actual_overlap_count * -params['w_overlap'])

    model.Maximize(cp_model.LinearExpr.Sum(total_objective_expression))

    # --- Solve the Model ---
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True 
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    # --- Process and Display Results ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n--- Optimization Complete ---")
        placed_sensors_coords = [locations[i] for i in range(num_locations) if solver.Value(x[i]) == 1]
        
        covered_targets_indices = [j for j in range(num_targets) if solver.Value(y[j]) == 1]
        actual_num_covered_targets = len(covered_targets_indices)
        actual_total_overlap = sum(max(0, solver.Value(num_covers[j]) - 1) for j in range(num_targets))

        print(f"Optimal / Feasible solution found!")
        print(f"  - Sensors deployed: {len(placed_sensors_coords)} / {params['max_sensors']} available")
        print(f"  - Unique targets covered: {actual_num_covered_targets} / {num_targets}")
        print(f"  - Total redundant coverage instances: {actual_total_overlap}")
        
        print("\nFinal Optimal Sensor Deployment Coordinates:")
        for loc in placed_sensors_coords:
            print(f"  - ({loc[0]:.2f}, {loc[1]:.2f})")
            
        plot_optimization_results(params, placed_sensors_coords, covered_targets_indices)
    else:
        print("\n--- Optimization Result ---")
        print(f"Solver Status: {solver.StatusName(status)}")
        print("No optimal or feasible solution found within the given constraints/time limit.")

if __name__ == '__main__':
    main()