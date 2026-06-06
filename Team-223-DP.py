import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import time

# ==========================================
# 0. METADATA CONFIGURATION
# ==========================================
# TODO: Modify these variables to match the first alphabetical student group ID
STUDENT_ID_LAST_DIGIT = 3  

print(f"Execution Timestamp: {datetime.datetime.now()}")
print(f"Virtual Machine ID: {os.uname().nodename if hasattr(os, 'uname') else 'VM-DP-Node'}")
print(f"Configuring environment profile matching student digit suffix: {STUDENT_ID_LAST_DIGIT}\n")

# Assign environment rules mapped from student ID parameters
if STUDENT_ID_LAST_DIGIT in [0, 1, 2, 3, 4]:
    GRID_SIZE = 5
    NUM_RESCUE = 2
    NUM_CHARGING = 1
    NUM_DANGER = 3
    NUM_BLOCKED = 2
    WIND_PROB = 0.20
else:
    GRID_SIZE = 6
    NUM_RESCUE = 3
    NUM_CHARGING = 2
    NUM_DANGER = 4
    NUM_BLOCKED = 3
    WIND_PROB = 0.30

MAX_BATTERY = 10 if (STUDENT_ID_LAST_DIGIT % 2 == 0) else 15
MAX_STEPS = 50 if GRID_SIZE == 5 else 75

# Actions layout mapping
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'HOVER']
ACTION_VECTORS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
    'HOVER': (0, 0)
}

# ==========================================
# 1. FIXED CUSTOM GRID LAYOUT SETUP (Task 1)
# ==========================================
# Define grid positions manually to guarantee static layouts
GRID_LAYOUT = np.full((GRID_SIZE, GRID_SIZE), 'F', dtype=object)
GRID_LAYOUT[0, 0] = 'S' # Top-left start location

if GRID_SIZE == 5:
    GRID_LAYOUT[1, 2] = 'R'; GRID_LAYOUT[3, 4] = 'R'  # Rescue Targets
    GRID_LAYOUT[2, 2] = 'C'                            # Charging Station
    GRID_LAYOUT[1, 1] = 'D'; GRID_LAYOUT[3, 2] = 'D'; GRID_LAYOUT[4, 3] = 'D' # Danger Zones
    GRID_LAYOUT[0, 3] = 'X'; GRID_LAYOUT[2, 4] = 'X'  # Blocked Debris
    GRID_LAYOUT[2, 1] = 'W'; GRID_LAYOUT[1, 3] = 'W'  # Wind Cells
else:
    GRID_LAYOUT[1, 2] = 'R'; GRID_LAYOUT[3, 4] = 'R'; GRID_LAYOUT[5, 2] = 'R'
    GRID_LAYOUT[2, 2] = 'C'; GRID_LAYOUT[4, 1] = 'C'
    GRID_LAYOUT[1, 1] = 'D'; GRID_LAYOUT[3, 2] = 'D'; GRID_LAYOUT[4, 4] = 'D'; GRID_LAYOUT[5, 5] = 'D'
    GRID_LAYOUT[0, 4] = 'X'; GRID_LAYOUT[2, 5] = 'X'; GRID_LAYOUT[3, 0] = 'X'
    GRID_LAYOUT[2, 1] = 'W'; GRID_LAYOUT[4, 3] = 'W'

print("Configured Structural Grid Layout:")
print(GRID_LAYOUT)

# ==========================================
# 2. STATE SPACE ENUMERATION
# ==========================================
# Generate state identifiers. State representation = ((r, c), battery, (target_0_status, target_1_status))
states = []
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if GRID_LAYOUT[r, c] == 'X':
            continue # Ignore impossible coordinates blocked by obstacles
        for b in range(MAX_BATTERY + 1):
            # Binary state tuples representing targets remaining (1) or cleared (0)
            if NUM_RESCUE == 2:
                for t1 in [0, 1]:
                    for t2 in [0, 1]:
                        states.append(((r, c), b, (t1, t2)))
            else:
                for t1 in [0, 1]:
                    for t2 in [0, 1]:
                        for t3 in [0, 1]:
                            states.append(((r, c), b, (t1, t2, t3)))

state_to_idx = {s: i for i, s in enumerate(states)}
num_states = len(states)
print(f"\nTotal Valid Reachable System States Enumerated: {num_states}")

# ==========================================
# 3. TRANSITION DYNAMICS & REWARD (Task 1)
# ==========================================
def get_transitions_and_rewards(state, action):
    """
    Computes all potential outcome state-destination paths, 
    probabilities, and associated immediate reward weights.
    """
    (r, c), b, targets = state
    
    # Terminal triggers check
    if b == 0 or sum(targets) == 0:
        return [(state, 1.0, 0)]
        
    # Process movement action modifications
    intended_moves = [action]
    probabilities = [1.0]
    
    # Check if the drone is currently standing on a wind disturbance grid cell
    if GRID_LAYOUT[r, c] == 'W' and action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        intended_moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        probabilities = [WIND_PROB / 4.0] * 4
        # Add tracking for the remaining intentional deterministic split balance
        intended_moves.append(action)
        probabilities.append(1.0 - WIND_PROB)
        
    outcomes = {}
    
    for move, prob in zip(intended_moves, probabilities):
        if prob == 0.0:
            continue
            
        dr, dc = ACTION_VECTORS[move]
        new_r, new_c = r + dr, c + dc
        
        # Grid boundaries and blocked cells check
        if new_r < 0 or new_r >= GRID_SIZE or new_c < 0 or new_c >= GRID_SIZE or GRID_LAYOUT[new_r, new_c] == 'X':
            new_r, new_c = r, c # Forced to bounce back to previous coordinate
            
        # Battery consumption tracking
        new_b = b - 1
        
        # Handle custom station recharge hovering rule
        if move == 'HOVER' and GRID_LAYOUT[r, c] == 'C':
            new_b = min(MAX_BATTERY, b + 2) # Recharging active
            
        if new_b < 0:
            new_b = 0
            
        # Direct instant full charge replenishment rule when arriving on station cells
        if GRID_LAYOUT[new_r, new_c] == 'C' and move != 'HOVER':
            new_b = MAX_BATTERY
            
        # Target status alteration processing
        new_targets = list(targets)
        reward = -1 # Standard step base penalty cost
        
        # Map target items to determine active pickup indexes
        target_positions = [(1, 2), (3, 4)] if GRID_SIZE == 5 else [(1, 2), (3, 4), (5, 2)]
        
        if (new_r, new_c) in target_positions:
            t_idx = target_positions.index((new_r, new_c))
            if new_targets[t_idx] == 1: # Validation check ensuring target is active
                new_targets[t_idx] = 0 # Mark collected
                reward += 20          # Apply rescue reward bonus
                
        # Handle specialized hazard / exhaustion penalties
        if GRID_LAYOUT[new_r, new_c] == 'D':
            reward += -10
        if new_b == 0 and sum(new_targets) > 0:
            reward += -20
            
        next_state = ((new_r, new_c), new_b, tuple(new_targets))
        outcomes[next_state] = outcomes.get(next_state, 0.0) + prob
        
    return [(ns, p, reward) for ns, p in outcomes.items()]

# ==========================================
# 4. DYNAMIC PROGRAMMING ENGINE (Task 2)
# ==========================================
V = np.zeros(num_states)
policy = np.zeros(num_states, dtype=int)
theta = 1e-3
gamma = 0.95

start_time = time.time()
iterations = 0

print("\nStarting Value Iteration Loop...")
while True:
    delta = 0
    iterations += 1
    for s_idx, state in enumerate(states):
        (r, c), b, targets = state
        if b == 0 or sum(targets) == 0:
            continue # Skip completed terminal loop execution steps
            
        q_values = []
        for a_idx, action in enumerate(ACTIONS):
            q_val = 0
            transitions = get_transitions_and_rewards(state, action)
            for next_state, prob, reward in transitions:
                ns_idx = state_to_idx[next_state]
                q_val += prob * (reward + gamma * V[ns_idx])
            q_values.append(q_val)
            
        best_value = max(q_values)
        delta = max(delta, abs(best_value - V[s_idx]))
        V[s_idx] = best_value
        policy[s_idx] = np.argmax(q_values)
        
    if delta < theta:
        break

runtime = time.time() - start_time
print(f"Convergence met in: {iterations} Iterations")
print(f"Total Computation Runtime Profile: {runtime:.4f} seconds")
print(f"Final Convergence Delta Residual: {delta:.6f}")

# ==========================================
# 5. STATE-VALUE ANALYSIS HEATMAP (Task 4)
# ==========================================
# Fix standard parameters slice: Fully intact battery capacity and all active targets remaining
fixed_battery = MAX_BATTERY
fixed_targets = tuple([1] * NUM_RESCUE)

value_grid = np.zeros((GRID_SIZE, GRID_SIZE))
policy_grid = np.full((GRID_SIZE, GRID_SIZE), ' ', dtype=object)

action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '⟳'}

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if GRID_LAYOUT[r, c] == 'X':
            value_grid[r, c] = np.nan # Void out obstacle coordinates
            continue
        state_key = ((r, c), fixed_battery, fixed_targets)
        if state_key in state_to_idx:
            s_idx = state_to_idx[state_key]
            value_grid[r, c] = V[s_idx]
            policy_grid[r, c] = action_symbols[policy[s_idx]]

plt.figure(figsize=(8, 6))
sns.heatmap(value_grid, annot=policy_grid, fmt='', cmap='viridis', cbar=True, square=True)
plt.title(f'State-Value V*(s) Heatmap Slice (Battery={fixed_battery}, Active Targets={fixed_targets})')
plt.xlabel('Grid Column Layout Coordinates')
plt.ylabel('Grid Row Layout Coordinates')
plt.show()