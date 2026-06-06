import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import os

# ==========================================
# 0. VIRTUAL LAB METADATA & CONFIGURATION
# ==========================================
# TODO: Replace with your actual Group Number
G = 1  

print(f"Execution Timestamp: {datetime.datetime.now()}")
print(f"Virtual Machine ID: {os.uname().nodename if hasattr(os, 'uname') else 'VM-Default-ID'}")
print(f"Group Number Assigned: {G}\n")

# Set random seeds for strict reproducibility
random.seed(G)
np.random.seed(G)

# Calculate problem parameters based on Group Number
K = (G % 3) + 5
hidden_probabilities = [0.4 + ((G + i) % 6) * 0.07 for i in range(K)]

print(f"Total Medicines (K): {K}")
for i, prob in enumerate(hidden_probabilities):
    print(f"  Medicine {i} Hidden Success Probability: {prob:.4f}")

# ==========================================
# 1. ENVIRONMENT & DATASET DESIGN (Task 1)
# ==========================================
def generate_base_environment(num_patients=1000):
    """
    Generates the static patient registry containing ID and disease severity.
    Severity scales from 1 (mild) to 5 (critical) using a deterministic sequence.
    """
    dataset = []
    for pid in range(num_patients):
        severity = (pid % 5) + 1
        dataset.append({
            'patient_id': pid,
            'severity_score': severity
        })
    return pd.DataFrame(dataset)

def simulate_treatment(medicine_idx, severity, hidden_probs):
    """
    Simulates the stochastic recovery outcome and computes the utility reward.
    """
    p_success = hidden_probs[medicine_idx]
    clinical_outcome = 1 if random.random() < p_success else 0
    utility_score = clinical_outcome * (1.0 - (severity / 10.0))
    return clinical_outcome, utility_score

# Display first 10 rows of the base environment
base_df = generate_base_environment()
print("\nFirst 10 Rows of Base Environment Dataset:")
print(base_df.head(10).to_string(index=False))

# ==========================================
# 2. STRATEGY 1: IMMEDIATE EXPLOITATION (Task 2)
# ==========================================
def run_immediate_exploitation(df, hidden_probs, K):
    """
    Tests each arm exactly 10 times initially, then pulls exclusively 
    the medicine with the highest empirical success rate.
    """
    df_exec = df.copy()
    assigned_medicines, outcomes, utilities = [], [], []
    
    successes = np.zeros(K)
    trials = np.zeros(K)
    
    for idx, row in df_exec.iterrows():
        sev = row['severity_score']
        
        # Initial phase: Test each medicine 10 times
        if idx < 10 * K:
            med = idx % K
        else:
            # Exploitation phase: lock into the best-performing medicine
            # Divide success by trials to find highest recovery rate
            empirical_rates = np.where(trials > 0, successes / trials, 0)
            med = np.argmax(empirical_rates)
            
        out, util = simulate_treatment(med, sev, hidden_probs)
        
        # Update empirical statistics based on clinical outcome
        successes[med] += out
        trials[med] += 1
        
        assigned_medicines.append(med)
        outcomes.append(out)
        utilities.append(util)
        
    df_exec['assigned_medicine'] = assigned_medicines
    df_exec['clinical_outcome'] = outcomes
    df_exec['utility_score'] = utilities
    return df_exec

# ==========================================
# 3. STRATEGY 2: EPSILON-GREEDY STRATEGIES (Task 3)
# ==========================================
def run_epsilon_greedy(df, hidden_probs, K, epsilon):
    """
    Explores alternate treatments with a probability of epsilon, 
    otherwise exploits the currently known best arm.
    """
    df_exec = df.copy()
    assigned_medicines, outcomes, utilities = [], [], []
    
    successes = np.zeros(K)
    trials = np.zeros(K)
    
    for idx, row in df_exec.iterrows():
        sev = row['severity_score']
        
        if random.random() < epsilon:
            med = random.randint(0, K - 1)  # Explore random arm
        else:
            empirical_rates = np.where(trials > 0, successes / trials, 0)
            med = np.argmax(empirical_rates) # Exploit best arm
            
        out, util = simulate_treatment(med, sev, hidden_probs)
        
        successes[med] += out
        trials[med] += 1
        
        assigned_medicines.append(med)
        outcomes.append(out)
        utilities.append(util)
        
    df_exec['assigned_medicine'] = assigned_medicines
    df_exec['clinical_outcome'] = outcomes
    df_exec['utility_score'] = utilities
    return df_exec

# ==========================================
# 4. STRATEGY 3: UCB1 STRATEGY (Task 4)
# ==========================================
def run_ucb1(df, hidden_probs, K):
    """
    Upper Confidence Bound algorithm. Balances exploration and exploitation
    by assessing variance/uncertainty limits for under-sampled arms.
    """
    df_exec = df.copy()
    assigned_medicines, outcomes, utilities = [], [], []
    
    successes = np.zeros(K)
    trials = np.zeros(K)
    
    for idx, row in df_exec.iterrows():
        sev = row['severity_score']
        t = idx + 1 # Current total global iterations
        
        # Guarantee every single arm is pulled at least once to initialize
        if t <= K:
            med = idx % K
        else:
            ucb_values = np.zeros(K)
            for i in range(K):
                empirical_rate = successes[i] / trials[i]
                # Calculate uncertainty boundary margin
                exploration_bonus = np.sqrt((2 * np.log(t)) / trials[i])
                ucb_values[i] = empirical_rate + exploration_bonus
            med = np.argmax(ucb_values)
            
        out, util = simulate_treatment(med, sev, hidden_probs)
        
        successes[med] += out
        trials[med] += 1
        
        assigned_medicines.append(med)
        outcomes.append(out)
        utilities.append(util)
        
    df_exec['assigned_medicine'] = assigned_medicines
    df_exec['clinical_outcome'] = outcomes
    df_exec['utility_score'] = utilities
    return df_exec

# ==========================================
# 5. EXECUTION & VISUAL COMPARISON (Task 5)
# ==========================================
print("\nRunning simulations over 1000 patients...")
df_imm = run_immediate_exploitation(base_df, hidden_probabilities, K)
df_eps_10 = run_epsilon_greedy(base_df, hidden_probabilities, K, epsilon=0.10)
df_eps_01 = run_epsilon_greedy(base_df, hidden_probabilities, K, epsilon=0.01)
df_eps_50 = run_epsilon_greedy(base_df, hidden_probabilities, K, epsilon=0.50)
df_ucb = run_ucb1(base_df, hidden_probabilities, K)

# Plot performance curves
plt.figure(figsize=(12, 7))
plt.plot(df_imm['utility_score'].cumsum(), label='Immediate Exploitation (10x initialization)')
plt.plot(df_eps_10['utility_score'].cumsum(), label='Epsilon-Greedy (Epsilon = 10%)')
plt.plot(df_eps_01['utility_score'].cumsum(), label='Epsilon-Greedy (Epsilon = 1%)')
plt.plot(df_eps_50['utility_score'].cumsum(), label='Epsilon-Greedy (Epsilon = 50%)')
plt.plot(df_ucb['utility_score'].cumsum(), label='UCB1 Strategy', linewidth=2, linestyle='--')

plt.title('MAB Strategies Comparative Evaluation: Cumulative Reward over Time')
plt.xlabel('Number of Patients Trailed')
plt.ylabel('Cumulative Reward (Utility Score)')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

# Print metrics summaries
print(f"Final Cumulative Rewards achieved:")
print(f"  Immediate Exploitation:        {df_imm['utility_score'].sum():.2f}")
print(f"  Epsilon-Greedy (10% Expl.):    {df_eps_10['utility_score'].sum():.2f}")
print(f"  Epsilon-Greedy (1% Expl.):     {df_eps_01['utility_score'].sum():.2f}")
print(f"  Epsilon-Greedy (50% Expl.):    {df_eps_50['utility_score'].sum():.2f}")
print(f"  UCB1 Algorithm:                {df_ucb['utility_score'].sum():.2f}")