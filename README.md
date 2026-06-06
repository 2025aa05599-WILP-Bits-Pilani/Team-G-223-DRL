Here is a comprehensive, production-ready **`README.md`** file for your GitHub repository. It perfectly outlines both assignments, references Group 223's dynamic parameter profiles, maps out the repository organization, and provides a clean installation guide.

---

```markdown
# Deep Reinforcement Learning (DRL) — Lab Assignment 1

This repository contains the complete implementation, operational benchmarks, and analytical reports for BITS Pilani WILP **Deep Reinforcement Learning (Lab Assignment 1)**. The project covers two key paradigms in Reinforcement Learning: addressing the exploration vs. exploitation trade-off via **Multi-Armed Bandits (MAB)**, and executing full-horizon sequential optimization using **Dynamic Programming (DP)**.

All simulations and computations were fully executed within the institutional Virtual Lab infrastructure.

---

## 👥 Group Profile & Parameter Assignments
* **Group ID ($G$):** `223` (Suffix Digit: `3`)
* **Part 1 (MAB) Configuration:** * Total Available Medicines ($K$): **6**
  * Seed Configuration: Reproducibility strictly bound via `random.seed(223)` and `numpy.random.seed(223)`.
* **Part 2 (DP) Configuration:** * Grid Size: **5 × 5** spatial grid layout
  * Target States: **2 Rescue Targets**, **1 Charging Station**, **3 Danger Zones**, **2 Blocked Obstacles**
  * Maximum Allowed Drone Battery: **15 units** (assigned due to odd group suffix)
  * Wind Stochasticity Vector: **20%** transition diversion constraint

---

## 📂 Repository Structure

```text
├── Team 223 - MAB.ipynb      # Interactive Jupyter Notebook for the Multi-Armed Bandit simulation
├── Team 223 - DP.ipynb       # Interactive Jupyter Notebook for the Drone Rescue MDP solver
├── Team 223 - MAB.pdf        # Evaluator PDF export containing MAB cell logs, outputs, and analysis
├── Team 223 - DP.pdf         # Evaluator PDF export containing DP state valuations and policy maps
├── Part1_MAP_SS.jpg          # Verified execution screenshot with system timestamp for Part 1
└── Part2_DP_SS.jpg           # Verified execution screenshot with system timestamp for Part 2

```

---

## 🧠 Core System Overviews

### Part 1: Adaptive Clinical Treatment Optimization (MAB)

This module simulates a sequential clinical framework administering treatments across 1,000 synthetic patient profiles. The objective is to maximize patient recovery outcomes while minimizing treatment risk by calculating utility values weighted against baseline disease severity scores.

* **Implemented Frameworks:** Immediate Exploitation (Pure Greedy with a 10-pull initial evaluation baseline), $\epsilon$-Greedy Policies ($\epsilon \in \{0.01, 0.10, 0.50\}$), and the **Upper Confidence Bound (UCB1)** strategy.
* **Key Observations:** Analysis highlights how tracking variance borders via the UCB1 algorithm guarantees stable, high-yield long-term rewards, outperforming heuristic exploration methods in safe clinical trial deployment.

### Part 2: Autonomous Search & Rescue Drone Navigation (DP)

This section frames an autonomous search-and-rescue drone flight route planner as a finite **Markov Decision Process (MDP)**. Operating under constrained battery thresholds, the drone must determine optimal trajectories to collect stranded victims while avoiding localized radar/danger hazards and handling stochastic wind disruptions.

* **Mathematical Resolution Engine:** A tabular **Value Iteration** loop solved to a strict terminal convergence stopping threshold of $\theta = 10^{-3}$.
* **Visual Annotations:** Generates specialized Seaborn state-value maps overlaid with optimal policy direction arrows ($\uparrow, \downarrow, \leftarrow, \rightarrow, $) showing deliberate charging actions, obstacle clearance, and target tracking behaviors.
* **Theoretical Deliverable:** Details the impact of the *Curse of Dimensionality* on state-space scale configurations ($S = (R \times C) \times (B + 1) \times 2^N$), proving the structural limitations of tabular methods and outlining why Deep RL (DQN/PPO) serves as a necessary alternative for real-world continuous control spaces.

---

## 🚀 Installation & Local Execution

### 1. Prerequisites

Ensure you have an active Python environment (Python 3.10+ recommended) equipped with standard scientific computing and visualization packages:

```bash
pip install numpy pandas matplotlib seaborn jupyterlab

```

### 2. Fetching the Workspace

Clone the repository locally:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 3. Launching the Interactive Solutions

To run the evaluation notebooks or inspect the code blocks directly:

```bash
jupyter lab

```

Open either `Team 223 - MAB.ipynb` or `Team 223 - DP.ipynb` and select **Run All Cells** from the top interactive workspace dropdown menu.

---

## 📜 Academic Integrity Note

This project was designed and compiled in compliance with the academic assessment rules of BITS Pilani. Notebook components contain explicit platform metadata outputs identifying specific virtual machine host keys alongside execution clocks to guarantee authentic individual reproduction.

```
***

### 💡 Implementation Tip
When publishing this repository to GitHub, go to your main repository landing page, look for the **"About"** gear icon on the right-hand side, and set the summary text box to:

> *Tabular Multi-Armed Bandit (MAB) clinical trial models and Markov Decision Process (MDP) drone rescue optimizations resolved via Value Iteration. Developed for BITS Pilani DRL Lab Assignment (Group 223).*

```