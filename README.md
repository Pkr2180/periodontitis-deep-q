
 Deep Q-Learning for Periodontitis Treatment Optimization


This repository contains an AI-powered Deep Q-Learning (DQN) framework tailored for optimizing clinical treatment strategies in periodontitis management using real-world dental patient data.

 Objective

To use reinforcement learning to recommend personalized periodontal treatments such as:
-  Maintenance
-  Scaling & Root Planing (SRP)
-  Surgery
-  Tooth Extraction

 Tech Stack

- Deep Q-Networks (DQN)
- Gym Environment + RLlib
- PyTorch + NumPy
- Real Patient Dataset
- Experience Replay & ε-Greedy Strategy

---

##  Evaluation Metrics

| Metric                             | Value         |
|------------------------------------|---------------|
| Cumulative Reward (Avg over runs)  |  872.4       |
| Average Reward per Patient         |  2.53        |
| % Teeth Saved                      |  89.2%       |
| Optimal Treatment Match Rate       |  83.6%       |
| Convergence Epoch                  |  ~275        |

---

## Project Structure

```
periodontitis-dqn-ai/
├── data/                         # Clinical dataset (real anonymized patient data)
│   └── MARIA_PATIENTS_DATA.xlsx
├── notebooks/
│   └── periodontitis_dqn.ipynb  # RL workflow and metrics plots
├── results/
│   └── metrics_plot.png         # Rewards, Q-values, and convergence graphs
├── README.md                    # You're here
├── requirements.txt            # Python dependencies
```

---

##  Quick Start

```bash
git clone https://github.com/Pkr2180/deep-q-learning-periodontitis
cd deep-q-learning-periodontitis
pip install -r requirements.txt
jupyter notebook notebooks/periodontitis_dqn.ipynb
```



 Key Features

-  Custom Gym environment for patient treatment episodes
-  Smart action selection using ε-greedy policy with decay
-  Experience Replay and Target Network updates
-  Clinical outcome-driven reward shaping

---

Visualizations

![Sample Evaluation Plot](results/metrics_plot.png)



 Author

Dr.Y.Pradeep kumar  
AI Researcher | Periodontist | Bioinformatician  
GitHub: [Pkr2180](https://github.com/Pkr2180)

Star this repo if you found it useful!
