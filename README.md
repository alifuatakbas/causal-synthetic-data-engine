# CausalTwin: A High-Fidelity Causal Synthetic Data Engine

**CausalTwin** is a professional-grade synthetic data generation engine designed to model complex real-world causal relationships. Unlike traditional generators that merely replicate correlations, CausalTwin leverages **Directed Acyclic Graphs (DAGs)** to simulate the underlying "genetic" structure of data, creating statistically consistent digital twins for privacy-preserving AI and research.



---

## 🚀 Key Features

* **Hybrid Node Architecture**: Seamlessly integrates Numeric (Linear) and Categorical (Discrete Mapping) data within the same causal chain.
* **Graph Safety (Cycle Detection)**: Implements a three-state Depth First Search (DFS) algorithm to automatically detect and prevent infinite loops in the causal structure.
* **Topological Execution**: Ensures all parent dependencies are calculated before child nodes, maintaining logical data integrity.
* **Precision Constraints**: Built-in support for physical boundaries (min/max clipping) and data precision (rounding) to ensure realistic output.
* **Vectorized Performance**: Optimized with NumPy for high-speed simulation, capable of generating millions of records in seconds.

---

## 🧠 Mathematical Foundation

CausalTwin is built upon a hybrid causal formula that allows categorical variables to act as multipliers on numeric outcomes:

$$Y = (\beta_{intercept} + \sum_{i=1}^{n} \beta_{i} X_{numeric,i}) \times \prod_{j=1}^{m} M_{categorical,j} + \epsilon$$

Where:
* $\beta$: Represents the linear slope for numeric parents.
* $M$: Represents the discrete mapping multiplier for categorical parents.
* $\epsilon$: Represents Gaussian noise to simulate real-world uncertainty.

---

## 🛠 Installation

```bash
git clone [https://github.com/YourUsername/CausalTwin.git](https://github.com/YourUsername/CausalTwin.git)
cd CausalTwin
pip install -r requirements.txt


# Quick Start
from engine import CausalGraph, CausalNode

# Initialize the engine
engine = CausalGraph()

# Define nodes with constraints
age = CausalNode("age", params={"mean": 35, "std": 10}, constraints={"min": 18, "max": 65})
city = CausalNode("city", distribution="categorical", params={"choices": ["London", "New York"], "probs": [0.6, 0.4]})

# Define dependent relationships
income = CausalNode("income", parents=[age, city], params={
    "intercept": 3000,
    "slope_age": 100,
    "map_city": {"London": 1.2, "New York": 1.5}
})

# Add nodes and generate data
engine.add_node(age)
engine.add_node(city)
engine.add_node(income)

df = engine.generate(n_samples=5000)
print(df.head())
```

## 📈 Development & Vision

This project is a high-performance implementation of causal inference algorithms, designed by a **Computer Engineering student** with a background in **competitive mathematics**. The goal is to move beyond black-box synthetic data generation by providing a transparent, mathematically grounded engine.

### Roadmap

- [x] **Core Bayesian Engine**: Topological sorting and dependency management.
- [x] **Hybrid Node Support**: Seamless integration of numeric and categorical distributions.
- [x] **Graph Integrity**: DFS-based cycle detection for DAG safety.
- [x] **Data Constraints**: Real-world boundary clipping and rounding.
- [ ] **Non-Linear Dynamics**: Implementation of logarithmic, exponential, and polynomial causal effects.
- [ ] **Sensitivity Analysis**: Automated tools to measure the impact of parent node perturbations.
- [ ] **API Layer**: Developing a lightweight REST interface for external integration.

