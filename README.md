# CleanSurvival: An Automated Preprocessing Framework for Survival Analysis

This repository contains the implementation of **CleanSurvival**, a reinforcement learning-based approach for optimizing data preprocessing pipelines in time-to-event modelling. 

Predictive performance in survival analysis is highly sensitive to preprocessing choices (imputation, outlier handling, and feature scaling). CleanSurvival automates these decisions using a Q-learning agent to identify optimal pipelines for specific survival models.

---

### Methodology Overview
CleanSurvival treats the construction of a preprocessing pipeline as a Markov Decision Process (MDP). The framework utilizes a Q-learning agent to navigate a search space of various preprocessing techniques, aiming to maximize a reward function based on model discriminative power (e.g., Harrell’s C-index).

#### System Architecture

The framework supports:
* **Models**: Cox Proportional Hazards, Random Survival Forests, and DeepHit.
* **Preprocessing Actions**: Multiple imputation strategies, variance-based filtering, and outlier detection methods.
* **Evaluation**: Inverse Probability of Censoring Weighting (IPCW) and Integrated Graf Score for robust assessment under censoring.

---

Key observations from the study:
* **Efficiency**: The Q-learning agent identifies high-performing pipelines significantly more efficiently than exhaustive or random search strategies.
* **Robustness**: The framework maintains stability across different missingness mechanisms (MCAR, MAR, MNAR).


---
## **Preprint & Citation**

The methodology and detailed results are available in our preprint:

**CleanSurvival: Automated data preprocessing for time-to-event models using reinforcement learning**  
*Yousef Koka, David Selby, Gerrit Großmann, Sebastian Vollmer*  
[![arXiv](https://img.shields.io/badge/arXiv-2502.03946-b31b1b.svg)](https://arxiv.org/abs/2502.03946)

### Citation
```bibtex
@misc{koka2025cleansurvival,
      title={CleanSurvival: Automated data preprocessing for time-to-event models using reinforcement learning}, 
      author={Yousef Koka and David Selby and Gerrit Großmann and Sebastian Vollmer},
      year={2025},
      eprint={2502.03946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
