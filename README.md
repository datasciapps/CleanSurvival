# CleanSurvival

**CleanSurvival** is a Python-based tool that automatically finds the optimal data cleaning pipeline for survival analysis models using Q-Learning.

## Q-Learning

**Q-Learning** is a reinforcement learning algorithm used to find the optimal sequence of data cleaning methods. The tool uses a reward matrix to guide the Q-Learning agent and find the pipeline that maximizes the performance of the chosen survival analysis model.


 The tool supports three different survival analysis models:

- **Cox PH Model**
- **Random Survival Forest**
- **DeepHit Neural Network**

## Imputation Methods

1. **Complete Case Analysis (CCA)**: Removes rows with missing values.
2. **Multiple Imputation using Chained Equations (MICE)**: Uses an iterative imputer to fill in missing values.
3. **Simple Mean Imputation**: Replaces missing values with the mean of the corresponding column.
4. **Simple Median Imputation**: Replaces missing values with the median of the corresponding column.
5. **KNN Imputation**: Uses K-Nearest Neighbors to impute missing values.

## Feature Selection Methods

1. **Univariate CoxPH Selection (UC)**: Selects features based on their importance scores from a univariate Cox Proportional Hazards model.
2. **Lasso Selection**: Uses Lasso regularization to select features.
3. **Recursive Feature Elimination (RFE)**: Recursively removes features based on their importance.
4. **Information Gain Selection (IG)**: Selects features based on their information gain.

## Deduplication Methods

1. **Exact Duplicate Removal (ED)**: Removes exact duplicate rows.
2. **Deduplication by Event ID (DBID)**: Removes duplicate rows based on a unique event identifier.
3. **Deduplication by Timestamp (DBT)**: Removes duplicate rows based on timestamps.

## Outlier Detection Methods

1. **Martingale Residuals (MR)**: Uses martingale residuals to detect outliers.
2. **Multivariate Outliers (MUO)**: Detects outliers using the EllipticEnvelope method.

## Different Modes
1. **CleanSurvival**: This mode utilizes the Q-Learning algorithm to determine the optimal data cleaning pipeline for your chosen survival analysis model.
2. **Random Cleaning**: This mode generates random cleaning pipelines and evaluates their performance. You can specify the number of random experiments to run.
3. **Custom Pipeline Design**: This mode allows you to define your own custom cleaning pipelines and evaluate their performance. You need to provide a text file containing the pipelines.
4. **No Preparation**: This mode applies no data cleaning and directly executes the chosen survival analysis model.

## Flexibility and Configuration
CleanSurvival offers flexibility in configuring its behavior and parameters. Two main configuration files are used:

1. **config.json**: This file allows you to tune the hyperparameters of the various data cleaning and machine learning methods used in the tool.

   - You can modify settings for methods like multiple imputation, KNN imputer, and neural networks.
   - This enables you to fine-tune the tool's performance for different datasets and scenarios.
2. **reward.json**: This file defines the reward matrix used in the Q-Learning algorithm.
   
   - You can adjust the rewards associated with different data cleaning actions and their combinations.
   - This allows you to influence the Q-Learning agent's exploration and learning process.


In addition to these configuration files, CleanSurvival supports importing different reward definitions. You can provide your own reward matrix or modify the existing one to suit your specific needs.

This flexibility allows you to customize CleanSurvival's behavior and adapt it to various data cleaning challenges and survival analysis tasks.


## Installation

You can install CleanSurvival as a Python package. Navigate to the `CleanSurvival` directory and run:

```bash
pip install -e .
```

This will automatically install dependencies like `scikit-survival` and `optuna`.

## How to Use

CleanSurvival can be used interactively or via batch processing:

1. **Interactive Prompt (`app.py`)**:

   - Open a terminal and navigate to the CleanSurvival directory.
   - Run `python app.py`.
   - Follow the interactive prompts to define your dataset path, time column, event column, ID columns to drop, and model parameters.

2. **Batch Processing via Terminal (`run.py`)**:

   - Open a terminal and navigate to the CleanSurvival directory.
   - Run the script with the following arguments:
   ```bash
   python run.py -d <dataset_path> -r <rewards_path> -md <model> -lm <load_mode> -lf <load_file> -a <algo> -ao <algo_op> -tc <time_col> -ec <event_col> -dc <drop_col>
   ```
   - `-d`: Path to the dataset.
   - `-r`: Path to the JSON file containing rewards (optional).
   - `-md`: Model for survival mode (RSF, COX, or NN).
   - `-lm`: Load mode for editing the Q-Learning graph (T, J, D or else for no change).
   - `-lf`: Path to the file for editing the Q-Learning graph.
   - `-a`: Cleaning algorithm (CleanSurvival (L), random (R), custom (C), or no_preparation (N)).
   - `-ao`: Options for the cleaning algorithm (number of experiments for random, or path to a file containing pipelines for custom).
   - `-tc`: Time column name.
   - `-ec`: Event column name.
   - `-dc`: Column to drop.

## Evaluation

Evaluation metrics are separated into the examples folder to show replication of the manuscript. IBS and Brier Score can be computed separately by running `python examples/replication/ibs_eval.py` and `python examples/replication/brier_eval.py`.

## Relationship to CleanSurv

This repository extends the original **CleanSurv** project with survival-analysis-specific preprocessing and a revised framework described in the paper.  
If you are looking for the earlier implementation, see: https://github.com/koka-afk/CleanSurv

## Citation

If you use CleanSurvival in your work, please cite the paper:

```bibtex
@article{koka2025cleansurvival,
  title={CleanSurvival: Automated data preprocessing for time-to-event models using reinforcement learning},
  author={Koka, Yousef and Selby, David and Gro{\ss}mann, Gerrit and Pandya, Kathan and Vollmer, Sebastian},
  journal={arXiv preprint arXiv:2502.03946},
  year={2025}
}
```
