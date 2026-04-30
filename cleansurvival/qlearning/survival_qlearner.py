import warnings
import time
import numpy as np
import json
import re
import random
import os
import os.path
from concurrent.futures import ThreadPoolExecutor
from random import randint
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# import classes 
from cleansurvival.imputation.imputer import Imputer 
from cleansurvival.duplicate_detection.duplicate_detector import Duplicate_detector
from cleansurvival.feature_selection.feature_selector import Feature_selector 
from cleansurvival.outlier_detection.outlier_detector import Outlier_detector 
from cleansurvival.survival_analysis.cox_model import CoxRegressor
from cleansurvival.survival_analysis.dh_neural_network import NeuralNetwork 
from cleansurvival.survival_analysis.random_survival_forest import RSF
from cleansurvival.regression.regressor import Regressor


def update_q(q, r, state, next_state, action, beta, gamma, states_dict):
    """Update a single Q-value using the Q-learning formula and renormalize the row.

    Q(s, a) = Q(s, a) + beta * [r(s, a) + gamma * max_a'(Q(s', a')) - Q(s, a)]

    After the update, positive Q-values in the current state's row are renormalized
    to sum to 1.

    Parameters:
    - q: The Q-value matrix (n_states x n_actions).
    - r: The reward dictionary keyed by state name.
    - state: Index of the current state.
    - next_state: Index of the next state.
    - action: Index of the action taken.
    - beta: Learning rate.
    - gamma: Discount factor.
    - states_dict: Mapping from state index to state name.

    Returns:
    - The immediate reward r(state, action).
    """
    action_name = states_dict[action]
    current_state_name = states_dict[state]
    rsa = r[current_state_name]['followed_by'][action_name]
    qsa = q[state, action]

    new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q

    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])

    q[state][q[state] > 0] = rn

    return r[current_state_name]['followed_by'][action_name]


def remove_adjacent(nums):
    """Remove consecutive duplicate elements from a list in-place.

    Parameters:
    - nums: A list from which adjacent duplicates will be removed.

    Returns:
    - nums: The modified list with adjacent duplicates removed.
    """
    previous = ''

    for i in nums[:]:  # using the copy of nums

        if i == previous:

            nums.remove(i)

        else:

            previous = i

    return nums


class SurvivalQlearner:

    def __init__(self, dataset, time_col, event_col, goal, verbose=False, json_path=None, file_name=None, threshold=None, metric="c-index", algorithm="CleanSurvival", n_episodes=1000):
        """Initialize the SurvivalQlearner.

        Parameters:
        - dataset: Input pandas DataFrame containing the survival data.
        - time_col: Name of the column representing survival time.
        - event_col: Name of the column representing the event indicator.
        - goal: Target model to optimize ('RSF', 'COX', 'NN', 'OLS', 'LASSO_REG', 'MARS').
        - verbose: If True, print detailed progress information. Default False.
        - json_path: Optional path to a JSON config file for hyperparameter settings.
        - file_name: Base name used when writing results to disk.
        - threshold: Optional threshold value (currently unused).
        - metric: Evaluation metric to use. Default 'c-index'.
        - algorithm: string defining the current task (CleanSurvival, Random, O).
        - n_episodes: Number of Q-learning training episodes used by Learn2Clean.
        """

        self.dataset = dataset

        self.time_col = time_col

        self.event_col = event_col

        self.goal = goal

        self.metric = metric

        self.algorithm = algorithm
        if self.algorithm == 'O':
            self.out_dir = './results/optuna'
        elif self.algorithm == 'Random':
            self.out_dir = './results/random'
        elif self.algorithm == 'CleanSurvival':
            self.out_dir = './results/qlearning'
        else:
            self.out_dir = './results'

        self.json_path = json_path

        if json_path is not None:
            with open(json_path) as file:
                data = json.load(file)
                self.json_file = data
        
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "reward.json")

        with open(path) as reward:
            data = json.load(reward)
            self.rewards = data

        self.verbose = verbose

        self.file_name = file_name

        self.threshold = threshold  #sds

        self.n_episodes = int(n_episodes)


    def get_params(self, deep=True):

        """
            Get parameters of the QLearner instance.

            Parameters:
            - self: The QLearner instance for which parameters are to be retrieved.
            - deep (boolean): Indicates whether to retrieve parameters deeply nested within the object.

            Returns:
            - params (dictionary): A dictionary containing the parameters of the QLearner instance.
                                The keys represent parameter names, and the values are their current values.
            """

        return {
                'goal': self.goal,           # Store the 'goal' parameter value.

                'event_col': self.event_col, # Store the 'event_col' parameter value.

                'time_col': self.time_col,   # Store the 'time_col' parameter value.

                'verbose': self.verbose,     # Store the 'verbose' parameter value.

                'file_name': self.file_name, # Store the 'file_name' parameter value.

                'threshold': self.threshold, # Store the 'threshold' parameter value.

                'n_episodes': self.n_episodes  # Store the Q-learning training episodes.

                }

    def set_params(self, **params):
        """Set parameters of the QLearner instance.

        Parameters:
        - **params: Keyword arguments corresponding to names returned by get_params().
                    Invalid parameter names produce a warning.
        """

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s)"
                              "Check the list of available parameters with "
                              "`qlearner.get_params().keys()`")

            else:

                setattr(self, k, v)


    def get_states_actions(self):
        """Return the total number of states/actions in the reward graph.

        Counts nodes with at least one outgoing edge, then adds one for the terminal state.

        Returns:
        - int: Number of states/actions.
        """
        n = 0
        for key in self.rewards:
            if len(self.rewards[key]["followed_by"]) != 0:
                n += 1
        return n + 1
    
    
    def get_imputers(self):
        """Return the number of imputer methods in the reward graph.

        Returns:
        - int: Count of nodes with type 'Imputer'.
        """
        imputer_no = 0
        for key in self.rewards:
            if self.rewards[key]['type'] == "Imputer":
                imputer_no += 1
        return imputer_no
    

    def get_methods(self):
        """Return the names of all preprocessing methods (non-model nodes) in the reward graph.

        Returns:
        - list[str]: Method names excluding survival models and regression goals.
        """
        methods = []
        for key in self.rewards:
            if self.rewards[key]['type'] not in ('Survival_Model', 'Regression'):
                methods.append(key)
        return methods
    

    def get_goals(self):
        """Return the names of all goal nodes (survival models and regressors) in the reward graph.

        Returns:
        - list[str]: Goal names of type 'Survival_Model' or 'Regression'.
        """
        goals = []
        for key in self.rewards:
            if self.rewards[key]['type'] in ('Survival_Model', 'Regression'):
                goals.append(key)
        return goals


    def edit_edge(self, u, v, weight):
        """Add, update, or remove an edge in the reward graph.

        Parameters:
        - u: Source node name.
        - v: Target node name.
        - weight: Reward weight for the edge. Pass -1 to remove the edge.
        """
        if weight == -1:
            self.rewards[u]['followed_by'].pop(v, None)
        else:
            self.rewards[u]['followed_by'][v] = weight

    
    def set_rewards(self, data):
        """Replace the entire reward graph with the provided data.

        Parameters:
        - data: A dictionary in the same format as reward.json.
        """
        self.rewards = data


    def disable(self, op):
        """Remove a preprocessing operation and all its edges from the reward graph.

        Accepts either a type name (e.g., 'Imputer') to remove all methods of that type,
        or a specific method name (e.g., 'Median') to remove just that method.

        Parameters:
        - op: Operation type or method name to disable.
        """
        ops_names = []
        for key in self.rewards:
            if self.rewards[key]['type'] == op:
                ops_names.append(key)
        for val in ops_names: # loop in case op parameter was a preprocessing step like "Imputer"
            for key in self.rewards:
                self.rewards[key]['followed_by'].pop(val, None)
            self.rewards.pop(val, None)

        for key in self.rewards: # loop in case op parameter was a single preprocessing method like "Median"
            self.rewards[key]['followed_by'].pop(op, None)
        self.rewards.pop(op, None)

    def Initialization_Reward_Matrix(self, dataset):
        """ [Data Preprocessing Reward/Connection Graph]

            This function initializes a reward matrix based on the input dataset.

            State: Initial Data

            Methods (Actions):
            1. CCA (missing values)
            2. MI (missing values)
            3. IPW (missing values)
            4. Mean (missing values)
            5. Median (missing values)
            6. UC (feature selection)
            7. LASSO (feature selection)
            8. RFE (feature selection)
            9. IG (feature selection)
            10. ED (deduplication)
            11. DBID (deduplication)
            12. DBT (deduplication)
            13. CR (outlier detection)
            14. MR (outlier detection)
            15. MUO (outlier detection)
            16. RSF (Survival model)
            17. COX (Survival model)
            18. NN (Survival model) 
        """
        # Check if there are missing values in the dataset
        if dataset.copy().isnull().sum().sum() > 0:

            r = self.rewards
            
            # Define the number of actions and states
            n_actions = self.get_states_actions()

            n_states = self.get_states_actions()

            check_missing = True

        else:  

            r = self.rewards

            # Define the number of actions and states
            n_actions = self.get_states_actions()

            n_states = self.get_states_actions()

            imputer_no = self.get_imputers()
            
            n_actions -= imputer_no
            n_states -= imputer_no

            check_missing = False

        # Initialize a Q matrix with zeros
        zeros_mat = [[0.0 for x in range(n_actions)] for y in range(n_states)]
        q = np.array(zeros_mat)

        # Print the reward matrix if verbose mode is enabled
        if self.verbose:

            print("Reward matrix")

            print(r)

        # Return the initialized Q matrix, reward matrix, number of actions, number of states, and a flag for missing values
        return q, r, n_actions, n_states, check_missing
    

    def get_config_file(self, class_name):
        """Retrieve the hyperparameter configuration for a given class from the JSON config file.

        Parameters:
        - class_name: Name of the class to look up (e.g., 'RSF', 'CoxRegressor').

        Returns:
        - dict or None: Configuration dictionary if found, None otherwise.
        """
        config = None
        if self.json_path is not None:
            if class_name in self.json_file.keys():
                config = self.json_file[class_name]
        return config
    

    def handle_categorical(self, dataset):
        """Ordinally encode all non-numeric columns in the dataset, preserving NaN values.

        Parameters:
        - dataset: Input pandas DataFrame.

        Returns:
        - tuple: (encoded DataFrame, dict mapping column names to their OrdinalEncoder instances).
        """
        from sklearn.preprocessing import OrdinalEncoder
        from pandas.api.types import is_numeric_dtype

        data = dataset

        print(f"\n\n **HANDLE CATEGORICAL WITHOUT IMPUTATION** \n\n {data}")

        oe_dict = {}

        for col_name in data:
            if not is_numeric_dtype(data[col_name]):
                oe_dict[col_name] = OrdinalEncoder()
                col = data[col_name]
                col_not_null = col[col.notnull()]
                reshaped_values = col_not_null.values.reshape(-1, 1) # TODO is this reshaping really needed? It might cause problems
                encoded_values = oe_dict[col_name].fit_transform(reshaped_values)
                data.loc[col.notnull(), col_name] = np.squeeze(encoded_values)
        
        print(f"\n\n **HANDLE CATEGORICAL WITHOUT IMPUTATION** \n\n {data}")

        return data, oe_dict
    

    def construct_pipeline(self, dataset, actions_list, time_col, event_col, check_missing):

        """
        This function represents a data preprocessing pipeline that applies a series of actions to the input dataset
        based on the provided list of actions. It can handle missing values and perform various data preprocessing steps.

        Parameters:
        - dataset: The input dataset to be preprocessed.
        - actions_list: A list of action indices indicating which data preprocessing steps to perform.
        - time_col: The name of the column representing time information.
        - event_col: The name of the column representing event information.
        - check_missing: A flag indicating whether to check for missing values in the dataset.

        Returns:
        - n: The preprocessed dataset after applying the specified actions.
        - res: Reserved for potential future use (currently set to None).
        - t: The CPU time taken to complete the data preprocessing pipeline.
        """

        # Define names of goals (used when executing survival models)
        goals_name = self.get_goals() #["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        # Initialize the result variable as None
        res = None

        # Check if missing values should be handled
        if check_missing:

            # Define names of actions (methods) for preprocessing
            actions_name = ["Mean", "CCA", "MI", "KNN", "Median",
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"] # TODO Replace "CR" action with "MR" until fixed ... and IPW with MI
            #update on 25th of june 2024 TODO replaced Mean with my version of KNN  

            # Define a list of classes corresponding to each action (used for instantiation)
            L2C_class = [Imputer, Imputer, Imputer, Imputer, Imputer,
                         Feature_selector, Feature_selector, Feature_selector, Feature_selector,
                         Duplicate_detector, Duplicate_detector, Duplicate_detector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         RSF, CoxRegressor, NeuralNetwork, Regressor, Regressor, Regressor
                         ]

        else:
            # If no missing values handling is needed, define names of other actions
            actions_name = [
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"] # TODO Replace "CR" action with "MR" until fixed

            # Define a list of classes corresponding to each action (used for instantiation)
            L2C_class = [Feature_selector, Feature_selector, Feature_selector, Feature_selector,
                         Duplicate_detector, Duplicate_detector, Duplicate_detector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         RSF, CoxRegressor, NeuralNetwork, Regressor, Regressor, Regressor]

        print()

        print("Start pipeline")
        print(actions_list)
        print("-------------")

        start_time = time.time()

        n = None

        for a in actions_list:

            if not check_missing:

                if a in range(0, 6):

                    # Deduplication (0-2) and feature selection (3-6) based on the action index.
                    config = None
                    if self.json_path is not None:
                        if a <= 3:
                            config = self.get_config_file("Feature_selector")
                        else:
                            config = self.get_config_file("Duplicate_detector")
                    
                    dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, strategy = actions_name[a], config=config, verbose=self.verbose, metric=self.metric).transform()

                if a in (7, 8, 9):
                    # Execute outlier detectors (7-9) based on the action index.

                    config = self.get_config_file("Outlier_detector")

                    print(f"IN OUTLIER CALL: \n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                    dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, strategy = actions_name[a], verbose = self.verbose).transform()
                    print(f"IN OUTLIER CALL: \n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                if a == 10:
                    # Execute Random Survival Forest
                    print(f'\nIN RSF --------------------------------> {dataset}\n\n')
                    import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_RSF_cleaned.csv", index=False)
                    config = self.get_config_file("RSF")
                    rsf = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose, metric=self.metric)
                    survival_probabilities, c_index = rsf.fit_rsf_model()
                    n = {"quality_metric": c_index}
                    print(f"IN RSF --------------------------------> {c_index} \n\n\n\n {n}")
                
                if a == 11:
                    # Execute Cox Model
                    print("\n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                    import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_COX_cleaned.csv", index=False)
                    # TODO continue developing this file starting by adding "mode" parameter and then adjusting this part (cox) and then continue
                    config = self.get_config_file("CoxRegressor")
                    res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                    cox_model = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose=self.verbose, metric=self.metric)
                    c_index = cox_model.updated_fit()
                    if isinstance(c_index, np.generic):
                            c_index = c_index.item()
                    time_dif = time.time() - start_time
                    n = {"quality_metric": c_index, 'time': time_dif}
                    print(f"IN SURVIVAL_QLEARNER --------------------------------> {c_index} \n\n\n\n {n}")

                if a == 12:
                    # Execute Neural Network
                    import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_NN_cleaned.csv", index=False)
                    config = self.get_config_file("NeuralNetwork")
                    res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                    nn = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose=self.verbose, metric=self.metric)
                    c_index = nn.fit_dh()
                    n = {"quality_metric": c_index}
                
                if a in (13, 14, 15):
                    config = self.get_config_file(self.goal)
                    res = dataset
                    reg = L2C_class[a](dataset=dataset, target=self.event_col, strategy=self.goal, verbose=self.verbose)
                    quality_metric = reg.transform()
                    n = quality_metric
                    

            else:

                if (dataset is not None and len(dataset.dropna()) == 0):

                    pass

                else:

                    config = None

                    if a in (0, 1, 2, 3, 4):
                        # Execute missing values handling methods (0-4) based on the action index.

                        config = self.get_config_file("Imputer")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()
                        #print("HANDLING MISSING VALUES: " + str(len(n)))
                    if a in (5, 6, 7, 8):
                        # Execute Feature selection methods (5-8) based on the action index.

                        config = self.get_config_file("Feature_selector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a in (9, 10, 11):
                        # Execute deduplication methods (9-11) based on the action index.

                        config = self.get_config_file("Duplicate_detector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a in (12, 13, 14):
                        # Execute outlier detection methods (12-14) based on the action index.

                        config = self.get_config_file("Outlier_detector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a == 15:
                        # Execute Random Survival Forest
                        print(f'\nIN RSF --------------------------------> {dataset}\n\n')
                        import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_RSF_cleaned.csv", index=False)
                        config = self.get_config_file("RSF")
                        rsf = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose, metric=self.metric)
                        survival_probabilities, c_index = rsf.fit_rsf_model()
                        n = {"quality_metric": c_index}
                        print(f"IN RSF --------------------------------> {c_index} \n\n\n\n {n}")

                    if a == 16:
                        # Execute Cox Model
                        import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_COX_cleaned.csv", index=False)
                        config = self.get_config_file("CoxRegressor")
                        res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                        cox_model = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose, metric=self.metric)
                        c_index = cox_model.updated_fit()
                        if isinstance(c_index, np.generic):
                            c_index = c_index.item()
                        time_dif = time.time() - start_time
                        n = {"quality_metric": c_index, 'time': time_dif}
                        print(f"IN SURVIVAL_QLEARNER --------------------------------> {c_index} \n\n\n\n {n}")

                    if a == 17:
                        # Execute Neural Network
                        import os; os.makedirs(self.out_dir, exist_ok=True); dataset.to_csv(f"{self.out_dir}/{self.file_name}_pipeline_{'_'.join(str(x) for x in actions_list)}_NN_cleaned.csv", index=False)
                        config = self.get_config_file("NeuralNetwork")
                        res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                        nn = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose=self.verbose, metric=self.metric)
                        c_index = nn.fit_dh()
                        n = {"quality_metric": c_index}
                    
                    if a in (18, 19, 20):
                        config = self.get_config_file(self.goal)
                        res = dataset
                        reg = L2C_class[a](dataset=dataset, target=self.event_col, strategy=self.goal, verbose=self.verbose)
                        quality_metric = reg.transform()
                        n = quality_metric
                        if n == 0:
                            n = {'quality_metric': 0}

        
        # Calculate the elapsed CPU time
        t = time.time() - start_time

        print("End Pipeline CPU time: %s seconds" % (time.time() - start_time))

        # Return the preprocessed dataset, result, and CPU time
        return n, res, t

    def show_traverse(self, dataset, q, g, check_missing):
        # show all the greedy traversals
        """
        This function displays all the greedy traversals of the reinforcement learning agent based on the learned Q-values.
        It explores different strategies for preprocessing data based on the Q-matrix.

        Parameters:
        - dataset: The input dataset to be preprocessed.
        - q: The Q-matrix representing the learned state-action values.
        - g: The index of the survival model goal.
        - check_missing: A flag indicating whether to check for missing values in the dataset.

        Returns:
        - actions_strategy: A list of strings describing the actions taken in each strategy.
        - strategy: A list of quality metrics corresponding to each strategy.
        """

         # Define lists of methods and goals based on whether missing values should be handled
        if check_missing:

            methods, goals = ["Mean", "CCA", "MI", "KNN", "Median",
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"], ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        else:

            methods, goals = ["UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"], ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        n_states = len(methods) + 1

        # Append the current goal to the list of methods (for traversal visualization)
        methods.append(str(goals[g]))

        strategy = []

        actions_strategy = []

        final_dataset = None

        for i in range(len(q)-1):
            
            # This 'for' loop iterates through the states (methods) represented by the Q-matrix, excluding the last state.
            actions_list = []

            current_state = i

            current_state_name = methods[i]
            traverse_name = "%s -> " % current_state_name

            n_steps = 0

            while current_state != n_states-1 and n_steps < 17:
                # This 'while' loop continues until either the current state is the final goal state or a maximum of 17 steps is reached.
                actions_list.append(current_state)

                next_state = np.argmax(q[current_state])

                current_state = next_state

                current_state_name = methods[next_state]
                traverse_name += "%s -> " % current_state_name

                actions_list.append(next_state)

                n_steps = n_steps + 1

                actions_list = remove_adjacent(actions_list)

            if not check_missing:

                traverse_name = traverse_name[:-4]

                del actions_list[-1]
                actions_list.append(g+len(methods)-1)

            else:

                del actions_list[-1]

                actions_list.append(g+len(methods)-1)

                traverse_name = traverse_name[:-4]

            print(f'BEFORE CHECK MISSING IF CONDITION ---> {check_missing}')

            if check_missing: # this if statement ensures that if there are NaNs in the dataset, there must be imputation 'before' model
                print("\n\n IN IMPUTATION CHECK >>>>>>>>>>>>>>>>\n\n")
                temp = traverse_name.split(" -> ")
                print(f'HERE IS TEMP ---> {temp} \n\n {actions_list} \n\n')
                has_imputer = False
                name = "" 
                imputer_list = ["Mean", "CCA", "MI", "KNN", "Median"]
                for im in imputer_list:
                    if im in temp:
                        has_imputer = True
                        name = im
                        break
                
                if not has_imputer:
                    random_index = 0
                    random_imputer = randint(0, 4)
                    temp.insert(random_index, imputer_list[random_imputer])
                    actions_list.insert(random_index, random_imputer)
                    traverse_name = ""
                    for item in range(len(temp) - 1):
                        traverse_name = traverse_name + temp[item] + " -> "
                    traverse_name += str(self.goal)
                    traverse_name = traverse_name.strip()
                else:
                    index = temp.index(name)
                    temp.pop(index)
                    temp.insert(0, name) # adjusting string sequence of strategy
                    pos = imputer_list.index(name)
                    # Some traversals may include imputer name in the string view while
                    # the numeric path no longer contains its corresponding action index.
                    # In that case, prepend the imputer instead of failing.
                    if pos in actions_list:
                        index_action_list = actions_list.index(pos)
                        actions_list.pop(index_action_list)
                    actions_list.insert(0, pos) # adjusting actual list of actions
                    traverse_name = ""
                    for item in range(len(temp) - 1):
                        traverse_name = traverse_name + temp[item] + " -> "
                    traverse_name += str(self.goal)
                    traverse_name = traverse_name.strip()
            else:
                dataset = self.handle_categorical(dataset)[0] # encoding categorical values outside Imputer class
                

            for idx in range(len(actions_list)): # convert numbers that for some reason randomly change to numpy type Ex. 2 -> np.int64(2)
                num = actions_list[idx]
                if isinstance(num, np.generic):
                    actions_list[idx] = num.item()
            
            print("\n\nStrategy#", i, ": Greedy traversal for "
                  "starting state %s" % methods[i])

            print(traverse_name)

            print(actions_list)

            
                    

            actions_strategy.append(traverse_name)

            # Execute the preprocessing pipeline for the current strategy and store the quality metric
            dataset_copy = dataset.copy()
            temp_val = self.construct_pipeline(dataset_copy, actions_list, self.time_col, self.event_col, check_missing)
            print(f'temp_val pipeline \n {temp_val}')
            strategy.append(temp_val[0])
            final_dataset = temp_val[1]

        print()

        print("==== Recap ====\n")

        print("List of strategies tried by Learn2Clean:")

        print(actions_strategy)

        print('\nList of corresponding quality metrics ****\n',
              strategy)

        print()

        return actions_strategy, strategy

    def Learn2Clean(self):

        """
        This function represents the main Learn2Clean algorithm. It learns an optimal policy for data preprocessing using Q-learning
        and executes the best strategy based on quality metrics for the specified survival analysis goal.

        Returns:
        - rr: A tuple containing information about the best strategy and its performance.
        """

        goals = ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

         # Check if the specified goal is valid
        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between RSF, COX, NN, OLS, LASSO or MARS")

        else:

            g = goals.index(self.goal)

            pass

        start_l2c = time.time()

        print("Start Learn2Clean")

        gamma = 0.91 #0.8

        beta = 0.1 #1.

        n_episodes = self.n_episodes

        epsilon = 0.2 #0.05

        random_state = np.random.RandomState(1999)

        # Initialize Q-matrix, reward matrix, number of actions, number of states, and missing value flag
        q, r, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        state_names = []
        for key in self.rewards:
            if (self.rewards[key]["type"] == "Survival_Model" or self.rewards[key]["type"] == "Regression") and key != self.goal:
                continue
            if check_missing:
                state_names.append(key)
            else:
                if self.rewards[key]['type'] != 'Imputer':
                    state_names.append(key)

        print(state_names)
        states_dict = {}
        states_dict_reversed = {}
        i = 0
        for x in state_names:
            states_dict[i] = x
            states_dict_reversed[x] = i
            i += 1

        for e in range(int(n_episodes)):

            states = list(range(n_states))

            random_state.shuffle(states)

            current_state = states[0]

            goal = False

            r_mat = r.copy()

            if e % int(n_episodes / 10.) == 0 and e > 0:

                pass

            while (not goal) and (current_state != n_states-1):

                # Epsilon-greedy exploration strategy to select actions
                valid = r_mat[states_dict[current_state]]['followed_by']

                temp = [] #r[current_state] >= 0
                for valid_state in valid:
                    if valid_state in states_dict_reversed.keys():
                        temp.append(states_dict_reversed[valid_state])
                valid_moves = [False for x in  range(n_states)]
                for x in temp:
                    valid_moves[x] = True
                valid_moves = np.array(valid_moves)
                

                if random_state.rand() < epsilon:

                    actions = np.array(list(range(n_actions)))

                    actions = actions[valid_moves]

                    if type(actions) is int:

                        actions = [actions]

                    random_state.shuffle(actions)

                    action = actions[0]

                    next_state = action

                else:


                    if np.sum(q[current_state]) > 0:

                        action = np.argmax(q[current_state])

                    else:

                        actions = np.array(list(range(n_actions)))

                        actions = actions[valid_moves]

                        random_state.shuffle(actions)

                        action = actions[0]

                    next_state = action
               
                reward = update_q(q, r, current_state, next_state, action, beta, gamma, states_dict)

                if reward > 1:

                    goal = True

                np.delete(states, current_state)

                current_state = next_state

        if self.verbose:

            print("Q-value matrix\n", q)

        print("Learn2Clean - Pipeline construction -- CPU time: %s seconds"
              % (time.time() - start_l2c))

        metrics_name = ["C-Index", "C-Index", "C-Index", "MSE", "MSE", "MSE"]

        print("=== Start Pipeline Execution ===")

        print(q)

        start_pipexec = time.time()

        # Execute strategies and store results
        result_list = self.show_traverse(self.dataset, q, g, check_missing)

        quality_metric_list = []
        best_overall = [] # for testing purposes -> maintain best obtained value so far
        timestamps = [] # for testing purposes -> when the pipelines finished

        print(f'result_list: \n {result_list}')

        if result_list[1]:
            

            for dic in range(len(result_list[1])):
                print(f'In QLearning: \n {result_list[1][dic]}')

                if result_list[1][dic] != None:
                    for key, val in result_list[1][dic].items():

                        if key == 'quality_metric':

                            quality_metric_list.append(val)
                        elif key == 'time':
                            timestamps.append(val)

            if g in range(0, 2):

                result = max(x for x in quality_metric_list if x is not None)  # changed from min to max

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for maximal ', # print changed from 'minimal' to 'maximal'
                      result, 'for', goals[g])

                print()

            else:

                result = min(x for x in quality_metric_list if x is not None)

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for Minimal',
                      metrics_name[g], ':', result, 'for', goals[g])

                print()

        else:

            result = None

            result_l = None
        
        best_so_far = 0
        for num in quality_metric_list: # populate the best_overall
            best_so_far = max(num, best_so_far)
            best_overall.append(best_so_far)
        for i in range(1, len(timestamps)):
            timestamps[i] += timestamps[i - 1]

        t = time.time() - start_pipexec

        print("=== End of Learn2Clean - Pipeline execution "
              "-- CPU time: %s seconds" % t)

        print()

        if result_l is not None:

            rr = (self.file_name, "CleanSurvival", goals[g], result_list[0][result_l], metrics_name[g], result, t)

        else:

            rr = (self.file_name, "CleanSurvival", goals[g], None, metrics_name[g], result, t)

        print("**** Best strategy ****")

        # Return information about the best strategy and its performance
        print(rr)
        
        with open(self.out_dir + '/'+str(self.file_name)+'_results.txt',
                  mode='a+') as rr_file:

            print("{}".format(rr), file=rr_file)

        best_overall.insert(0, "Best So Far")
        best_overall.insert(0, "CleanSurv")
        timestamps.insert(0, "Timestamps")
        timestamps.insert(0, "CleanSurv")
        with open(self.out_dir + '/'+str(self.file_name)+'_timestamps.txt', mode='a') as rr_file:
            print("{}".format(best_overall), file=rr_file)
            print("{}".format(timestamps), file=rr_file)

        


    def optuna_search(self, dataset_name="None", loop=1):
        """
         This function generates an Optuna based cleaning strategy and executes it on the dataset.
         Using Tree-structured Parzen Estimator (TPE) algorithm as a powerful SOTA baseline.
         
         Args:
         - dataset_name: The name of the dataset being cleaned.
         - loop: The number of trials to run (optuna n_trials).
        """
        check_missing = self.dataset.isnull().sum().sum() > 0
        rr = ""
        obtained_scores = []
        best_so_far = []
        timestamps = []
        start_time = time.perf_counter()

        def objective(trial):
            random.seed(time.perf_counter())

            if check_missing:
                methods = ["-", "CCA", "MI", "Mean", "KNN", "Median", "-", "UC", "LASSO", "RFE", "IG",
                        "-", "DBID", "DBT", "ED", "-", "MR", "MR", "MUO", "-", "-", "-"]
                a1 = trial.suggest_int("imp", 1, 5)     # Imputation
                a2 = trial.suggest_int("fs", 7, 10)     # Feature Selection
                a3 = trial.suggest_int("od", 12, 14)    # Outlier Detection
                # actions for duplicate detection are always 16-18
                a4 = trial.suggest_int("dd", 16, 18)
                rand_actions_list = [a1, a2, a3, a4, 19] # 19 is dummy for goal, overridden later
            else:
                methods = ["-", "UC", "LASSO", "RFE", "IG", "-",  "DBID", "DBT", "ED",
                        "-",  "MR", "MR", "MUO", "-", "-", "-"]
                a1 = trial.suggest_int("fs", 1, 4)      # Feature Selection
                a2 = trial.suggest_int("od", 6, 8)      # Outlier Detection
                a3 = trial.suggest_int("dd", 10, 12)
                rand_actions_list = [a1, a2, a3, 13]

            goals = ["RSF", "COX", "NN"]
            metrics_name = ["C-Index", "C-Index", "C-Index"]
            if self.goal not in goals:
                raise ValueError("Goal invalid. Please choose between RSF, COX, NN")
            else:
                g = goals.index(self.goal)

            traverse_name = methods[rand_actions_list[0]] + " -> "
            for i in range(1, len(rand_actions_list)):
                traverse_name += "%s -> " % methods[rand_actions_list[i]]
            traverse_name = re.sub('- -> ', '', traverse_name) + goals[g]
            name_list = re.sub(' -> ', ',', traverse_name).split(",")

            if check_missing:
                rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-6
                clean_methods = ["CCA", "MI", "Mean", "KNN", "Median", "UC", "LASSO", "RFE", "IG", "DBID", "DBT", "ED", "MR", "MR", "MUO"]
                new_list = []
                for i in range(len(name_list)-1):
                    m = clean_methods.index(name_list[i])
                    new_list.append(m)
                new_list.append(g+len(clean_methods))
            else:
                self.dataset = self.handle_categorical(self.dataset)
                rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-5
                clean_methods = ["UC", "LASSO", "RFE", "IG", "DBID", "DBT", "ED", "MR", "MR", "MUO"]
                new_list = []
                for i in range(len(name_list)-1):
                    m = clean_methods.index(name_list[i])
                    new_list.append(m)
                new_list.append(g+len(clean_methods))
            
            dataset_copy = self.dataset.copy()
            try:
                p = self.construct_pipeline(dataset=dataset_copy, actions_list=new_list, time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
                score = p[0]['quality_metric']
            except Exception as e:
                print(e)
                score = 0
            
            nonlocal rr
            rr += str((dataset_name, "Optuna", goals[g], traverse_name, metrics_name[g], "Quality Metric: ", score)) + "\n"
            obtained_scores.append(score)
            
            # record time and best
            current_best = max(obtained_scores) if obtained_scores else 0
            best_so_far.append(current_best)
            timestamps.append(time.perf_counter() - start_time)

            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=loop)

        # compute statistics
        average = sum(obtained_scores)
        mean = average / len(obtained_scores) if obtained_scores else 0
        sum_sq = sum((x - mean) ** 2 for x in obtained_scores)
        standard_deviation = (sum_sq / len(obtained_scores)) ** 0.5 if obtained_scores else 0
        
        print(rr)
        print(f"**Average score over {loop} experiments is: {mean}**")
        print(f"**Standard deviation:{standard_deviation}**")
        average_score_str = f"**Average score over {loop} experiments is: {mean}**\n**Standard deviation:{standard_deviation}**\n\n"
        rr += average_score_str

        # save to file (similar to random_search behavior to log bests and timestamps)
        best_overall = best_so_far.copy()
        time_overall = timestamps.copy()
        best_overall.insert(0, "Best So Far")
        best_overall.insert(0, "Optuna")
        time_overall.insert(0, "Timestamps")
        time_overall.insert(0, "Optuna")
        
        with open(self.out_dir + '/' + str(self.file_name) + '_results.txt', mode='a+') as rr_file:
            print("{}".format(rr), file=rr_file)
        
        with open(self.out_dir + '/' + str(self.file_name) + '_timestamps.txt', mode='a+') as rr_file:
            print("{}".format(best_overall), file=rr_file)
            print("{}".format(time_overall), file=rr_file)

        return None

    def random_cleaning(self, dataset_name="None", loop=1):

        """
         This function generates a random cleaning strategy and executes it on the dataset.
        
         Args:
         - dataset_name: The name of the dataset being cleaned.
        
         Returns:
         - p[1]: The result of the cleaning strategy, including quality metrics.
        """

        check_missing = self.dataset.isnull().sum().sum() > 0
        rr = ""
        average = 0
        obtained_scores = []
        timestamps = []

        if not check_missing:
            # Encode once before parallel runs to avoid repeated work.
            self.dataset, _ = self.handle_categorical(self.dataset)

        goals = ["RSF", "COX", "NN"]
        metrics_name = ["C-Index", "C-Index", "C-Index"]

        if self.goal not in goals:
            raise ValueError("Goal invalid. Please choose between RSF, COX, NN")

        g = goals.index(self.goal)

        def _run_one_repeat(repeat_index):
            start_t = time.time()
            rng = random.Random(time.perf_counter() + repeat_index)

            p_result = ({'quality_metric': 0}, None, 0)
            score = 0
            traverse_name = ""
            last_error = None

            for _attempt in range(3):
                if check_missing:
                    methods = ["-", "CCA", "MI", "Mean", "KNN", "Median", "-", "UC", "LASSO", "RFE", "IG",
                               "-", "DBID", "DBT", "ED",
                               "-", "MR", "MR", "MUO",
                               "-", "-", "-"]

                    rand_actions_list = [
                        rng.randint(1, 5),   # imputation
                        rng.randint(7, 10),  # feature selection
                        rng.randint(12, 14), # duplicate detection
                        rng.randint(16, 18), # outlier detection
                        rng.randint(19, 21), # placeholder before goal replacement
                    ]
                else:
                    methods = ["-", "UC", "LASSO", "RFE", "IG",
                               "-", "DBID", "DBT", "ED",
                               "-", "MR", "MR", "MUO",
                               "-", "-", "-"]

                    rand_actions_list = [
                        rng.randint(1, 4),   # feature selection
                        rng.randint(6, 8),   # duplicate detection
                        rng.randint(10, 12), # outlier detection
                        rng.randint(13, 15), # placeholder before goal replacement
                    ]

                traverse_name = methods[rand_actions_list[0]] + " -> "
                for i in range(1, len(rand_actions_list)):
                    traverse_name += "%s -> " % methods[rand_actions_list[i]]
                traverse_name = re.sub('- -> ', '', traverse_name) + goals[g]
                name_list = re.sub(' -> ', ',', traverse_name).split(",")

                if check_missing:
                    methods = ["CCA", "MI", "Mean", "KNN", "Median",
                               "UC", "LASSO", "RFE", "IG",
                               "DBID", "DBT", "ED",
                               "MR", "MR", "MUO"]

                    new_list = []
                    for i in range(len(name_list)-1):
                        m = methods.index(name_list[i])
                        new_list.append(m)
                    new_list.append(g + len(methods))
                else:
                    methods = ["UC", "LASSO", "RFE", "IG",
                               "DBID", "DBT", "ED",
                               "MR", "MR", "MUO"]
                    new_list = []
                    for i in range(len(name_list)-1):
                        m = methods.index(name_list[i])
                        new_list.append(m)
                    new_list.append(g + len(methods))

                dataset_copy = self.dataset.copy()
                try:
                    p_result = self.construct_pipeline(
                        dataset=dataset_copy,
                        actions_list=new_list,
                        time_col=self.time_col,
                        event_col=self.event_col,
                        check_missing=check_missing
                    )
                    score = p_result[0]['quality_metric']
                    break
                except Exception as e:
                    last_error = e

            if last_error is not None and score == 0:
                print(f"Random cleaning pipeline failed on repeat {repeat_index} after retries: {last_error}")

            elapsed = time.time() - start_t
            result_line = str((dataset_name, "Random", goals[g], traverse_name, metrics_name[g], "Quality Metric: ", score)) + "\n"
            return p_result, score, elapsed, result_line

        max_workers = max(1, min(loop, os.cpu_count() or 1))
        p = ({'quality_metric': 0}, None, 0)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for p, score, elapsed, result_line in executor.map(_run_one_repeat, range(loop)):
                timestamps.append(elapsed)
                rr += result_line
                average += score
                obtained_scores.append(score)

        mean = average / loop
        for i in  range(len(obtained_scores)):
            obtained_scores[i] -= mean
            obtained_scores[i] = obtained_scores[i] ** 2
        standard_deviation = (sum(obtained_scores) / len(obtained_scores)) ** (1.0 / 2.0)
        print(rr)
        print(f"**Average score over {loop} experiments is: {average/loop}**")
        print(f"**Standard deviation:{standard_deviation}**")
        average_score_str = f"**Average score over {loop} experiments is: {average/loop}**\n**Standard deviation:{standard_deviation}**\n\n"
        rr += average_score_str

        if p[1] is not None:

            with open(self.out_dir + '/'+dataset_name+'_results.txt',
                    mode='a+') as rr_file:

                print("{}".format(rr), file=rr_file)

        obtained_scores_copy = list(obtained_scores)
        obtained_scores_copy.insert(0, "Best So Far")
        obtained_scores_copy.insert(0, "Random")
        timestamps.insert(0, "Timestamps")
        timestamps.insert(0, "Random")
        
        with open(self.out_dir + '/'+dataset_name+'_timestamps.txt', mode='a+') as rr_file:
            print("{}".format(obtained_scores_copy), file=rr_file)
            print("{}".format(timestamps), file=rr_file)

        return p[1]
    

    def custom_pipeline(self, pipelines_file, model_name, dataset_name="None"):
        """Execute one or more user-specified preprocessing pipelines and record results.

        Parameters:
        - pipelines_file: An iterable of strings, each describing a pipeline as
          space-separated method names (e.g., ['Mean UC DBID MR']).
        - model_name: Survival model to use ('RSF', 'COX', 'NN').
        - dataset_name: Dataset identifier used for output file naming. Default 'None'.

        Returns:
        - p: Result tuple from the last pipeline executed via construct_pipeline.
        """
        pipeline_counter = 0
        rr = ""

        for line in pipelines_file:
            steps = list(line.split(" "))
            goals = ["RSF", "COX", "NN"]
            metrics_name = ["C-Index", "C-Index", "C-Index"]
            methods = ["UC", "LASSO", "RFE", "IG",
                        "DBID", "DBT", "ED",
                        "MR", "MR", "MUO"]
            
            g = goals.index(model_name)
            missing = False
            for step in steps:
                if step not in methods:
                    methods = ["CCA", "MI", "Mean", "KNN", "Median",
                                "UC", "LASSO", "RFE", "IG",
                                "DBID", "DBT", "ED",
                                "MR", "MR", "MUO"]
                    missing = True
                    break 

            steps.append(model_name)
            action_list = []
            traverse_name = ""

            for i in range(len(steps) - 1):
                name = "".join(steps[i].splitlines())
                print(name)
                steps[i] = name
                traverse_name += steps[i] + " -> "
                m = methods.index(steps[i])
                action_list.append(m)

            traverse_name += model_name
            action_list.append(g+len(methods))
            check_missing = missing

            print()

            print()

            print("--------------------------")

            print("Custom Pipeline strategy:\n", traverse_name)

            print("--------------------------")

            print(traverse_name)
            print(action_list)
            dataset_copy = self.dataset.copy()
            p = self.construct_pipeline(dataset=dataset_copy, actions_list=action_list, time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
            print(f'P IS HERE {p}')
            print()
            rr += str((dataset_name, "Custom", goals[g], traverse_name, metrics_name[g], "Quality Metric: ", p[0]['quality_metric'])) + "\n"
            pipeline_counter += 1
        print(rr)

        with open(self.out_dir + '/'+str(self.file_name)+'_results.txt',
                  mode='a+') as rr_file:
            print("{}".format(rr), file=rr_file)

        print(f'**{pipeline_counter} Strategies Have Been Tried**')
        return p
    

    def no_prep(self, dataset_name='None'):
        """Run the survival model with no preprocessing as a baseline.

        If missing values are present, they are dropped via CCA before fitting.
        Categorical columns are ordinally encoded.

        Parameters:
        - dataset_name: Dataset identifier used for output file naming. Default 'None'.
        """

        goals = ["RSF", "COX", "NN"]

        metrics_name = ["C-Index", "C-Index", "C-Index"]

        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between RSF, COX, NN")

        else:

            g = goals.index(self.goal)

        check_missing = self.dataset.isnull().sum().sum() > 0

        if check_missing:
            self.dataset.dropna(inplace=True)
            len_m = 15

        else:
            len_m = 10
        
        self.dataset = self.handle_categorical(self.dataset)[0]

        p = self.construct_pipeline(dataset=self.dataset, actions_list=[g+len_m], time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
        rr = (dataset_name, "no-prep", goals[g], goals[g], metrics_name[g],"Quality Metric: ", p[0]['quality_metric'])

        print(f'\n\n{rr}\n\n')

        if p[1] is not None:

            with open(self.out_dir + '/'+dataset_name+'_results.txt',
                      mode='a') as rr_file:

                print("{}".format(rr), file=rr_file)
    

    
    def get_imputers(self):
        """Return the names of all imputer methods in the reward graph.

        Returns:
        - list[str]: Method names with type 'Imputer'.
        """
        imputers = []
        for method in self.rewards:
            if self.rewards[method]['type'] == 'Imputer':
                imputers.append(method)
        return imputers
    

    def generate_pipeline(self, current_step, pipeline, imputers):
        """Recursively build and execute all valid pipelines starting from a given step.

        Traverses the reward graph depth-first. When a terminal node (the goal or a leaf)
        is reached, prepends a randomly selected imputer and runs the pipeline via
        custom_pipeline.

        Parameters:
        - current_step: Name of the current preprocessing step.
        - pipeline: List of steps accumulated so far (passed by value via .copy()).
        - imputers: List of imputer method names to choose from.

        Returns:
        - The result of custom_pipeline for terminal paths, None for intermediate nodes.
        """
        pipeline.append(current_step)
        if current_step == self.goal or len(self.rewards[current_step]['followed_by']) == 0:
            pipeline.pop()
            random_imputer = imputers[random.randint(0, len(imputers) - 1)]
            pipeline.insert(0, random_imputer)
            formatted_pipeline = ""
            for method in pipeline:
                formatted_pipeline += method + " "
            formatted_pipeline = formatted_pipeline[:-1]
            res = self.custom_pipeline([formatted_pipeline], self.goal)
            return res
        else:
            next_steps = self.rewards[current_step]['followed_by']
            for next_step, reward in next_steps.items():
                if next_step not in pipeline:
                    self.generate_pipeline(next_step, pipeline.copy(), imputers)


    def grid_search(self, dataset_name='None', trials=1):
        """Exhaustively search all combinations of preprocessing methods as a baseline.

        For each trial, the order of method groups (feature selection, duplicate detection,
        outlier detection) is randomized while imputation is always applied first. Search
        stops early if the 10-minute time limit is reached. Results and timestamps are
        appended to files in the results directory.

        Parameters:
        - dataset_name: Dataset identifier used for output file naming. Default 'None'.
        - trials: Number of independent trials with reshuffled method orderings. Default 1.
        """
        imputers, feature_selectors, duplicate_detectors, outlier_detectors = [], [], [], []

        for method in self.rewards:
            if method == "CR":
                continue
            method_type = self.rewards[method]['type']
            if method_type == 'Imputer':
                imputers.append(method)
            elif method_type == 'Feature_selector':
                feature_selectors.append(method)
            elif method_type == 'Duplicate_detector':
                duplicate_detectors.append(method)
            elif method_type == 'Outlier_detector':
                outlier_detectors.append(method)
        for trial in range(trials):
            random.shuffle(imputers)
            random.shuffle(feature_selectors)
            random.shuffle(duplicate_detectors)
            random.shuffle(outlier_detectors)
            all_methods = [feature_selectors, duplicate_detectors, outlier_detectors]
            random.shuffle(all_methods)
            all_methods.insert(0, imputers)
            start_time = time.time()
            results = []
            timestamps = []
            timeout = False
            best_so_far = 0
            for i in all_methods[0]:
                for j in all_methods[1]:
                    for k in all_methods[2]:
                        for z in all_methods[3]:
                            string = i + " " + j + " " + k + " " + z
                            pipeline = [string]
                            res = self.custom_pipeline(pipeline, self.goal)[0]['quality_metric']
                            best_so_far = max(best_so_far, res)
                            time_dif = time.time() - start_time
                            results.append(best_so_far)
                            timestamps.append(time_dif)
                            if time_dif >= 600:
                                timeout = True
                                print()
                                print(f"Time limit for Grid Search reached in {time_dif / 60} mins")
                                break
                        if timeout:
                            break
                    if timeout:
                        break
                if timeout:
                    break
            else:
                print()
                print(f'Grid Search Completed in {(time.time() - start_time) / 60} mins')
            results.insert(0, "Best So Far")
            results.insert(0, "Grid_Search")
            timestamps.insert(0, "Timestamps")
            timestamps.insert(0, "Grid_Search")

            with open(self.out_dir + '/'+dataset_name+'_timestamps.txt', mode='a') as rr_file:
                print("{}".format(results), file=rr_file)
                print("{}".format(timestamps), file=rr_file)


            

        