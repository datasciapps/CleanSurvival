import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score

class RSF:
    def __init__(self, dataset, target_goal, time_column, config=None, verbose=False, metric="c-index"):
        self.dataset = dataset
        self.event_column = target_goal
        self.time_column = time_column
        self.config = config
        self.verbose = verbose
        self.metric = metric

    def timepoints_grid(self, y_train, y_test, n_pts=50):
        # Timepoint generation for integrated brier score
        train_times, test_times = y_train["time"], y_test["time"]
        t_min = max(train_times.min(), test_times.min())
        t_max = min(train_times.max(), test_times.max()) * 0.95
        times = np.linspace(t_min, t_max, n_pts)
        return times
    
    def fit_rsf_model(self, test_size=0.2, random_state=42):

        X = self.dataset

        y = np.array(list(zip(X[self.event_column].astype(bool), X[self.time_column])),
                          dtype=[('event', '?'), ('time', '<f8')])
        
        X.drop([self.time_column, self.event_column], axis=1, inplace=True)

        
        print("Building Random Survival Forest model.....")

        if len(X.columns) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            rsf = RandomSurvivalForest(n_estimators=100, random_state=random_state, n_jobs=-1)

            print(f"\n\nIN RSF X_train is ------------> \n\n{X}\n\n")

            rsf.fit(X_train, y_train)

            survival_probabilities = rsf.predict_survival_function(X_test)
            
            if self.metric == 'ibs':
                times = self.timepoints_grid(y_train, y_test)
                estimate = np.asarray([fn(times) for fn in survival_probabilities])
                ibs = integrated_brier_score(y_train, y_test, estimate, times)
                c_index = 1.0 - ibs
                print(f"Building Random Survival Forest is done: \n IBS metric (inverted for Q-learning):  {c_index:.4f}")
            else:
                c_index = rsf.score(X_test, y_test)
                print(f"Building Random Survival Forest is done: \n C-Index score:  {c_index:.4f}")
        else:
            survival_probabilities = 0
            c_index = 0

        return survival_probabilities, c_index
