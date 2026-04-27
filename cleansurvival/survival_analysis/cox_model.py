import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

class CoxRegressor:
    def __init__(self, dataset, target_goal, time_column, config=None, verbose=False, metric="c-index"):
        self.dataset = dataset
        self.event_column = target_goal
        self.time_column = time_column
        self.config = config
        self.verbose = verbose
        self.metric = metric
        self.model = None

    def timepoints_grid(self, y_train, y_test, n_pts=50):
        train_times, test_times = y_train["time"], y_test["time"]
        t_min = max(train_times.min(), test_times.min())
        t_max = min(train_times.max(), test_times.max()) * 0.95
        times = np.linspace(t_min, t_max, n_pts)
        return times

    def updated_fit(self):

        if self.dataset is None:
            return 0

        self.model = CoxPHFitter(penalizer=0.1)
                
        x = self.dataset
        x[self.event_column] = x[self.event_column].astype(bool)

        y = x[[self.time_column, self.event_column]]

        print("Building Cox proportional-hazards model.....")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        c_index = 0.0
        try:
            self.model.fit(x_train, duration_col=self.time_column, event_col=self.event_column)
            
            if self.metric == 'ibs':
                # Convert y variables for sksurv
                y_train_surv = Surv.from_arrays(event=y_train[self.event_column].values, time=y_train[self.time_column].values)
                y_test_surv = Surv.from_arrays(event=y_test[self.event_column].values, time=y_test[self.time_column].values)
                
                # Get predicted survival probabilities
                surv_dfs = self.model.predict_survival_function(x_test)
                
                # Align times properly for Integrated Brier Score
                times = self.timepoints_grid(y_train_surv, y_test_surv)
                
                # Interp curves into specific times to shape (n_samples, n_times)
                from scipy.interpolate import interp1d
                estimate_matrix = np.zeros((x_test.shape[0], len(times)))
                for idx, col in enumerate(surv_dfs.columns):
                    f = interp1d(surv_dfs.index.values, surv_dfs[col].values, kind='previous', bounds_error=False, fill_value=(1.0, surv_dfs[col].values[-1]))
                    estimate_matrix[idx, :] = f(times)
                    
                ibs = integrated_brier_score(y_train_surv, y_test_surv, estimate_matrix, times)
                c_index = 1.0 - ibs
                print(f" Buidling Cox proportional-hazards model is done\n IBS metric (inverted for Q-learning): {c_index:.4f}")
            else:
                c_index = concordance_index(y_test[self.time_column], -self.model.predict_partial_hazard(x_test))
                print(f" Buidling Cox proportional-hazards model is done\n C-Index score: {c_index:.4f}")
        except ValueError:
            print("Problem occured while fitting Cox Model")

        
        return c_index

    
            