import shap

class SHAP_FI():

    def __init__(self, model_type, seed_num = None, time = False, trace = False, max_display = 999):
        '''
        Instantiates the SHAP_FI class.
        @param model_type: Determine which version of SHAP to use
        @param seed_num : Random seed for reproducibility.
        @param time: Set time functionality for runtime.
        @param trace : Turn on/off trace messages for debugging.
        @param max_display : Set max display of features.
        @return ICE data (dataframe) with N observations.
        @examples
        SHAP_FI(420, time = True, trace = False, max_display = 10)
        '''
        self.model_type = model_type
        self.seed_num = seed_num
        self.trace = trace
        self.time = time
        self.max_display = max_display

    def fit(self, X, model):
        '''
        Uses Shapley values to explain any machine learning model.
        @param X : Covariate matrix.
        @param model : Model to interpet.
        '''
        self.features = X.columns

        start = datetime.now()

        if self.model_type == "linear":
            shap_values = shap.explainers\
                .Linear(model, X)\
                .shap_values(X)
        elif self.model_type == "tree":
            shap_values = shap.explainers\
                .Tree(model)\
                .shap_values(X)
        elif self.model_type == "neural-network":
            pass
        else:
            raise("Unrecognized model type.")

        end = datetime.now()

        if type(shap_values) == "list":
            shap_values = shap_values[1]

        self.shap_values = shap_values.mean(axis = 0)

        if self.time:
            print(f"SHAP fits in {(end - start).total_seconds():.2f} seconds")

        return

    def plot(self, save_path = None):
        '''
        Plot the SHAP values.
        '''
        if self.shap is None:
            raise("Fit shap first.")

        fig = shap.summary_plot(self.shap_values, self.features, 
            plot_type = "bar", max_display = self.max_display)


        if save_path is not None:
            fig.savefig(save_path,
                        bbox_inches = 'tight',
                        pad_inches = 1)


    def feature_table(self):
        '''
        Return SHAP Value table.
        '''

        fi_df = pd.DataFrame({'Feature':self.features,
                              'Shapley Value': self.shap_values}).\
                sort_values('Shapley Value', ascending = False).\
                reset_index().\
                drop(['index'], axis=1)

        return fi_df  
