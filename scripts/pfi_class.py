from sklearn.inspection import permutation_importance

class PFI_FI():

    def __init__(self, seed_num = None, time = True, trace = False, max_display = 999):
        '''
        Instantiates the SHAP_FI class.
        @param seed_num : Random seed for reproducibility.
        @param time: Set time functionality for runtime.
        @param trace : Turn on/off trace messages for debugging.
        @param max_display : Set max display of features.
        @return ICE data (dataframe) with N observations.
        @examples
        PFI_FI(420, time = True, trace = False, max_display = 10)
        '''
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
        
        start = datetime.now()
        self.features = X.columns
        self.result = permutation_importance(model, X, y, n_repeats=10,
                                random_state=42)
        
        end = datetime.now()
        
        if self.time:
            print(f"PFI fits in {(end - start).total_seconds():.2f} seconds")
        
        return
        

    def plot(self, save_path = None):
        '''
        Plot the SHAP values.
        '''
        
        pfi = pd.DataFrame({'Feature' : self.features,
              'PFI' : self.result.importances_mean}).\
        sort_values('PFI').\
        head(self.max_display)
        
        ax = pfi.plot.barh(x = 'Feature',
                     y = 'PFI',
                     legend = False,
                     figsize = (10,12))
        ax.set_title('Permutation Feature Importance')

        for i, v in enumerate(pfi.PFI):
            if v != 0:
                ax.text(v  , i, str(np.round(v,4)), color='black', fontweight='bold', size = 10)


        if save_path is not None:
            fig.savefig(save_path,
                        bbox_inches = 'tight',
                        pad_inches = 1)


    def feature_table(self):
        '''
        Return SHAP Value table.
        '''
    
        pfi_df = pd.DataFrame({'Feature' : self.features,
              'PFI' : self.result.importances_mean}).\
        sort_values('PFI').\
        head(self.max_display)
        
                            
        return pfi_df        