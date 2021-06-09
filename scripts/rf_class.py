class TREE_FI():

    def __init__(self, model, seed_num = None, time = True, trace = False, max_display = 999):
        '''
        Instantiates the SHAP_FI class.
        @param seed_num : Random seed for reproducibility.
        @param time: Set time functionality for runtime.
        @param trace : Turn on/off trace messages for debugging.
        @param max_display : Set max display of features.
        @return ICE data (dataframe) with N observations.
        @examples
        SHAP_FI(420, time = True, trace = False, max_display = 10)
        '''
        self.seed_num = seed_num
        self.trace = trace
        self.time = time
        self.max_display = max_display
        self.model = model
        
        if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
            self.fi_type = 'RF'
        elif isinstance(self.model, (GradientBoostingClassifier, GradientBoostingRegressor)):
            self.fi_type = 'GB'
        else:
            raise ValueError('Incorrect model type!')


    def fit(self, X, model):
        '''
        Uses Shapley values to explain any machine learning model.
        @param X : Covariate matrix.
        @param model : Model to interpet.
        '''
        
        self.features = X.columns
        if hasattr(model, 'feature_importances_') == True:
            pass
        else:
            raise ValueError('Non tree-based model!')


    def plot(self, save_path = None):
        '''
        Plot the SHAP values.
        '''
        fi_df = pd.DataFrame({'Feature': self.features,
                              self.fi_type + ' Feature Importance' : self.model.feature_importances_*100}).\
                            sort_values(self.fi_type + ' Feature Importance').\
                            head(self.max_display)
        
        
        
    
        ax = fi_df.plot.barh(x = 'Feature',
                             y = self.fi_type + ' Feature Importance',
                             legend = False,
                             figsize = (10,12))
        ax.set_title('{} Feature Importance'.format(self.fi_type))


        for p in ax.patches:
            width = p.get_width()
            plt.text(0.5+p.get_width(), p.get_y()+0.55*p.get_height(),
                     '{:1.2f}'.format(width),
                     ha='center', va='center', size = 9)


        if save_path is not None:
            fig.savefig(save_path,
                        bbox_inches = 'tight',
                        pad_inches = 1)


    def feature_table(self):
        '''
        Return SHAP Value table.
        '''
    
        fi_df = pd.DataFrame({'Feature': self.features,
                              self.fi_type + ' Feature Importance' : self.model.feature_importances_*100}).\
                            sort_values(self.fi_type + ' Feature Importance').\
                            head(self.max_display)
        return fi_df        