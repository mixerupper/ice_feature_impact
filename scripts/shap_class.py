class SHAP_FI():
	def __init__(self, model_type, frac_sample = 0.9, seed_num = None, time = True, trace = False):
		'''
		Instantiates the ICE class
		@param model_type : "binary" or "continuous" y-variable
		@param frac_sample : Fraction of data set to sample for ICE df.
		@param seed_num : Random seed for reproducibility.
		@param trace : Turn on/off trace messages for debugging
		@return ICE data (dataframe) with N observations.
		@examples
		ICE("binary", num_per_ob = 50, frac_sample = 0.5, seed_num = 420)
		'''
		self.model_type = model_type
		self.frac_sample = frac_sample
		self.seed_num = seed_num
		self.trace = trace
		self.time = time

		# Initializations to raise exceptions
		self.fit_all = False

		self.fis = {}

	def fit(self, X, model, lice = False):
		'''
		Creates all ICE datasets for each feature
		@param X : Covariate matrix
		@param model : Model to interpet
		@param lice : Linearly spaced feature distribution instead of unique values
		'''
		pass
		# shap.TreeExplainer(rf).shap_values(X)

	def plot(self, save_path = None):
		'''
		Plot the SHAP values in a chart. Save to save_path
		'''

		if save_path is not None:
			fig.savefig(save_path,
		            	bbox_inches = 'tight',
		            	pad_inches = 1)

	def fi_table(self):
		fi_df = pd.DataFrame()

		for feature in ice.ice_fis:
			fi_df = fi_df\
				.append(self.get_feature_impact(feature), ignore_index = True)
		
		for fi in fi_df.drop('Feature', axis = 1):
			fi_df[f'Normalized {fi}'] = fi_df[fi]/fi_df[fi].sum()*100

		fi_df = fi_df.fillna(0)

		return fi_df