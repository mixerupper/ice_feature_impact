class ICE():
	def __init__(self, model_type, num_per_ob = 30, frac_sample = 0.9, seed_num = None, trace = False):
		'''
		Instantiates the ICE class
		@param model_type : "binary" or "continuous" y-variable
		@param num_per_ob : Number of points to generate per observation.
							Points used in line plots.
		@param frac_sample : Fraction of data set to sample for ICE df.
		@param seed_num : Random seed for reproducibility.
		@param trace : Turn on/off trace messages for debugging
		@return ICE data (dataframe) with N observations.
		@examples
		ICE("binary", num_per_ob = 50, frac_sample = 0.5, seed_num = 420)
		'''
		self.model_type = model_type
		self.num_per_ob = num_per_ob
		self.frac_sample = frac_sample
		self.seed_num = seed_num
		self.trace = trace

		# Initializations to raise exceptions
		self.fit_all = False

		self.ice_dfs = {}


	def fit(self, X, model):
		'''
		Creates all ICE datasets for each feature
		@param X : Covariate matrix
		@param model : Model to interpet
		'''

		self.features = list(X.columns)

		for feature in X:
			try:
				start = datetime.now()
				self.ice_dfs[feature] = self.ice_single_feature(X, model, feature)
				end = datetime.now()
				print(f"Fit {feature} in {(end - start).seconds} seconds")
			except ValueError:
				print(f"Could not fit {feature} because of ValueError")

		self.fit_all = True

		return

	def fit_single_feature(self, X, model, feature):
		'''
		Create single ICE dataset for a feature.
		Used when only some features are of interest.
		@param X : Covariate matrix
		@param model : Model to interpet
		@param feature : Single feature to create ICE dataset for
		'''

		start = datetime.now()

		self.ice_dfs[feature] = self.ice_single_feature(X, model, feature)
		
		end = datetime.now()
		print(f"Fit {feature} in {(end - start).seconds} seconds")



	def ice_single_feature(self, X, model, feature):
		'''
		Create ICE dataset for a single feature. Called by fit.
		@param X : Covariate matrix
		@param model : Model to interpet
		@param feature : Single feature to create ICE dataset for
		'''

		# uniformly sample
		X = self.uniform_sample(X, feature, self.frac_sample)

		feature_min = np.min(X[feature])
		feature_max = np.max(X[feature])
		feature_range = np.linspace(feature_min, feature_max, num = self.num_per_ob)

		df = pd.DataFrame()

		for i in X.index:
			# make temp df for each instance
			temp_df = X.loc[np.repeat(i, self.num_per_ob)]\
				.copy()\
				.reset_index(drop = True)
			temp_df[feature] = feature_range
			temp_df['obs'] = i

			df = df.append(temp_df, ignore_index = True)

		# get predictions
		if self.model_type == "binary":    
		  preds = model.predict_proba(df.drop('obs', axis = 1))[:,1]
		else:
		  preds = model.predict(df.drop('obs', axis = 1))
		
		df['y_pred'] = preds

		return df
		

	def plot_single_feature(self, feature, plot_num = 300):
		'''
		Plots the ICE chart for a single feature.
		Can only be called after fitting for that feature.
		@param feature : Target covariate to plot.
		@param plot_num : Number of lines to plot.
		@examples
		plot_single_feature('Age', plot_num = 500)
		'''
		plot_data = self.ice_dfs[feature]
		
		ob_sample = np.random.choice(plot_data.obs.unique(), 
		                           size = plot_num, replace = False)

		ob_sample = np.append(ob_sample, [-1])

		mean_line = plot_data\
			.groupby(feature)\
			.agg(y_pred = ('y_pred', 'mean'))\
			.reset_index()\
			.assign(obs = -1,
			        mean_line = 1)

		plot_sub_data = plot_data\
			.loc[lambda x:x.obs.isin(ob_sample)]\
			.assign(mean_line = 0)\
			.append(mean_line, ignore_index = True)

		# set fig size
		fig, ax = plt.subplots()

		# plot ICE
		for ob in ob_sample:
			d = plot_sub_data.loc[lambda x:x.obs == ob]
			if max(d.mean_line) == 0:
			    alpha = 0.1
			    color = "black"
			    label = ""
			elif max(d.mean_line) == 1:
			    alpha = 5
			    color = "red"
			    label = "Mean line"
			ax.plot(feature, 'y_pred', label = label, alpha = alpha, data = d, color = color)
		
		ax.set_title('{} ICE Plot'.format(feature), fontsize=18)
		ax.set_xlabel(feature, fontsize=18)
		ax.set_ylabel('Predicted Probability', fontsize=16)
		ax.legend()

		return (fig, ax)


	def plot(self, save_path, plot_num = 300, ncols = 3):
		'''
		Plot all ICE plots in a grid
		'''
		if not self.fit_all:
			raise Exception("Call `fit` method before trying to plot. You can also call `plot_single_feature`.")

		fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (5*ncols,1*num_plots))
		all_features = np.sort(list(self.ice_dfs.keys()))

		for i, feature in enumerate(all_features):
		    plot_data = ice.ice_dfs[feature]
		    ob_sample = np.random.choice(plot_data.obs.unique(), 
		                               size = plot_num, replace = False)

		    ob_sample = np.append(ob_sample, [-1])

		    mean_line = plot_data\
		        .groupby(feature)\
		        .agg(y_pred = ('y_pred', 'mean'))\
		        .reset_index()\
		        .assign(obs = -1,
		                mean_line = 1)

		    plot_sub_data = plot_data\
		        .loc[lambda x:x.obs.isin(ob_sample)]\
		        .assign(mean_line = 0)\
		        .append(mean_line, ignore_index = True)

		    # plot ICE
		    for ob in ob_sample:
		        d = plot_sub_data.loc[lambda x:x.obs == ob]
		        if max(d.mean_line) == 0:
		            alpha = 0.1
		            color = "black"
		            label = ""
		        elif max(d.mean_line) == 1:
		            alpha = 5
		            color = "red"
		            label = "Mean line"
		        axs[int(i/3),i%3].plot(feature, 'y_pred', label = label, alpha = alpha, data = d, color = color)

		    axs[int(i/3),i%3].set_xlabel(feature, fontsize=10)

		handles, labels = axs[0,0].get_legend_handles_labels()
		# fig.subplots_adjust(hspace=.5)
		fig.legend(handles, labels, loc='lower center', borderaxespad = 0.5, borderpad = 0.5)
		plt.tight_layout()

		fig.savefig(save_path, 
		            bbox_inches = 'tight',
		            pad_inches = 1)


	def uniform_sample(self, df, feature, frac_sample):
		'''
		Uniformly sample across quantiles of feature to ensure not to leave out 
		portions of the dist of the feature.
		@param df : Covariate matrix.
		@param feature : Target covariate bin.
		@examples
		uniform_sample(df, 'Age')
		'''
		
		# Determine if categorical or continuous
		num_obs = df.shape[0]
		num_unique_feature_values = len(df[feature].unique())

		if num_unique_feature_values > 10:
			featureIsCategorical = False
		else:
			featureIsCategorical = True

		if self.trace:
			print(f"{feature} is categorical: {featureIsCategorical}")

		# Categorical
		if featureIsCategorical:
			sample_df = df\
				.groupby(feature)\
				.apply(lambda x:x.sample(int(np.ceil(x.shape[0] * frac_sample))))\
				.reset_index(drop = True)
		elif not featureIsCategorical:
			sample_df = df.copy()

			sample_df['quantile'] = pd.qcut(sample_df[feature], q = 10, duplicates = 'drop')

			sample_df = sample_df\
				.groupby('quantile')\
				.apply(lambda x:x.sample(int(np.ceil(x.shape[0] * frac_sample))))\
				.reset_index(drop = True)\
				.drop('quantile', axis = 1)

		if self.trace:
			print(f"Sample df has {sample_df.shape[0]} observations, {sample_df.shape[0]/num_obs}% of the observations in the original df.")

		return sample_df

