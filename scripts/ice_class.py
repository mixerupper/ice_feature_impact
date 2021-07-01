from sklearn.linear_model import LogisticRegression

class ICE():
	def __init__(self, model_type, frac_sample = 1, seed_num = None, time = False, trace = False):
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

		self.ice_dfs = {}
		self.ice_fis = {}


	def fit(self, X, model, lice = False):
		'''
		Creates all ICE datasets for each feature
		@param X : Covariate matrix
		@param model : Model to interpet
		@param lice : Linearly spaced feature distribution instead of unique values
		'''

		self.features = list(X.columns)
		self.data = X.copy()

		if self.model_type == "binary":
		  self.data['y_pred'] = model.predict_proba(X)[:,1]
		else:
		  self.data['y_pred'] = model.predict(X)

		for feature in X:
			try:
				start = datetime.now()
				self.ice_dfs[feature], self.ice_fis[feature] = self.ice_fit_helper(X, model, feature, lice)
				end = datetime.now()
				if self.time:
					print(f"Fit {feature} in {(end - start).total_seconds():.2f} seconds")
			except ValueError:
				print(f"Could not fit {feature} because of ValueError")

		self.fit_all = True

		return

	def fit_single_feature(self, X, model, feature, lice = False):
		'''
		Create single ICE dataset for a feature.
		Used when only some features are of interest.
		@param X : Covariate matrix
		@param model : Model to interpet
		@param feature : Single feature to create ICE dataset for
		'''

		start = datetime.now()

		self.ice_dfs[feature], self.ice_fis[feature] = self.ice_fit_helper(X, model, feature, lice)
		self.data = X.copy()

		if self.model_type == "binary":
		  self.data['y_pred'] = model.predict_proba(X)[:,1]
		else:
		  self.data['y_pred'] = model.predict(X)

		end = datetime.now()
		if self.time:
			print(f"Fit {feature} in {(end - start).total_seconds():.2f} seconds")



	def ice_fit_helper(self, X, model, feature, lice = False,
					   min_obs_per_feature = 10, likelihood_decay = 0.75):
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
		
		if lice:
			feature_range = np.linspace(feature_min, feature_max, num = self.num_per_ob)
		else:
			feature_range = np.sort(np.unique(X[feature]))

		df = X.loc[np.repeat(X.index, len(feature_range))]
		df['orig_'+feature] = df[feature]
		df['obs'] = df.index
		df[feature] = np.tile(feature_range, len(X.index))

		# get predictions
		if self.model_type == "binary":
		  preds = model.predict_proba(
		  	df.drop(['obs', 'orig_'+feature], axis = 1))[:,1]
		else:
		  preds = model.predict(df.drop(['obs', 'orig_'+feature], axis = 1))

		df['y_pred'] = preds

		df['y_pred_centered'] = df\
			.groupby('obs')['y_pred']\
			.transform(lambda x:(x - x.shift(1)).cumsum())\
			.fillna(0)

		# Add on dydx for histogram and feature importance
		# TODO: Deal with case where these names collide with existing feature names.
		df['feature_distance'] = np.abs(df[feature] - df['orig_'+feature])
		df['original_point'] = (df['feature_distance'] == 0)*1
		feature_std = np.std(X[feature])
		
		# Add likelihood on phantom/real obs based on logistic regression
		# logr = LogisticRegression(class_weight = 'balanced')
		# logr.fit(df[[feature]], df['original_point'])
		
		if feature_std != 0:
			df['likelihood'] = likelihood_decay ** (df['feature_distance']/feature_std)
		else:
			df['likelihood'] = 1

		# Add feature impact
		df['dy'] = df\
		    .groupby('obs')['y_pred']\
		    .transform(lambda x:x - x.shift(1))

		df['dx'] = df\
		    .groupby('obs')[feature]\
		    .transform(lambda x:x - x.shift(1))

		df['dydx'] = df['dy'] / df['dx']

		# Account for NA of very first unique value that doesn't have a lag
		df['dydx'] = df\
			.groupby('obs')['dydx']\
			.transform(lambda x:np.where(x.isna(), x.shift(-1), x))

		df['dydx_abs'] = np.abs(df['dydx'])

		df = df.loc[lambda x:~x.dydx_abs.isna()]

		if df.shape[0] == 0:
			fi_dict = {'Feature':feature,
			'ICE FI':0,
			'ICE In-Dist FI':0}
		else:
			# Calculate feature impact
			# Normalize a feature by subtracting mean and dividing by SD
			# Therefore, we normalize these FIs by multiplying by SD

			temp_df = df.loc[lambda x:~x.dydx_abs.isna()]

			# Feature impact/In-Dist Feature impact
			fi_raw = np.mean(temp_df['dydx_abs'])
			fi_in_dist_raw = np.sum(temp_df['dydx_abs'] * temp_df['likelihood'])/np.sum(temp_df['likelihood'])
			fi_standard = fi_raw * feature_std
			fi_in_dist_standard = fi_in_dist_raw * feature_std

			# Heterogeneity
			fi_het = temp_df\
				.groupby(feature)\
				.agg(dydx_std = ('dydx', 'std'))\
				.reset_index(drop = True)\
				.loc[:,'dydx_std']\
				.mean()

			fi_het = fi_het * feature_std

			# Non-linearity
			fi_nl = temp_df\
				.groupby('obs')\
				.agg(dydx_std = ('dydx', 'std'))\
				.reset_index(drop = True)\
				.loc[:,'dydx_std']\
				.mean()

			fi_nl = fi_nl * feature_std

			
			fi_dict = {'Feature':feature,
				'ICE FI':fi_standard,
				'ICE In-Dist FI':fi_in_dist_standard,
				'ICE Heterogeneity':fi_het,
				'ICE Non-linearity':fi_nl}

		# TODO: drop every column except necessary ones for plotting to save space

		return df, fi_dict

	def ice_plot_single_feature(self, feature, save_path = None,
		plot_num = 200, close_multiple = 0.5, mode = "ice"):
		'''
		Plots the ICE chart for a single feature.
		Can only be called after fitting for that feature.
		@param feature : Target covariate to plot.
		@param plot_num : Number of lines to plot.
		@param close_multiple : Mark parts of the line within close_multiple
								times standard deviation of feature as "close"
								with a solid line
		@param mode: ice|d-ice|c-ice
		@examples
		plot_single_feature('Age', plot_num = 500)
		'''
		start = datetime.now()
		
		plot_data = self.ice_dfs[feature]

		unique_features = plot_data[feature].unique()
		if len(unique_features) > 10:
			feature_continuous = True
		else:
			feature_continuous = False

		y_var = np.select(
			[mode == "ice",
			 mode == "d-ice",
			 mode == "c-ice"],
			["y_pred", "dydx", "y_pred_centered"]).item()

		unique_obs = plot_data.obs.unique()
		ob_sample = np.random.choice(unique_obs, 
			size = min(len(unique_obs), plot_num), replace = False)

		mean_line = plot_data\
			.groupby(feature)\
			.agg(y_pred = (y_var, 'mean'))\
			.reset_index()\
			.rename({'y_pred':y_var}, axis = 1)\
			.assign(obs = -1,
			        mean_line = 1)

		plot_sub_data = plot_data\
			.loc[lambda x:x.obs.isin(ob_sample)]\
			.assign(mean_line = 0)\
			.append(mean_line, ignore_index = True)

		# set fig size
		fig, ax = plt.subplots()

		end = datetime.now()
		if self.time:
			print(f"Preprocessed data in {(end - start).total_seconds():.2f} seconds")

		# plot ICE
		start = datetime.now()
		self.ice_plot_helper(plot_data = plot_sub_data, ax = ax, 
			feature = feature, y_var = y_var,plot_close = feature_continuous)
		
		handles, labels = ax.get_legend_handles_labels()

		unique_labels, i = np.unique(labels, return_index = True)
		unique_handles = np.array(handles)[i]

		ax.legend(unique_handles, unique_labels, 
			 	  markerscale = 0.6, fontsize = 'x-small')

		end = datetime.now()
		if self.time:
			print(f"Plotted in {(end - start).total_seconds():.2f} seconds")

		plt.tight_layout()

		if save_path is not None:
			fig.savefig(save_path,
		            	bbox_inches = 'tight',
		            	pad_inches = 0.1)

		return
		# return (ax, fig)

	def ice_plot(self, save_path = None, 
		plot_num = 200, ncols = 3, mode = "ice"):
		'''
		Plot all ICE plots in a grid
		'''
		if not self.fit_all:
			raise Exception("Call `fit` method before trying to plot. You can also call `plot_single_feature`.")

		nrows, num_plots = int(np.ceil(len(self.ice_dfs.keys()) / ncols)), len(self.ice_dfs.keys())
		all_features = np.sort(list(self.ice_dfs.keys()))

		y_var = np.select(
			[mode == "ice",
			 mode == "d-ice",
			 mode == "c-ice"],
			["y_pred", "dydx", "y_pred_centered"]).item()

		if nrows == 1:
			ncols = num_plots

		fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (5*ncols,1*num_plots))

		if self.trace:
			print(f"Num rows: {nrows}, Num columns: {ncols}, Num plots: {num_plots}")

		for i, feature in enumerate(all_features):
		    plot_data = self.ice_dfs[feature]
		    unique_features = plot_data[feature].unique()
		    if len(unique_features) >= 10:
		    	feature_continuous = True
		    else:
		    	feature_continuous = False

		    unique_obs = plot_data.obs.unique()
		    ob_sample = np.random.choice(plot_data.obs.unique(),
		    	size = min(len(unique_obs), plot_num), replace = False)

		    mean_line = plot_data\
				.groupby(feature)\
				.agg(y_pred = (y_var, 'mean'))\
				.reset_index()\
				.rename({'y_pred':y_var}, axis = 1)\
				.assign(obs = -1,
				        mean_line = 1)

		    plot_sub_data = plot_data\
				.loc[lambda x:x.obs.isin(ob_sample)]\
				.assign(mean_line = 0)\
				.append(mean_line, ignore_index = True)

		    # plot ICE
		    if self.trace:
		    	print(f"Plotting for {feature}")

		    if nrows == 1:
		    	self.ice_plot_helper(plot_data = plot_sub_data,
		    		ax = axs[i], feature = feature, 
		    		y_var = y_var,
		    		plot_close = feature_continuous)
		    else:
	        	self.ice_plot_helper(plot_data = plot_sub_data,
		    		ax = axs[int(i/ncols),i%ncols], feature = feature,
		    		y_var = y_var,
		    		plot_close = feature_continuous)

		if nrows == 1:
			handles, labels = axs[0].get_legend_handles_labels()
		else:
			handles, labels = axs[0,0].get_legend_handles_labels()

		unique_labels, i = np.unique(labels, return_index = True)
		unique_handles = np.array(handles)[i]

		# fig.subplots_adjust(hspace=.5)
		fig.legend(unique_handles, unique_labels, 
			loc='lower center', borderaxespad = 0.5, borderpad = 0.5)
		plt.tight_layout()

		if save_path is not None:
			fig.savefig(save_path,
		            	bbox_inches = 'tight',
		            	pad_inches = 1)

	def ice_plot_helper(self, plot_data, ax, feature, y_var,
		plot_mean = True, plot_points = True, plot_close = True,
		close_multiple = 0.5, axis_font_size = 10):
		'''
		Given the 'obs' column in @plot_data, plot the ICE plot onto @ax.
		@param plot_data: Dataset to plot with 'obs', @feature, 
			'feature_distance,' and 'y_pred' columns
		@param ax: Plot axis object
		@param feature: Feature to make ICE plot of
		@param plot_mean: whether to plot the mean line
		@param plot_points: Whether to plot a scatterplot of original data
		@param close_multiple: Multiple of standard deviation to be "close"
			to original data point
		@param axis_font_size: Font size of x- and y-labels
		'''

		unique_obs = plot_data.obs.unique()
		unique_obs = unique_obs[unique_obs != -1]
		close_radius = close_multiple*np.std(plot_data[feature])

		# Plot observation lines
		for ob in unique_obs:
			d = plot_data.loc[lambda x:x.obs == ob]
			
			if plot_close:
				d_close = d.loc[lambda x:x.feature_distance <= close_radius]
				ax.plot(feature, y_var, 
					label = "Full range", 
					alpha = 0.3, data = d, color = "grey", ls = "--")
				ax.plot(feature, y_var, 
					label = fr'Close: $\pm {close_multiple} \sigma$', 
					alpha = 0.3, data = d_close, color = "black", ls = "-")
			else:
				ax.plot(feature, y_var, 
					label = fr'Close: $\pm {close_multiple} \sigma$', 
					alpha = 0.3, data = d, color = "black", ls = "-")

		# Plot mean line
		if plot_mean:
			d = plot_data.loc[lambda x:x.obs == -1]
			ax.plot(feature, y_var, label = "Mean line", alpha = 5, 
				data = d, color = "gold", ls = "-")

		# Plot scatterplot of points
		if plot_points:
			point_data = plot_data\
				.loc[lambda x:x.feature_distance == 0]\

			ax.scatter(point_data[feature], 
				   point_data[y_var], 
				   color = 'green', 
				   alpha = 0.5,
				   label = "Original data")

		ax.set_xlabel(feature, fontsize=axis_font_size)

		if self.model_type == 'binary':
			ax.set_ylabel('Predicted Probability', fontsize=axis_font_size)
		elif self.model_type == 'continuous':
			ax.set_ylabel('Target', fontsize=axis_font_size)
		else:
			raise ValueError


		return ax

	def feature_hist(self, save_path = None, remove_zeros = True, ncols = 3, plot_num = 300):
		'''
		Plot all feature importance histograms in a grid
		'''
		if not self.fit_all:
			raise Exception("Call `fit` method before trying to plot.")

		nrows, num_plots = int(np.ceil(len(self.ice_dfs.keys())/ ncols)), len(self.ice_dfs.keys())
		all_features = np.sort(list(self.ice_dfs.keys()))

		if nrows == 1:
			ncols = num_plots

		fig, axs = plt.subplots(nrows = nrows, ncols = ncols, 
			figsize = (5*ncols,1*num_plots), sharey = True)

		for i, feature in enumerate(all_features):
		    plot_data = self.ice_dfs[feature]\
		    	.loc[:,['dydx']]\
		    	.dropna(how = 'any')

		    if remove_zeros:
		    	plot_data = plot_data\
		    		.loc[lambda x:x.dydx != 0]
		    if nrows == 1:
		    	axs[i].hist(plot_data['dydx'])
		    	axs[i].set_xlabel(feature, fontsize=10)
		    else:
			    axs[int(i/3),i%3].hist(plot_data['dydx'])
			    axs[int(i/3),i%3].set_xlabel(feature, fontsize=10)

		# fig.subplots_adjust(hspace=.5)
		plt.tight_layout()

		if save_path is not None:
			fig.savefig(save_path,
					 	bbox_inches = 'tight',
					 	pad_inches = 1)

	def feature_table(self):
		fi_df = pd.DataFrame()

		for feature in ice.ice_fis:
			fi_df = fi_df\
				.append(self.get_feature_impact(feature), ignore_index = True)

		fi_df = fi_df.fillna(0)

		return fi_df


	def get_feature_impact(self, feature):
		return self.ice_fis[feature]

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