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

			temp_df = temp_df.append(X.iloc[i,:]) # append in-sample instance to temp_df for plotting
			temp_df['obs'] = i
			df = df.append(temp_df, ignore_index = True)

		df = df\
			.sort_values(['obs', feature])

		# get predictions
		if self.model_type == "binary":
		  preds = model.predict_proba(df.drop('obs', axis = 1))[:,1]
		else:
		  preds = model.predict(df.drop('obs', axis = 1))

		df['y_pred'] = preds

		# Add on dydx for histogram and feature importance
		df['dy'] = df\
		    .groupby('obs')['y_pred']\
		    .transform(lambda x:x - x.shift(1))

		df['dx'] = df\
		    .groupby('obs')[feature]\
		    .transform(lambda x:x - x.shift(1))

		df['dydx'] = df['dy'] / df['dx']
		df['dydx_abs'] = np.abs(df['dydx'])

		return df


	def ice_plot_single_feature(self, feature, in_sample = True, plot_num = 300):
		'''
		Plots the ICE chart for a single feature.
		Can only be called after fitting for that feature.
		@param feature : Target covariate to plot.
		@param plot_num : Number of lines to plot.
		@param in_sample : Show in-sample data points.
		@examples
		plot_single_feature('Age', plot_num = 500)
		'''
        
		plot_data = self.ice_dfs[feature]

		orig_ob_sample = np.random.choice(plot_data.obs.unique(),
		                           size = plot_num, replace = False)

		ob_sample = np.append(orig_ob_sample, [-1])

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
            
			d = d.sort_values(feature) # sort values by feature for plotting
			ax.plot(feature, 'y_pred', label = label, alpha = alpha, data = d, color = color)
            
		if in_sample == True:
			ax.scatter(plot_sub_data.loc[orig_ob_sample, feature], plot_sub_data.loc[orig_ob_sample, feature],c="green")

		ax.set_title('{} ICE Plot'.format(feature), fontsize=18)
		ax.set_xlabel(feature, fontsize=18)
		ax.set_ylim((0,1))

		if self.model_type == 'binary':
			ax.set_ylabel('Predicted Probability', fontsize=16)
		elif self.model_type == 'continuous':
			ax.set_ylabel('Target', fontsize=16)
		else:
			raise ValueError



		ax.legend()

		return (fig, ax)


	def ice_plot(self, model, save_path = None, plot_num = 300, in_sample = True, ncols = 3):
		'''
		Plot all ICE plots in a grid
		'''
		if not self.fit_all:
			raise Exception("Call `fit` method before trying to plot. You can also call `plot_single_feature`.")

		nrows, num_plots = int(np.ceil(len(self.ice_dfs.keys()) / ncols)), len(self.ice_dfs.keys())
		all_features = np.sort(list(self.ice_dfs.keys()))

		if nrows == 1:
			ncols = num_plots

		fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (5*ncols,1*num_plots))

		if self.trace:
			print(f"Num rows: {nrows}, Num columns: {ncols}, Num plots: {num_plots}")

		for i, feature in enumerate(all_features):
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

		    # plot ICE
		    if self.trace:
		    	print(f"Plotting for {feature}")

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

		        if nrows == 1:
		        	d = d.sort_values(feature)
		        	axs[i].plot(feature, 'y_pred', label = label, alpha = alpha, 
		        		data = d, color = color)
		        else:
		        	d = d.sort_values(feature)
		        	axs[int(i/ncols),i%ncols].plot(feature, 'y_pred', label = label, alpha = alpha, 
		        		data = d, color = color)
             
            
		    if in_sample == True:
		        if nrows == 1:
		        	for item in ob_sample[:-1]:
		        		axs[i].scatter(np.array(plot_sub_data[plot_sub_data.obs==item][feature])[-1], model.predict_proba(np.array(plot_sub_data[plot_sub_data.obs==item].iloc[-1,:-7]).reshape(1,-1))[:,1],c="green")
		        else:
		        	for item in ob_sample[:-1]:
		        		axs[int(i/ncols),i%ncols].scatter(np.array(plot_sub_data[plot_sub_data.obs==item][feature])[-1], model.predict_proba(np.array(plot_sub_data[plot_sub_data.obs==item].iloc[-1,:-7]).reshape(1,-1))[:,1],c="green")
                
                
		    if nrows == 1:
		    	axs[i].set_xlabel(feature, fontsize=10)
		    else:
		    	axs[int(i/ncols),i%ncols].set_xlabel(feature, fontsize=10)

		if nrows == 1:
			handles, labels = axs[0].get_legend_handles_labels()
		else:
			handles, labels = axs[0,0].get_legend_handles_labels()
		# fig.subplots_adjust(hspace=.5)
		fig.legend(handles, labels, loc='lower center', borderaxespad = 0.5, borderpad = 0.5)
		plt.tight_layout()

		if save_path is not None:
			fig.savefig(save_path,
		            	bbox_inches = 'tight',
		            	pad_inches = 1)

	def feature_importance_hist(self, save_path = None, remove_zeros = True, ncols = 3, plot_num = 300):
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

	def feature_importance_table(self):
		if not self.fit_all:
			raise Exception("Call `fit` method before trying to plot.")

		all_features = np.sort(list(self.ice_dfs.keys()))
		fi_mean = []
		fi_abs_mean = []
		fi_sd = []
		fi_mean_normalized = []
		fi_abs_mean_normalized = []

		for i, feature in enumerate(all_features):
			df = self.ice_dfs[feature]
			fi_mean.append(np.mean(df.dydx))
			fi_abs_mean.append(np.mean(df.dydx_abs))
			fi_sd.append(np.std(df.dydx))
			fi_mean_normalized.append(np.mean(df.dydx) * np.std(df[feature]))
			fi_abs_mean_normalized.append(np.mean(df.dydx_abs) * np.std(df[feature]))

		fi_df = pd.DataFrame({
			'Feature':all_features,
			'Mean':fi_mean,
			'Mean Abs':fi_abs_mean,
			'St. Dev.':fi_sd,
			'Normalized Mean':fi_mean_normalized,
			'Normalized Absolute Mean':fi_abs_mean_normalized
		})\

		fi_var = 'Normalized Absolute Mean'

		fi_df['Feature Importance'] = fi_df[fi_var]/fi_df[fi_var].sum()*100

		fi_df = fi_df.fillna(0)

		return fi_df



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


