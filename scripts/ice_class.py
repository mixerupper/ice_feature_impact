class ICE():
	def __init__(self, model_type, num_per_ob = 30, frac_to_plot = 0.9, seed_num = None, trace = False):
		'''
		Instantiates the ICE class
		@param model_type : "binary" or "continuous" y-variable
		@param num_per_ob : Number of points to generate per observation.
							Points used in line plots.
		@param frac_to_plot : Fraction of data set to plot.
		@param seed_num : Random seed for reproducibility.
		@param trace : Turn on/off trace messages for debugging
		@return ICE data (dataframe) with N observations.
		@examples
		ICE("binary", num_per_ob = 50, frac_to_plot = 0.5, seed_num = 420)
		'''
		self.model_type = model_type
		self.num_per_ob = num_per_ob
		self.frac_to_plot = frac_to_plot
		self.seed_num = seed_num
		self.trace = trace

		self.ice_dfs = {}


	def fit(self, X, model):
		'''
		Creates all ICE datasets for each feature
		@param X : Covariate matrix
		@param model : Model to interpet
		'''

		for feature in X:
			start = datetime.now()
			self.ice_dfs[feature] = self.ice_single_feature(X, model, feature)
			end = datetime.now()
			print(f"Fit {feature} in {(end - start).seconds} seconds")

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
		X = self.uniform_sample(X, feature)

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

			# get predictions
			if self.model_type == "binary":    
			  preds = model.predict_proba(temp_df)[:,1]
			else:
			  preds = model.predict(temp_df)
			temp_df['y_pred'] = preds
			temp_df['obs'] = i

			df = df.append(temp_df, ignore_index = True)

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


	def plot(ncols = 3):
		'''
		Plot all ICE plots in a grid
		'''
		# num_plots = len(self.ice_dfs)
		# rows = np.ceil(num_plots/ncols)

		# fig, axs = plt.subplots(nrows = nrows, ncols = ncols)

		# for f in self.ice_dfs:
		# 	axs[]


		return


	def uniform_sample(self, df, feature):
		'''
		Uniformly sample across quantiles of feature to ensure not to leave out 
		portions of the dist of the feature.
		@param df : Covariate matrix.
		@param feature : Target covariate bin.
		@examples
		uniform_sample(df, 'Age')
		'''
		df = df.copy()

		# get number of rows to sample
		N = df.shape[0] * self.frac_to_plot
		if self.trace:
		    print(f"Sampling {N} observations")

		# get amount for each quantile (sometimes uneven)
		quantile = [N // 4 + (1 if x < N % 4 else 0)  for x in range (4)]
		if self.trace:
		    print(f"quantile {quantile}")

		# create labels and bins for quantiles
		bins, labels = [0, .25, .5, .75, 1.], ['q1', 'q2', 'q3', 'q4'],

		# create col to get quantiles of x_j to not leave out portions of the dist'n of x
		df['quantile'] = pd.qcut(df[feature], q = bins, labels = labels)
		if self.trace:
		    print(df['quantile'][:3])

		# uniformly sample quantiles
		out = pd.concat([df[df['quantile'].eq(label)].sample(int(quantile[i])) 
		               for i, label in enumerate(labels)]).\
		  drop(columns = ['quantile'])

		return out

