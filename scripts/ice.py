
def uniform_sample(df, feature, frac_to_plot, seednum = None):
  '''
  #' Uniformly sample across quantiles of x_j
  #' to ensure not to leave out portions of the
  #' dist'n of x.
  #' @param df : Covariate matrix.
  #' @param feature : Target covariate bin.
  #' @param frac_to_plot : Fraction of data set to plot.
  #' @param seed_num : Random seed for reproducibility.
  #' @return Uniformly sampled dataframe with N * frac_to_plot observations.
  #' @examples
  #' uniform_sample(X, 'Age', .33, 420)
  '''
  # get number of rows to sample
  N = df.shape[0] * frac_to_plot

  # get amount for each quantile (sometimes uneven)
  quantile = [N // 4 + (1 if x < N % 4 else 0)  for x in range (4)]

  # create labels and bins for quantiles
  labels, bins = ['q1', 'q2', 'q3', 'q4'], [0, .25, .5, .75, 1.]

  # create col to get quantiles of x_j to not leave out portions of the dist'n of x
  df['quantile'] = pd.qcut(X[feature], q = bins, labels = labels)

  # uniformly sample quantiles
  out = pd.concat([df[df['quantile'].eq(label)].sample(int(quantile[i])) for i, label in enumerate(labels)]).\
      drop(columns = ['quantile'])

  return out


def ice_plot(plot_data, feature):

  '''
  #' Generates ICE plot
  #' @param plot_data : ICE data to plot
  #' @param feature : Target covariate to plot.
  #' @return ICE plot
  #' @examples
  #' ice_plot(X, 'Age')
  '''

  # set fig size
  plt.figure(figsize=(12,10))

  # set lines to all black
  palette = sns.color_palette(['black'], len(plot_data['Obs'].unique()))

  # plot ICE
  sns.lineplot(data=plot_data, x= feature, y="Predicted Probability", hue='Obs', palette=palette, legend=False)
  plt.title('{} ICE Plot'.format(feature))
  plt.xlabel(feature, fontsize=18)
  plt.ylabel('Predicted Probability', fontsize=16)
  plt.show()


 def ice(X, clf, feature, frac_to_plot = 1, seednum = None):

  '''
  #' Generates ICE data
  #' @param X : Covariate matrix.
  #' @param clf : ML classifier.
  #' @param feature : Target covariate to plot.
  #' @param frac_to_plot : Fraction of data set to plot.
  #' @param seed_num : Random seed for reproducibility.
  #' @return ICE data (dataframe) with N observations.
  #' @examples
  #' ice_plot(X, rf, 'Age', frac_to_plot = .33 , seednum = 420)
  '''

  # initialize dict
  d = {feature: np.array([]), 'Predicted Probability': np.array([]), 'Obs': np.array([])}

  # uniformly sample
  X = uniform_sample(X, feature, frac_to_plot, seednum)
  for index, row in X.iterrows():

    # make temp df for each instance
    temp_df = X.copy()

    # make copy for fixed feature
    feat = X[feature].copy()

    # alter rows of DF except fixed feature
    temp_df.loc[:, :] = temp_df.loc[index, :].values
    temp_df[feature] = feat
    # get predictions
    preds = rf.predict_proba(temp_df)[:,1]
    d[feature] = np.append(feat.array, d[feature])
    d['Predicted Probability'] = np.append(preds, d['Predicted Probability'])
    d['Obs'] = np.append(np.repeat(index, X.shape[0]), d['Obs'])

  df = pd.DataFrame(d)

  return ice_plot(df, feature)
