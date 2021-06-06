def build_comparator_table(X, model, fi_classes):
	'''
	Purpose: Build a table to compare our feature importance/impact metrics
	@X: Dataset with features as column names
	@model: Model we're analyzing
	@fi_classes: List of feature importance/impact class objects
		Each class should have standardized functions to fit and extract
		the feature importance/impact table
	'''

	output_table = pd.DataFrame({'feature':X.columns})

	for fi in fi_classes:
		# fi.fit(X, model)
		# TODO: Merge each fi_class's feature importance table into the output table
		# output_table = output_table\
		# 	.merge(fi.fi_table, how = "left", on = "feature")
		# TODO: Plot it
		pass

	return
	# return output_table