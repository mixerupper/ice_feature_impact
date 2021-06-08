def build_comparator_table(X, model, ice_object, fi_classes):
    '''
    Purpose: Build a table to compare our feature importance/impact metrics
    @X: Dataset with features as column names
    @model: Model we're analyzing
    @ice: ICE object
    @fi_classes: List of feature importance/impact class objects
        Each class should have standardized functions to fit and extract
        the feature importance/impact table
    '''


    output_table = ice_object.feature_impact_table()

    for fi in fi_classes:
        fi.fit(X, model)

        output_table = output_table.\
                merge(fi.fi_table(X), how = "left", on = "Feature")

    return output_table
