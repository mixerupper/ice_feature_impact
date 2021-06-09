class Comparator():
    def __init__(trace = False):
        self.trace = trace

    def fit(X, model, fi_classes):
        for fi in fi_classes:
            fi.fit(X, model)

    def build_raw_table:

    def build_normalized_table:

def build_comparator_table(X, model, fi_classes):
    '''
    Purpose: Build a table to compare our feature importance/impact metrics
    @X: Dataset with features as column names
    @model: Model we're analyzing
    @ice: ICE object
    @fi_classes: List of feature importance/impact class objects
        Each class should have standardized functions to fit and extract
        the feature importance/impact table
    '''


    output_table = None

    for fi in fi_classes:
        fi.fit(X, model)

        if output_table is None:
            output_table = fi.feature_table()
        else:
            output_table = output_table.\
                merge(fi.feature_table(), how = "left", on = "Feature")

    return output_table
