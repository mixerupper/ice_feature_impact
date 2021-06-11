class Comparator():
    def __init__(self, trace = False):
        self.trace = trace

    def fit(self, X, model, fi_classes):
        '''
        Purpose: Build a table to compare our feature importance/impact metrics
        @X: Dataset with features as column names
        @model: Model we're analyzing
        @fi_classes: List of feature importance/impact class objects
            Each class should have standardized functions to fit and extract
            the feature importance/impact table
        '''
        for fi in fi_classes:
            if self.trace:
                print(f"Fitting for {fi}")
            fi.fit(X, model)

        self.fi_classes = fi_classes

    def build_raw_table(self):
        output_table = None

        for fi in self.fi_classes:
            if output_table is None:
                output_table = fi.feature_table()
            else:
                output_table = output_table.\
                    merge(fi.feature_table(), how = "left", on = "Feature")

        return output_table

    def build_normalized_table(self):
        output_table = self.build_raw_table()

        for i in output_table.columns[1:]:
            output_table[i] = output_table[i]/np.sum(np.abs(output_table[i])) * 100

        return output_table
