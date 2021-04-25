def data_root(path):
	return '../../data/'+path

def raw_root(path):
	return data_root('raw/' + path)

def processed_root(path):
	return data_root('processed/' + path)

def interim_root(path):
	return data_root('interim/' + path)

def results_root(path):
	return data_root('results/' + path)


# Import ICE class
exec(open("../../scripts/ice_class.py").read())