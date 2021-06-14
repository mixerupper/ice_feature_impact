import numpy as np
from datetime import datetime

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
exec(open("../../scripts/shap_class.py").read())
exec(open("../../scripts/native_class.py").read())
exec(open("../../scripts/pfi_class.py").read())
exec(open("../../scripts/fi_comparators.py").read())