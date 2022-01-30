PYC=python -m jupyter nbconvert --to script

part030_training_the_mlp.py: part030_training_the_mlp.ipynb part020_normalize_features.py
	$(PYC) $<

part020_normalize_features.py: part020_normalize_features.ipynb part010_splitting_the_data.py
	$(PYC) $<

part010_splitting_the_data.py: part010_splitting_the_data.ipynb
	$(PYC) $<
