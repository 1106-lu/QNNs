import pandas as pd
from typing import List


def extract_words(csv_path: str):
	""" Extracts info about the database
	Gets the unique words that forms the vocabulary
	and creates the pd.DataFrame from all the data

	Args:
		csv_path: the path of the database (csv file)

	Returns:
		vocabulary: the unique word used in the phrases
		data_frame: the csv file as a pd.DataFrame
	"""
	data_frame = pd.read_csv(csv_path, sep=',')  # reads the csv file as a DataFrame
	vocabulary: List[str] = []
	# goes thought all the words and appends them if they aren't  in the vocabulary
	for i in data_frame.values:
		for u in i[0].split():
			if u not in vocabulary:
				vocabulary.append(u)
	return vocabulary, data_frame
