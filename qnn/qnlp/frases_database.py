import pandas as pd
from typing import List


def extract_words(excel_path: str):

	"""Gets the unique words of a given database of phrases
	that is in form of a .xlsx file
	Return: List[str]"""

	data_frame = pd.read_excel(excel_path)
	vocabulary: List[str] = []
	for i in data_frame.values:
		for u in i[0].split():
			if u not in vocabulary:
				vocabulary.append(u)
	return vocabulary, data_frame
