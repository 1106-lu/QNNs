import pandas as pd
from typing import List


def extract_words(excel_path: str):
	"""Gets the unique words of a given database of phrases
    that is in form of a .xlsx file
    Return: List[str]"""

	data_frame = pd.read_csv(excel_path, sep=',')
	vocabulary: List[str] = []
	for i in data_frame.values:
		for u in i[0].split():
			if u not in vocabulary:
				vocabulary.append(u)
	return vocabulary, data_frame


def generate_data(text):
	words = text.split()
	dict_word = {}
	a = 0
	for k in words:
		dict_word.setdefault(k, []).append(a)
		a = a + 1
	print(dict_word)
