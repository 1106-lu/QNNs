import cirq
import numpy as np
import pandas as pd
import sympy

from qnn.qnlp.phrases_database import extract_words


class CircuitsWords:
	def __init__(self, data: str, num_qubits: int, num_phrases: int):
		""""
		Creates the gates of the words present in the input phrases
		:parameter data: path to the excel file with the phrases"""

		self.params_used = []
		self.results = []
		self.circuit_list = []
		self.words_used = []
		self.data = data
		self.theta = sympy.symbols("theta:1000")
		self.num_phrases = num_phrases
		self.num_qubits = num_qubits

		self.voc, self.df = extract_words(self.data)
		circuits, q, self.gates = [], [], []

		for i in range(self.num_qubits):
			q.append(cirq.GridQubit(i, 0))

		a = 0
		for k in range(len(self.voc)):
			gates_words = []
			for j in range(self.num_qubits):
				gates_words.append(cirq.rx(self.theta[a])(q[j]))
				a += 1
			self.gates.append(gates_words)
		self.dic_gates = {self.voc[e]: self.gates[e] for e in range(len(self.voc))}

		for i in self.df.transpose().values[3][:self.num_phrases]:
			for k in i.split():
				if k == 'nulo':
					break
				self.words_used.append(k)
		self.words_used = list(set(self.words_used))

		for i in self.words_used:
			index_word = 0
			for k in self.voc:
				if k == i:
					break
				index_word += 1
			stop = index_word * self.num_qubits
			for j in list(np.arange(stop, stop + num_qubits)):
				self.params_used.append(j)

	def __repr__(self):
		for i in self.circuit_list:
			yield i

	def create(self):
		global bitstring
		data_frame = pd.read_csv(self.data, sep=',')
		e = 0
		for i in data_frame.transpose().values[3][:self.num_phrases]:
			if i != 'nulo':
				last = data_frame.transpose().values[4][e]
				o = 0
				for j in data_frame.transpose().values[0]:

					if j == last:
						bitstring = str(data_frame.transpose().values[2][o])
					else:
						bitstring = None
					o += 1
				c = cirq.Circuit()
				for k in i.split():
					a = 0
					for f in self.voc:
						if f == k:
							c.append(self.gates[a])
						a += 1
				e += 1
				self.circuit_list.append([i, last, bitstring, c])
		return self.circuit_list

	def sample_run_global(self, theta_sample, repetitions):
		circuits = []
		for i in self.circuit_list:
			circuits.append(i[3])
		a = 1
		results = []
		for u in circuits:
			if not u.has_measurements():
				for i in u.all_qubits():
					u.append(cirq.measure(i))
					a = a + 1

		resolver = cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in range(len(theta_sample))})
		for u in circuits:
			results.append(cirq.Simulator().run(program=u, param_resolver=resolver, repetitions=repetitions))
		return results
