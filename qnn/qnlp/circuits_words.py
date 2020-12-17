import cirq
import numpy as np
import pandas as pd
import sympy

from qnn.qnlp.phrases_database import extract_words


class CircuitsWords:
	""" A class used to represent the circuits and some information associated to them
	"""

	def __init__(self, data: str, num_qubits: int, num_phrases: int):
		""" Initializes the class and creates the gates that represent each one of the words in the vocabulary and
		evaluates the parameters that use each gate (these parameters are determined by the number of qubits)
		Args:
			data: the path of the csv file used as database
			num_qubits: the number of qubits used in the optimization
			num_phrases: the number of phrases used in the optimization
		"""

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

		# creates the qubits on which the circuits are created
		for i in range(self.num_qubits):
			q.append(cirq.GridQubit(i, 0))

		a = 0
		# goes thought the vocabulary and creates a parameterized gate of each word
		# each gate is applied on all the qubits
		for k in range(len(self.voc)):
			gates_words = []
			for j in range(self.num_qubits):
				gates_words.append(cirq.rx(self.theta[a])(q[j]))  # the X gate is parameterized but with sympy.symbols
				a += 1
			self.gates.append(gates_words)
		self.dic_gates = {self.voc[e]: self.gates[e] for e in range(len(self.voc))}

		# specifies the words (e.i. the gates) used on the circuits so we can specify the parameters used
		for i in self.df.transpose().values[3][:self.num_phrases]:
			for k in i.split():
				if k == 'nulo':
					break
				self.words_used.append(k)
		self.words_used = list(set(self.words_used))

		# specifies the parameters used on the gates
		# that way we only update this parameters on the optimization
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
		"""prints the circuits one by one"""
		for i in self.circuit_list:
			yield i

	def create(self):
		""" Creates the cirq.Circuit that are optimized. Each phrase has an equivalent circuit formed by the gates that
		are equivalent to the words that the phrase has.
		Returns:
			list of circuits List[cirq.Circuit()]
		"""
		bitstring = None
		data_frame = pd.read_csv(self.data, sep=',')  # converts the csv into a pd.DataFrame
		e = 0
		# goes thought the phrases and constructs the circuits according to them
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
				# goes thought the words in the phrase and appends the corresponding gates to the circuits
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
		""" Parameterizes the gates (on the circuits the gates are parameterised with with sympy.symbols and the
		cirq.Resolver maps the the symbol to a float value) and runs the circuits (based on a stochastic simulation)

		Args:
			theta_sample:
			repetitions:

		Returns:
			the cirq.TrialResult of the circuits
		"""
		circuits = []
		for i in self.circuit_list:
			circuits.append(i[3])
		a = 1
		results = []

		# puts measurements on all the circuits (in case they don't have one)
		for u in circuits:
			if not u.has_measurements():
				for i in u.all_qubits():
					u.append(cirq.measure(i))
					a = a + 1

		# creates the resolver that maps the parameters
		resolver = cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in range(len(theta_sample))})
		for u in circuits:
			# runs each circuit according to the parameters on the resolver
			results.append(cirq.Simulator().run(program=u, param_resolver=resolver, repetitions=repetitions))
		return results
