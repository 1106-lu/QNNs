import cirq
import sympy

from qnn.qnlp.frases_database import extract_words


class CircuitsWords:
	def __init__(self, data: str):
		""""
		Creates the gates of the words present in the input phrases
		:parameter data: path to the excel file with the phrases"""

		self.data = data
		self.theta = sympy.symbols("theta:100")
		self.num_qubits = 3

		self.voc, self.df = extract_words(self.data)
		circuits, q, self.gates = [], [], []

		for i in range(self.num_qubits):
			q.append(cirq.GridQubit(i, 0))

		a = 0
		for k in range(len(self.voc)):
			gates_words = []
			for j in range(self.num_qubits):
				gates_words.append(cirq.rx(self.theta[a])(q[j]))
				a = a + 1
			self.gates.append(gates_words)
		self.dic_gates = {self.voc[e]: self.gates[e] for e in range(len(self.voc))}

	def __repr__(self):
		"""prints the gates of the words in the vocabulary"""
		for i in self.gates:
			print(i)
		return str(self.gates)

	def create(self, phrase: str, next: str):
		c = cirq.Circuit()
		for k in phrase.split():
			a = 0
			for f in self.voc:
				if f == k:
					c.append(self.gates[a])
				a += 1
		return c, next
