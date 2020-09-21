import cirq
import sympy
from typing import List

def create_circuits():
	circuits = cirq.Circuit()
	q = []
	gate_1, gate_2, gate_3, gate_4 = [], [], [], []

	for i in range(5):
		q.append(cirq.GridQubit(i, 0))

	theta = sympy.symbols("theta:33")
	for i in range(5):
		gate_1.append(cirq.rx(theta[i])(q[i]))
		#  gate_1.append(cirq.ry(theta[i + 4])(q[i]))
		#  gate_2.append(cirq.rx(theta[i + 8])(q[i]))
		#  gate_2.append(cirq.ry(theta[i + 12])(q[i]))
		#  gate_3.append(cirq.rx(theta[i + 16])(q[i]))
		#  gate_3.append(cirq.ry(theta[i + 20])(q[i]))
		#  gate_4.append(cirq.rx(theta[i + 24])(q[i]))
		#  gate_4.append(cirq.ry(theta[i + 28])(q[i]))

	gates_num = [gate_1, gate_2, gate_3, gate_4]
	circuits.append(gates_num)
	return circuits


def cc_12_34():
	circuits, q = [], []
	gate_1, gate_2, gate_3, gate_4 = [], [], [], []
	theta = sympy.symbols("theta:20")

	for i in range(5):
		q.append(cirq.GridQubit(i, 0))

	for i in range(5):
		gate_1.append(cirq.rx(theta[i])(q[i]))
		gate_2.append(cirq.rx(theta[i + 5])(q[i]))
		gate_3.append(cirq.rx(theta[i + 10])(q[i]))
		gate_4.append(cirq.rx(theta[i + 15])(q[i]))

	circuits.append(cirq.Circuit())
	circuits[0].append([gate_1, gate_2])
	circuits.append(cirq.Circuit())
	circuits[1].append([gate_3, gate_4])

	return circuits


def cc_1234567():
	circuits, q = [], []
	gate_1, gate_2, gate_3, gate_4, gate_5, gate_6 = [], [], [], [], [], []
	theta = sympy.symbols("theta:32")

	for i in range(7):
		q.append(cirq.GridQubit(i, 0))

	for i in range(7):
		gate_1.append(cirq.rx(theta[i])(q[i]))
		gate_2.append(cirq.rx(theta[i + 5])(q[i]))
		gate_3.append(cirq.rx(theta[i + 10])(q[i]))
		gate_4.append(cirq.rx(theta[i + 15])(q[i]))
		gate_5.append(cirq.rx(theta[i + 20])(q[i]))
		gate_6.append(cirq.rx(theta[i + 25])(q[i]))

	circuits.append(cirq.Circuit())
	circuits[0].append([gate_1, gate_2, gate_3, gate_4, gate_5, gate_6])
	circuits.append(cirq.Circuit())
	circuits[1].append([gate_1, gate_2, gate_3, gate_4, gate_5])
	circuits.append(cirq.Circuit())
	circuits[2].append([gate_1, gate_2, gate_3, gate_4])
	circuits.append(cirq.Circuit())
	circuits[3].append([gate_1, gate_2, gate_3])
	circuits.append(cirq.Circuit())
	circuits[4].append([gate_1, gate_2])

	return circuits


def cc_1234567_bitstring():
	circuits, q = [], []
	gate_1, gate_2, gate_3, gate_4, gate_5, gate_6 = [], [], [], [], [], []
	theta = sympy.symbols("theta:32")

	for i in range(3):
		q.append(cirq.GridQubit(i, 0))

	for i in range(3):
		gate_1.append(cirq.rx(theta[i])(q[i]))
		gate_2.append(cirq.rx(theta[i + 3])(q[i]))
		gate_3.append(cirq.rx(theta[i + 6])(q[i]))
		gate_4.append(cirq.rx(theta[i + 9])(q[i]))
		gate_5.append(cirq.rx(theta[i + 12])(q[i]))
		gate_6.append(cirq.rx(theta[i + 15])(q[i]))

	circuits.append(cirq.Circuit())
	circuits[0].append([gate_1, gate_2, gate_3, gate_4, gate_5, gate_6])
	circuits.append(cirq.Circuit())
	circuits[1].append([gate_1, gate_2, gate_3, gate_4, gate_5])
	circuits.append(cirq.Circuit())
	circuits[2].append([gate_1, gate_2, gate_3, gate_4])
	circuits.append(cirq.Circuit())
	circuits[3].append([gate_1, gate_2, gate_3])
	circuits.append(cirq.Circuit())
	circuits[4].append([gate_1, gate_2])

	return circuits


def sample_run_global(circuits: List[cirq.Circuit], theta_sample, repetitions):
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


def sample_run(circuits: cirq.Circuit(), theta_sample, repetitions):
	a = 1
	if not circuits.has_measurements():
		for i in circuits.all_qubits():
			circuits.append(cirq.measure(i))
			a = a + 1

	resolver = cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in range(len(theta_sample))})
	return cirq.Simulator().run(program=circuits, param_resolver=resolver, repetitions=repetitions)


def sample_simulate(circuits: cirq.Circuit(), theta_sample):
	resolver = cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in range(len(theta_sample))})
	return cirq.Simulator().simulate(program=circuits, param_resolver=resolver)

