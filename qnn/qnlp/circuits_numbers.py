import cirq
import sympy

def create_circuits():
	circuits = cirq.Circuit()
	q = []
	gate_1, gate_2, gate_3, gate_4 = [], [], [], []

	for i in range(5):
		q.append(cirq.GridQubit(i, 0))

	theta = sympy.symbols("theta:33")
	for i in range(5):
		gate_1.append(cirq.rx(theta[i])(q[i]))
		gate_1.append(cirq.ry(theta[i + 4])(q[i]))
		gate_2.append(cirq.rx(theta[i + 8])(q[i]))
		gate_2.append(cirq.ry(theta[i + 12])(q[i]))
		gate_3.append(cirq.rx(theta[i + 16])(q[i]))
		gate_3.append(cirq.ry(theta[i + 20])(q[i]))
		gate_4.append(cirq.rx(theta[i + 24])(q[i]))
		gate_4.append(cirq.ry(theta[i + 28])(q[i]))

	gates_num = [gate_1, gate_2, gate_3, gate_4]
	circuits.append(gates_num)
	return circuits


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

