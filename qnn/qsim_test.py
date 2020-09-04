import cirq
import qsimcirq

c = cirq.Circuit()
c.append(cirq.H(cirq.GridQubit(1, 0)))

qsim_options = {'t': 8, 'v': 0}
simulator = qsimcirq.QSimSimulator(qsim_options)
myres = simulator.simulate(program=c)