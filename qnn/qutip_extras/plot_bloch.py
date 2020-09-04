from qutip import Qobj, expect, sigmax, sigmaz, sigmay, Bloch


def add_binary_points(states_a, states_b, bloch):
	a_x, a_y, a_z = [], [], []
	b_x, b_y, b_z = [], [], []

	for i in states_a:
		a_x.append(expect(sigmax(), Qobj(inpt=i)))
		a_y.append(expect(sigmay(), Qobj(inpt=i)))
		a_z.append(expect(sigmaz(), Qobj(inpt=i)))
	for k in states_b:
		b_x.append(expect(sigmax(), Qobj(inpt=k)))
		b_y.append(expect(sigmay(), Qobj(inpt=k)))
		b_z.append(expect(sigmaz(), Qobj(inpt=k)))

	pnt_a = [a_x, a_y, a_z]
	pnt_b = [b_x, b_y, b_z]

	if isinstance(bloch, Bloch):
		bloch.add_points(pnt_a)
		bloch.add_points(pnt_b)
	return bloch
