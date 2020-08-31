from qutip import Qobj, expect, sigmax, sigmaz, sigmay, Bloch

def add_binary_points(states_a, states_b, bloch):
	a_x, a_y, a_z = [], [], []
	b_x, b_y, b_z = [], [], []
	a_s, vec_a, b_s, vec_b = [], [], [], []

	for i in states_a:
		a_s.append(Qobj(inpt=i))
	for k in states_b:
		b_s.append(Qobj(inpt=k))

	for st in a_s:
		a_x.append(expect(sigmax(), st))
		a_y.append(expect(sigmay(), st))
		a_z.append(expect(sigmaz(), st))
	for st in b_s:
		b_x.append(expect(sigmax(), st))
		b_y.append(expect(sigmay(), st))
		b_z.append(expect(sigmaz(), st))

	pnt_a = [a_x, a_y, a_z]
	pnt_b = [b_x, b_y, b_z]

	if isinstance(bloch, Bloch):
		bloch.add_points(pnt_a)
		bloch.add_points(pnt_b)
	return bloch