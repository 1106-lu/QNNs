from numpy import array, complex64
from qnn.qutip_extras.plot_bloch import add_binary_points
from qutip import Bloch
import matplotlib.pyplot as plt

sample_vectors = [array([-0.3734891 - 1.4901161e-08j, -0.2772529 - 8.8523257e-01j],
                        dtype=complex64),
                  array([-0.64384806 + 1.4901161e-08j, -0.133582 - 7.5340277e-01j],
                        dtype=complex64),
                  array([-0.01224577 - 2.9802322e-08j, -0.40438393 - 9.1450751e-01j],
                        dtype=complex64),
                  array([-0.34250805 + 2.9802322e-08j, -0.29040772 - 8.9350522e-01j],
                        dtype=complex64),
                  array([-0.42388362 + 2.9802322e-08j, -0.25469527 - 8.6916792e-01j],
                        dtype=complex64),
                  array([-0.5673448 + 5.2154064e-08j, -0.18083268 + 8.0338001e-01j],
                        dtype=complex64),
                  array([-0.5999606 + 4.4703484e-08j, -0.16151446 + 7.8355616e-01j],
                        dtype=complex64),
                  array([-0.33245653 - 2.9802322e-08j, -0.29456693 + 8.9593709e-01j],
                        dtype=complex64),
                  array([-0.22667933 + 4.4703484e-08j, -0.33540434 + 9.1439605e-01j],
                        dtype=complex64),
                  array([-0.37634954 - 1.4901161e-08j, -0.27601194 + 8.8440853e-01j],
                        dtype=complex64),
                  array([-0.42152995 + 0.j, 0.90586174 - 0.04155887j],
                        dtype=complex64),
                  array([-0.46164975 + 1.4901161e-08j, 0.8797116 + 1.1396010e-01j],
                        dtype=complex64),
                  array([-0.42512128 - 1.1175871e-08j, 0.90356 - 5.3393066e-02j],
                        dtype=complex64),
                  array([-0.434758 - 3.7252903e-09j, 0.8973474 + 7.5848125e-02j],
                        dtype=complex64),
                  array([-0.52232504 - 2.9802322e-08j, 0.83812654 + 1.5722761e-01j],
                        dtype=complex64),
                  array([-0.41617435 - 6.9849193e-10j, 0.90927994 - 2.9861990e-03j],
                        dtype=complex64),
                  array([-0.49312547 - 1.4901161e-08j, 0.8584694 - 1.4091648e-01j],
                        dtype=complex64),
                  array([-0.41681242 + 9.3132257e-10j, 0.9088735 + 1.4708021e-02j],
                        dtype=complex64),
                  array([-0.64143956 - 2.9802322e-08j, 0.747146 - 1.7414951e-01j],
                        dtype=complex64),
                  array([-0.43556735 - 3.7252903e-09j, 0.89682305 + 7.7389732e-02j],
                        dtype=complex64)]

b = Bloch()
add_binary_points(sample_vectors[10:], sample_vectors[:10], b)
b.render(b.fig, b.axes)
plt.show()