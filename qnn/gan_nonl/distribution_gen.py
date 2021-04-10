import numpy as np
import time


def gen_real_data(a, b, num_samples: int, file_path: str):
    """
    Generates the real data from a continuous uniform distribution.

    Args:
        a: minimum value of the unif
        b: maximum value of the unif
        num_samples: number of examples that are going to be generated (N_{e})
        file_path: path where the .txt file is going to be created

    Returns: .txt file path of the generated data
    """
    real_data = []
    x0 = np.random.default_rng().uniform(a, b, num_samples)
    x1 = np.random.default_rng().uniform(a, b, num_samples)

    for i in range(len(x1)):
        real_data.append([[x0[i], 0], [x1[i], 0]])

    time_file = time.strftime("%Y_%m_%d-%H%M%S")

    f = open(file_path + time_file + '.txt', 'w')
    f.write('REAL_DATA' + "\n" + time_file + "\n" + str(num_samples) + "\n \n")
    for item in real_data:
        f.write(str(item) + "\n")
    f.close()

    return 'C:/Users/usuario/Desktop/QIT/QNNs/qnn/gan_nonl/dataset/real_data_' + time_file + '.txt'


file_data_path = gen_real_data(.4, .6, 10, 'C:/Users/usuario/Desktop/QIT/QNNs/qnn/gan_nonl/dataset/real_data_')
print(file_data_path)
