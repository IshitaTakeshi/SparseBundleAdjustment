import re
import csv
from io import StringIO

from scipy.sparse import csr_matrix
import numpy as np


def parse(row):
    n_frames = int(row[3])
    ground_truth = [float(e) for e in row[:3]]

    data = []
    indices = []
    for i in range(n_frames):
        k = 4+i*3
        frame_index = int(row[k])
        observation = [float(e) for e in row[k+1:k+3]]

        data.append(observation)
        indices += [2 * frame_index, 2 * frame_index + 1]
    return ground_truth, data, indices


def strip(f):
    raw = f.read()
    raw = re.sub(r" +", " ", raw)
    f = StringIO(raw)
    f = filter(lambda row: row[0] != '#', f)
    return f


def sparse_nan_matrix(data, indices, indptr):
    """
    Same interface as scipy.sparse.csr_matrix
    """

    n_columns = np.max(indices) + 1
    n_rows = len(indptr) - 1

    M = np.empty((n_rows, n_columns))
    M[:, :] = np.nan

    for i, (begin, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        M[i, indices[begin:end]] = data[begin:end]
    return M



def load_sba(path):
    with open(path) as f:
        reader = csv.reader(strip(f), delimiter=' ')

        ground_truth = []
        data = []
        indptr = [0]
        indices = []
        for row in reader:
            gt, d, indices_ = parse(row)
            ground_truth.append(gt)
            data += d
            indices += indices_
            indptr.append(len(indices))

    ground_truth = np.array(ground_truth)
    data = np.array(data)
    M = sparse_nan_matrix(data.flatten(), indices, indptr)
    return ground_truth, M


GT, M = load_sba("data/9pts.txt")
np.set_printoptions(precision=2, linewidth=100)
print(M)
