import numpy as np


# MATRIX FORMATS TO GUARANTEE DETERMINANT OF 0
'''
[[n + 2, n + 1, n], [n, n, n], [n, n + 1, n + 2]]
[[A, B, C], [D, BD/A, CD/A], [G, BG/A, CG/A]]
[[1, 0], [0, 1], [a, b]] * S * [1, 0, c], [0, 1, d]], where S is an invertible 2x2 matrix
'''


# A MATRIX OF INTEGERS WILL HAVE AN INVERSE OF INTEGERS IF THE DETERMINANT IS 1 OR -1
# UNIMODULAR MATRICES ALWAYS HAVE DETERMINANT 1 OR -1
# THESE INCLUDE PASCAL MATRICES, PERMUTATION MATRICES, AND TRANSFORMATIONS ON PYTHAGOREAN TRIPLES (?)
# ONE PARAMETRIC FAMILY FOR 3X3 UNIMODULAR MATRIX, WITH HELP FROM WOLFRAM ALPHA
'''
[[8n^2 + 8n, 2n + 1, 4n], [4n^2 + 4n, n + 1, 2n + 1], [4n^2 + 4n + 1, n, 2n - 1]]
'''


def get_unimodular_matrix(n):
    u = np.zeros((3, 3))
    u[0][0] = 8 * n**2 + 8 * n
    u[0][1] = 2 * n + 1
    u[0][2] = 4 * n
    u[1][0] = 4 * n**2 + 4 * n
    u[1][1] = n + 1
    u[1][2] = 2 * n + 1
    u[2][0] = 4 * n**2 + 4 * n + 1
    u[2][1] = n
    u[2][2] = 2 * n - 1
    return u


def get_determinant_2x2(m):
    """
    Gets the determinant of a 2x2 matrix using ad - bc
    :param m: matrix
    :return: int
    """
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def get_determinant_3x3(m):
    """
    Gets the determinant of a 3x3 matrix using the diagonals method
    :param m: matrix
    :return: int
    """
    d = np.array(m).tolist()
    for i in range(2):
        for j in range(3):
            d[j].append(d[j][i])

    sum1 = 0
    for a in range(0, 3):
        x = 1
        for i in range(len(d)):
            x *= d[i][i + a]
        sum1 += x

    sum2 = 0
    for i in range(len(d)):
        x = 1
        for a in range(3):
            x *= d[a][len(d[0]) - (1 + i + a)]
        sum2 += x

    return sum1 - sum2


def get_cofactor_3x3(m):
    """
    Gets the cofactor matrix
    :param m: matrix
    :return: matrix
    """
    d = np.array(m)
    c = np.array(m)
    # MATRIX OF MINORS
    for h in range(len(m)):
        for i in range(len(m)):
            for j in range(len(m[0])):
                c = np.array(m)
                c = np.delete(c, i, axis=1)
            c = np.delete(c, h, axis=0)
            d[h][i] = get_determinant_2x2(c)

    for i in range(len(d)):
        for j in range(len(d[0])):
            if (i + j) % 2 != 0:
                d[i][j] *= -1

    return d


def get_inverse_3x3(m):
    """
    Gets the inverse of the matrix by dividing the adjugate of the cofactor matrix by the determinant
    :param m: matrix
    :return: matrix
    """

    x = np.zeros((len(m), len(m[0])))
    c = get_cofactor_3x3(np.array(m))
    d = get_determinant_3x3(m)

    for i in range(len(c)):
        for j in range(len(c[0])):
            try:
                # BECAUSE WE NEED THE ADJUGATE OF THE COFACTOR MATRIX, THE ROW BECOMES THE COLUMN, I SWITCHED WITH J
                x[j][i] = (1/d) * c[i][j]
            except ZeroDivisionError:
                print('Matrix is invertible')
                exit()

    return x


def matrix_multiplication(m1, m2):  # O(n^3) :( Very inefficient for large strings. Find out more efficient way
    """
    Multiplies two matrices together, assuming the row length of 1 is the same as the column length of 2
    :param m1: matrix
    :param m2: matrix
    :return: matrix
    """
    m = np.zeros((len(m1), len(m2[0])))
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m[i][j] += m1[i][k] * m2[k][j]

    return m


def matrix_division(i, m):
    """
    Multiplies by the inverse of the key matrix to get the original matrix
    :param i: inverse-key matrix
    :param m: matrix
    :return: matrix
    """
    # TO GET THE ORIGINAL MATRIX, MULTIPLY BY THE KEY'S INVERSE
    m = matrix_multiplication(i, m)

    return m

