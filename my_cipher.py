# PYTHON IMPLEMENTATION OF MY CIPHER
# PROCESS
# A 3 x 3 matrix is randomly generated that is used to encode the message
# The message is transformed to numbers and put into a 3 x n matrix
# The message matrix is multiplied by the encryption matrix

import numpy as np
import string
import matrices


integer = int(input('Enter an integer to encode the message: '))
key = matrices.get_unimodular_matrix(integer)
inv = matrices.get_inverse_3x3(key)


def preprocessing(s):
    """
    Removes all punctuation and spaces and makes everything lower-case
    :param s: str
    :return: str
    """
    s = s.replace(' ', '')
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace(';', '')
    s = s.replace('?', '')
    s = s.replace(':', '')
    s = s.replace("'", '')
    s = s.replace('"', '')
    s = s.lower()
    return s


def let_to_num(l):
    """
    Converts a letter to a number
    :param l: str
    :return: int
    """
    return string.ascii_lowercase.index(l)


def num_to_let(n):
    """
    Converts a number to a letter
    :param n: float
    :return: char
    """
    return chr(int(n) + 97)


def mod(x):
    """
    Gets the modulo of a number
    :param x: int
    :return: int
    """
    while x < 0:
        x += 26

    while x > 25:
        x = x - 26

    return x


def mod_inv(a, n=26):
    """
    Gets the inverse modulo of a number, assuming a and n are coprime
    :param a: int
    :param n: int (26)
    :return: int
    """
    # (a * inv_a) % n = 1
    for i in range(1, n):
        if (i * a) % n == 1:
            return i


# CONVERT NUMBERS TO MATRIX
def convert_to_matrix(s):
    """
    Converts string to matrix
    :param s: string
    :return: matrix
    """
    s = preprocessing(s)

    text = []
    for i in range(len(s)):
        text.append(let_to_num(s[i]))

    rows = 0
    if len(text) % 3 == 0:
        rows = int(len(text)/3)
    else:
        rows = int(len(text)/3) + 1

    matrix = np.zeros((3, rows))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            try:
                matrix[i][j] = text[i * rows + j]
            except IndexError:
                matrix[i][j] = 16

    return matrix


def encrypt(m, k):
    """
    Encrypts the matrix based on the key
    :param m: matrix
    :param k: key-matrix
    :return: matrix
    """
    m = matrices.matrix_multiplication(k, m)
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = mod(m[i][j])

    return m


def decrypt(i, m):
    """
    Decrypts the encrypted matrix based on the inverse key
    :param i: inverse-key matrix
    :param m: matrix
    :return: matrix
    """
    m = matrices.matrix_division(i, m)
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = mod(m[i][j])

    return m


def convert_to_string(m):
    """
    Converts a matrix to a string
    :param m: matrix
    :return: str
    """
    s = ''
    for i in range(len(m)):
        for j in range(len(m[0])):
            s += num_to_let(m[i][j])

    if len(s) % 3 == 0:
        while s[len(s) - 1] == 'q':
            s = s[0:len(s) - 1]

    return s


print('The quick brown fox jumped over the lazy dog.')
text = convert_to_matrix('The quick brown fox jumped over the lazy dog.')

encrypted = encrypt(text, key)

e = convert_to_string(encrypted)
print(e)

decrypted = decrypt(inv, encrypted)

s = convert_to_string(decrypted)
print(s)




