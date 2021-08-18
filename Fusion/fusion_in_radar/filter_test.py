import unittest
import numpy as np


def delete(a):
    a_index = [0]
    a_index = np.delete(a_index, 0, axis=0)
    for i in range(len(a)):
        if i > 1:
            a_index = np.append(a_index, i)
    a = np.delete(a, a_index, axis=0)
    return a

class MyTestCase(unittest.TestCase):
    def test_something(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        print('a.shape', a.shape)
        print('a', a)
        a = delete(a)
        print('a.shape', a.shape)
        print('a', a)

        b = np.array([[1, 11, 12, 13, 2],
                      [3, 11, 12, 13, 4],
                      [5, 11, 12, 13, 6]])
        print('b.shape', b.shape)
        print('b', b)
        b = np.delete(b, 1, axis=0)
        print('b.shape', b.shape)
        print('b', b)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
