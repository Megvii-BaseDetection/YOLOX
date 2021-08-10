import unittest
from Fusion.utils.convert import calculate_depth, nonlinear, get_distance


class MyTestCase(unittest.TestCase):
    def test_calculate_depth(self):
        for i in range(0, 1080):
            # 1920,1080
            left = 960
            top = 0
            right = 960
            bottom = i
            rect_roi = [left, top, right, bottom]
            camera_xyz, distance = calculate_depth(rect_roi)
            print('i', i, '  ', 'distance', distance)
        self.assertEqual(True, True)

    def test_nonlinear(self):
        nonlinear()
        self.assertEqual(True, True)

    def test_get_distance(self):
        for i in range(0, 1080):
            left = 960
            top = 0
            right = 960
            bottom = i
            rect_roi = [left, top, right, bottom]
            camera_xyz, distance = get_distance(rect_roi)
            print('i', i, '  ', 'distance', distance)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
