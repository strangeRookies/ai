import unittest

from ai.detection.yolo_person_detector import filter_person_boxes


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[0, 1, 10, 20], [30, 40, 50, 60]])
    conf = FakeTensor([0.9, 0.8])
    cls = FakeTensor([0, 2])


class YoloPersonFilterTest(unittest.TestCase):
    def test_filter_person_only(self):
        boxes = filter_person_boxes(FakeBoxes(), {0: "person", 2: "car"})
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["class_name"], "person")
        self.assertEqual(boxes[0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
