#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import tempfile
import unittest

from yolox.evaluators.voc_eval import voc_eval

ANNO_TEMPLATE = """<annotation>
  <object>
    <name>dog</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
  </object>
</annotation>
"""


class TestVOCEvalCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = self.tmpdir.name
        self.annopath = os.path.join(root, "{}.xml")
        self.imagesetfile = os.path.join(root, "test.txt")
        self.detpath = os.path.join(root, "det_{}.txt")
        self.cachedir = os.path.join(root, "annotations_cache")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_dataset(self, imagenames):
        for name in imagenames:
            with open(self.annopath.format(name), "w") as f:
                f.write(ANNO_TEMPLATE)
        with open(self.imagesetfile, "w") as f:
            f.write("\n".join(imagenames) + "\n")
        with open(self.detpath.format("dog"), "w") as f:
            for name in imagenames:
                f.write(f"{name} 0.9 10 10 50 50\n")

    def _eval(self):
        return voc_eval(
            self.detpath, self.annopath, self.imagesetfile, "dog", self.cachedir
        )

    def test_stale_cache_is_rebuilt(self):
        # first eval writes a cache covering only img1
        self._write_dataset(["img1"])
        _, _, ap = self._eval()
        self.assertAlmostEqual(ap, 1.0)

        # the image set grows, the old cache must not be trusted
        self._write_dataset(["img1", "img2"])
        rec, _, ap = self._eval()
        self.assertAlmostEqual(ap, 1.0)
        self.assertAlmostEqual(rec[-1], 1.0)

        with open(os.path.join(self.cachedir, "annots.pkl"), "rb") as f:
            recs = pickle.load(f)
        self.assertEqual(set(recs), {"img1", "img2"})

    def test_valid_cache_is_reused(self):
        self._write_dataset(["img1", "img2"])
        self._eval()

        # a cache that covers every image is used as is, without parsing the xml
        os.remove(self.annopath.format("img1"))
        os.remove(self.annopath.format("img2"))
        _, _, ap = self._eval()
        self.assertAlmostEqual(ap, 1.0)


if __name__ == "__main__":
    unittest.main()
