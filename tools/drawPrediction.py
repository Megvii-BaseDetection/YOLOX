import argparse
import cv2
import os
import xml.etree.ElementTree as ET

from yolox.data.datasets import PLAYERS_CLASSES


def make_parser():
    parser = argparse.ArgumentParser("Analyse Player Dataset Split")
    parser.add_argument("-d", "--data-dir", type=str, default=None, help="Path to dataset directory(VOC format)")
    parser.add_argument("-r", "--result-dir", type=str, default=None, help="Path to result directory")

    return parser


def main(args):
    imageDir = os.path.join(args.data_dir, "JPEGImages")
    annotDir = os.path.join(args.data_dir, "Annotations")

    for player in PLAYERS_CLASSES:
        resultFile = os.path.join(args.result_dir, "det_test_" + player + ".txt")

        print(resultFile)
        with open(resultFile) as file:
            for line in file:
                d = line.rstrip().split()

                img = cv2.imread(os.path.join(imageDir, d[0]+'.jpeg'))

                img = cv2.rectangle(img,
                                    (int(float(d[2])), int(float(d[3]))),
                                    (int(float(d[4])), int(float(d[5]))),
                                    (0, 0, 255), 2)

                annotPath = os.path.join(annotDir, d[0]+'.xml')
                tree = ET.parse(annotPath)
                target = tree.getroot()

                for obj in target.findall("object"):
                    name = obj.find("name").text.strip()
                    if name == player:
                        bbox = obj.find("bndbox")
                        pts = ["xmin", "ymin", "xmax", "ymax"]
                        bndbox = []
                        for i, pt in enumerate(pts):
                            cur_pt = int(float(bbox.find(pt).text)) - 1
                            # scale height or width
                            # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                            bndbox.append(cur_pt)

                        img = cv2.rectangle(img,
                                            (bndbox[0], bndbox[1]),
                                            (bndbox[2], bndbox[3]),
                                            (0, 255, 0), 2)

                dis = d[0] + "(" + d[1] + ")"
                cv2.imshow(dis, img)
                cv2.waitKey(0)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)