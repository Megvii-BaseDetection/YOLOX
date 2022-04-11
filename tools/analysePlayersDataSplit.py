import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET
from tabulate import tabulate

from yolox.data.datasets import PLAYERS_CLASSES


def make_parser():
    parser = argparse.ArgumentParser("Analyse Player Dataset Split")
    parser.add_argument("-d", "--data-dir", type=str, default=None, help="Path to dataset directory(VOC format)")

    return parser


def main(args):
    sampleDict = {}
    for player in PLAYERS_CLASSES:
        sampleDict[player] = {'train': 0, 'validation': 0, 'test': 0}

    root = args.data_dir

    image_sets = ['test', 'validation', 'train']

    for setName in image_sets:
        for line in open(os.path.join(root, "ImageSets", setName + ".txt")):
            annotPath = os.path.join(root, "Annotations", line.strip() + ".xml")
            target = ET.parse(annotPath).getroot()

            for obj in target.iter("object"):
                name = obj.find("name").text.strip()

                sampleDict[name][setName] += 1

    df = pd.DataFrame(sampleDict)
    print(tabulate(df.T, headers='keys', tablefmt='psql'))


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)

