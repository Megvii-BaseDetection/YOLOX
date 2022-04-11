import argparse
import os
import xml.etree.ElementTree as ET
import shutil


def make_parser():
    parser = argparse.ArgumentParser("Create Player Dataset Split")
    parser.add_argument("-d", "--data-dir", type=str, default=None, help="Path to dataset directory(VOC format)")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Path to save data split")
    parser.add_argument("-tr", "--train-data-split", type=int, default=80, help="Train data split")
    parser.add_argument("-te", "--test-data-split", type=int, default=10, help="Test data split")
    parser.add_argument("-va", "--validation-data-split", type=int, default=10, help="Validation data split")

    return parser


def main(args):
    assert args.train_data_split + args.test_data_split + args.validation_data_split == 100, "Split sum should be equal to 100."
    dataDir = args.data_dir

    annotationDir = os.path.join(dataDir, "Annotations")

    data = {}
    for file in os.listdir(annotationDir):
        fileName = os.path.splitext(file)[0]

        annotPath = os.path.join(annotationDir, file)
        tree = ET.parse(annotPath)
        target = tree.getroot()

        for obj in target.findall("object"):
            name = obj.find("name").text.strip()
            if name in data:
                data[name].append(fileName)
            else:
                data[name] = [fileName]

    # minimum image required to split data
    percentSort = sorted([args.train_data_split, args.test_data_split, args.validation_data_split])
    minImageCount = 1 + int(percentSort[1] / percentSort[0]) + int(percentSort[2] / percentSort[0])

    trainSet = []
    validationSet = []
    testSet = []

    for name in data:
        dataLength = len(data[name])
        if dataLength < minImageCount:
            print("Skipping dataset file for {}... Only {} instances found".format(name, dataLength))
            continue
        trainSplit = int((args.train_data_split / 100) * dataLength)
        validationSplit = trainSplit + int((args.validation_data_split / 100) * dataLength)

        trainSet.extend(data[name][:trainSplit])
        validationSet.extend(data[name][trainSplit: validationSplit])
        testSet.extend(data[name][validationSplit:])

    trainSet = list(set(trainSet))
    validationSet = list(set(validationSet))
    testSet = list(set(testSet))

    validationSet = [x for x in validationSet if x not in trainSet]
    testSet = [x for x in testSet if x not in trainSet and x not in validationSet]

    print("Train Set: ", len(trainSet))
    print("Validation Set: ", len(validationSet))
    print("Test Set: ", len(testSet))

    # save data
    outputDir = args.output_dir
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)

    os.makedirs(os.path.join(outputDir, "ImageSets"))

    shutil.copytree(os.path.join(dataDir, "JPEGImages"), os.path.join(outputDir, "JPEGImages"))
    shutil.copytree(os.path.join(dataDir, "Annotations"), os.path.join(outputDir, "Annotations"))

    with open(os.path.join(outputDir, "ImageSets", "train.txt"), 'w') as f:
        for name in trainSet:
            f.write("%s\n" % name)

    with open(os.path.join(outputDir, "ImageSets", "validation.txt"), 'w') as f:
        for name in validationSet:
            f.write("%s\n" % name)

    with open(os.path.join(outputDir, "ImageSets", "test.txt"), 'w') as f:
        for name in testSet:
            f.write("%s\n" % name)

    with open(os.path.join(outputDir, "classes.txt"), 'w') as f:
        for name in data:
            f.write("%s\n" % name)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)

# root = "./datasets/playersData"
# image_sets = ['test', 'validation', 'train']
#
# newRoot = "/Users/aditya/Documents/Work/YOLOX/pascal_voc-part2"
#
# for setName in image_sets:
#     for line in open(os.path.join(root, "ImageSets", setName + ".txt")):
#         imagePath = os.path.join(root, "JPEGImages", line.strip() + ".jpeg")
#         annotPath = os.path.join(root, "Annotations", line.strip() + ".xml")
#         tree = ET.parse(annotPath)
#         target = tree.getroot()
#
#         for obj in target.findall("object"):
#             name = obj.find("name").text.strip()
#             if name not in PLAYERS_CLASSES:
#                 target.remove(obj)
#
#         if len(target.findall("object")) == 0:
#             continue
#
#         newImagePath = os.path.join(newRoot, "JPEGImages", line.strip() + ".jpeg")
#         newAnnotPath = os.path.join(newRoot, "Annotations", line.strip() + ".xml")
#
#         shutil.copy(imagePath, newImagePath)
#         tree.write(newAnnotPath)

