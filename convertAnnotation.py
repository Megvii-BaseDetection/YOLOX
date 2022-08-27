import csv
import argparse


def convertClassCSVToList(classCSV):
    csvFile = open(classCSV)
    csvreader = csv.reader(csvFile)
    label_name_list = []
    for row in csvreader:
        label_name_list.append(row[0].split(",")[0])
    # print(label_name_list)
    return label_name_list

def reWriteCSV(args):
    # print(args)
    class_content = convertClassCSVToList(args.classFile)
    csvFile = open(args.annotation)
    csvreader = csv.reader(csvFile)
    header = []
    header = next(csvreader)
    # print(header)
    rows = []
    for row in csvreader:
        # row 2 is labelname
        if row[2] in class_content:
            rows.append(row)

    # print(class_content)
    # print(rows)
    with open(args.output, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for row in rows:
            writer.writerow(row)

def parse():
    parser = argparse.ArgumentParser(description='pick annotations')

    parser.add_argument(
        '-i',
        '--annotation',
        type=str,
        required=True,
        help='e.g. oidv6-train-annotations-bbox.csv')

    parser.add_argument(
        '-c',
        '--classFile',
        type=str,
        required=True,
        help='class file name e.g. class.csv')

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='annotation.csv')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    reWriteCSV(parse())


# python3 convertAnnotation.py -i oidv6-train-annotations-bbox.csv -c class.csv -o annotation.csv
