import csv
import argparse


def txtFileToList(txtFile):
    with open(txtFile) as file:
        lines = file.readlines()
        words = [line.rstrip() for line in lines]
    # print("The list is: ", words)
    return words

def reWriteCSV(args):
    # print(args)
    words = txtFileToList(args.label)
    csvFile = open(args.classInfo)
    csvreader = csv.reader(csvFile)
    rows = []
    for row in csvreader:
        if row[1] in words:
            # print(row[1])
            rows.append(row)

    # print(rows)
    with open(args.output, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the data
        for row in rows:
            writer.writerow(row)

def parse():
    parser = argparse.ArgumentParser(description='create class info csv')

    parser.add_argument(
        '-c',
        '--classInfo',
        type=str,
        required=True,
        help='e.g. class-descriptions-boxable.csv')

    parser.add_argument(
        '-l',
        '--label',
        type=str,
        required=True,
        help='label file name e.g. label.txt')

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='class.csv')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    reWriteCSV(parse())


# python3 createClassCsv.py -c class-descriptions-boxable.csv -l label.txt -o class.csv
