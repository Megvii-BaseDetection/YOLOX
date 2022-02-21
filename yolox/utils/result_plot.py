import matplotlib.pyplot as plt
from datetime import datetime


class ResultPlot:
    # field
    list_mAP: list
    list_epoch_num: list

    # OS parent directory
    parent_dir = "/content/YOLOX/"

    # date/time
    execution_date: datetime

    def __init__(self):
        self.list_mAP = []
        self.list_epoch_num = []
        self.execution_date = datetime.now()

    def add_new_mAP(self, new_mAP):
        self.list_mAP.append(new_mAP)
        print(f"New mAP {new_mAP} has been added to the list")
        print(self.list_mAP)

    def plot_and_save(self):
        # allocating values for x-axis
        for i in range(1, len(self.list_mAP) + 1):
            self.list_epoch_num.append(i)

        plt.plot(self.list_epoch_num, self.list_mAP)
        plt.title("Metrics/mAP_0.5")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")

        plt.ylim(0, max(self.list_mAP))
        plt.xlim(0, max(self.list_epoch_num))
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8])

        fig_name = "mAP_Collection_Result_" + self.execution_date.strftime("%d%m%Y%H%M%S") + ".png"

        plt.savefig(self.parent_dir + "YOLOX_outputs/Graph/" + fig_name)

        plt.show()


