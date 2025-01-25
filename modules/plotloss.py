import matplotlib.pyplot as plt
import pandas as pd

def plotLoss(data, type):
    if type == "train":
        plotTitle = "Training Loss"
        saveName = "trainloss.png"
    elif type == "dev":
        plotTitle = "Dev Loss"
        saveName = "devloss.png"
    else:
        print("Invalid plotLoss type")
        exit()
    plt.clf()
    data.plot(x="Steps", y="Loss", legend=False)
    plt.xlabel("Number of steps")
    plt.ylabel("Loss")
    plt.title(plotTitle)
    plt.savefig(saveName)
    plt.close()