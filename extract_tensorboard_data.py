"""
This file has been written to extract the data from tensorflow eventfile (log files) to print scalers on a single file

(losses, precisions, learning rates etc.)

Data for all scalers will be extracted and plotted on a single graph
later on the graph and corresponding scaler data will be saved in the respective directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_events(dpath: str) -> dict:

    final_out = {}
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}", end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            tag_values = []
            wall_time = []
            steps = []

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            out[tag] = pd.DataFrame(data=dict(
                zip(steps,
                    np.array([tag_values, wall_time]).transpose())),
                                    columns=steps,
                                    index=['value', 'wall_time'])

        if len(tags) > 0:
            df = pd.concat(out.values(), keys=out.keys())
            df.to_csv(dpath + '/' + f'{dname}.csv')
            print("- Done")
        else:
            print('- Not scalers to write')

        final_out[dname] = df

    return final_out


"""
This method is plotting the extracted data for every scalar

Inputs:
    Extracted vlues from event files (output of extract_events function)
Outputs:
    CSV file containing the values of scalars
    PNG image of plot
    
Usage:
    Every model has scaler names(e.g. loss, val_loss, class_loss, class_val_loss etc.)
    please run the "extract_events()" function first and see the names of your model variables,
    then, replace the "epoch_class_loss, epoch_reg_loss, epoch_loss, epoch_val_loss, epoch_val_class_loss, epoch_val_reg_loss, epoch_mAP" with the names of your scalers
    ---------------------------------------------------------------
    NOTE: Without correct descriptions plot function will not work
    ---------------------------------------------------------------
"""


def plot_model_details(model_details: list) -> None:

    plt.rcParams["font.family"] = "Times New Roman"

    scaler_df = pd.DataFrame(model_details)
    epoch_class_loss = list(
        scaler_df.loc['epoch_classification_loss'].loc['value'])
    epoch_reg_loss = list(scaler_df.loc['epoch_regression_loss'].loc['value'])
    epoch_loss = list(scaler_df.loc['epoch_loss'].loc['value'])
    epoch_val_loss = list(scaler_df.loc['epoch_val_loss'].loc['value'])
    epoch_val_class_loss = list(
        scaler_df.loc['epoch_val_classification_loss'].loc['value'])
    epoch_val_reg_loss = list(
        scaler_df.loc['epoch_val_regression_loss'].loc['value'])
    epoch_mAP = list(scaler_df.loc['mAP'].loc['value'])

    x_axis = np.arange(1, len(epoch_loss) + 1, 1)
    ref_scatter_axis = np.arange(1, len(epoch_loss) + 25, 25)
    ref_y_axis_1 = []
    ref_y_axis_2 = []
    ref_y_axis_3 = []
    ref_y_axis_4 = []
    ref_y_axis_5 = []
    ref_y_axis_6 = []
    ref_y_axis_7 = []

    for i in np.arange(1, len(epoch_loss) + 1, 25):

        ref_y_axis_1.append(epoch_class_loss[i])
        ref_y_axis_2.append(epoch_reg_loss[i])
        ref_y_axis_3.append(epoch_loss[i])
        ref_y_axis_4.append(epoch_val_loss[i])
        ref_y_axis_5.append(epoch_val_class_loss[i])
        ref_y_axis_6.append(epoch_val_reg_loss[i])
        ref_y_axis_7.append(epoch_mAP[i])

    ref_y_axis_1.append(epoch_class_loss[-1])
    ref_y_axis_2.append(epoch_reg_loss[-1])
    ref_y_axis_3.append(epoch_loss[-1])
    ref_y_axis_4.append(epoch_val_loss[-1])
    ref_y_axis_5.append(epoch_val_class_loss[-1])
    ref_y_axis_6.append(epoch_val_reg_loss[-1])
    ref_y_axis_7.append(epoch_mAP[-1])

    fig, ax = plt.subplots(figsize=(20, 10))

    # x2=16.2373
    # y2 = 0.8663

    ax.plot(x_axis,
            epoch_loss,
            lw=2,
            color='cyan',
            label="epoch training loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_1, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis,
            epoch_class_loss,
            lw=2,
            color='green',
            label="epoch classification loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_2, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis,
            epoch_reg_loss,
            lw=2,
            color='orange',
            label="epoch regression loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_3, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis,
            epoch_val_loss,
            lw=2,
            color='brown',
            label="epoch validation loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_4, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis,
            epoch_val_class_loss,
            lw=2,
            color='blue',
            label="epoch validation classification loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_5, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis,
            epoch_val_reg_loss,
            lw=2,
            color='black',
            label="epoch validation regression loss")
    # ax.scatter(ref_scatter_axis, ref_y_axis_6, lw=2, color = 'black', marker = 'o')

    ax.plot(x_axis, epoch_mAP, lw=4, color='red', label="epoch mAP")
    # ax.scatter(ref_scatter_axis, ref_y_axis_7, lw=2, color = 'black', marker = 'o')

    ax.set_xlabel('no. of epochs', labelpad=15, fontsize=30, color="black")
    ax.set_ylabel('epoch details: loss and mAP',
                  labelpad=15,
                  fontsize=30,
                  color="black")
    plt.xlim(0, len(epoch_loss) + 10)
    plt.ylim(0, 2)

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    #     ax.annotate("(16.23, 0.86)", fontsize = 8, family = "Times New Roman", xy=(x2,y2), xycoords = "data", xytext=(+20,+30), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle = "arc3, rad=0.2"))

    # ax.text(1, 9, 'Data Acquisition = 120 s\nShear(D) = 200 1/s \nTemperature = 20 {}'.format("\u2103"), verticalalignment='top', horizontalalignment='left', color='black', fontname = 'serif', fontsize=8)

    ax.legend(fontsize=20)
    ax.grid(True)
    plt.savefig(path + '/tensorboard_graphs.png')
    return None


"""
path = "log folder where the particular event file is present"
scaler_values = "title of the event file which you want to extract/ plot"

"""
if __name__ == '__main__':
    path = "logs/2020-08-13 (B0 MND WBiFPN)"
    steps = extract_events(path)
    pd.concat(steps.values(),
              keys=steps.keys()).to_csv(path + '/all_result.csv')
    scaler_values = steps['events.out.tfevents.1597329330.43654777943f']
    plot_model_details(scaler_values)
