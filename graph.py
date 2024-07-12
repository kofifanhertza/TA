import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

data = pd.read_csv("WIN_8H.csv")


# Extract the necessary columns
gpu_temp_split = data['GPU Temperature'].tolist()
gpu_perc_split = (data['GPU Memory']/1600 * 100).tolist()
mem_perc_split = data['RAM Percentage'].tolist()

# Define the function (as provided by you)
def draw_graph(device, gpu_temp_split, gpu_perc_split, mem_perc_split):
    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    # Set titles and labels
    ax.set_title(f'{device} Performance Result', pad=20)  # pad=20 to provide space for top x-axis labels
    ax.set_xlabel('Time (in hours)')

    width = 0.4

    # Convert lists to DataFrames
    df_gpu_temp = pd.DataFrame({'Value': gpu_temp_split})
    df_gpu_perc = pd.DataFrame({'Value': gpu_perc_split})
    df_mem_perc = pd.DataFrame({'Value': mem_perc_split}) 

    print(df_gpu_perc)
    print(df_mem_perc)

    # Plot the area graph first
    #df_gpu_temp.plot(kind='area', color='green', label="GPU Temperature", ax=ax)

    # Then plot the bar charts
    df_gpu_perc.plot(kind='bar', color='red', label="GPU Utilization", ax=ax2, width=width, position=1)
    df_mem_perc.plot(kind='bar', color='blue', label="Memory Utilization", ax=ax2, width=width, position=0)

    # Set x-axis labels.
    new_labels = range(5, 65, 5)  # This will create a list [5, 10, 15, ..., 60]
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels)

    # Set y-axis for the right axis (percentages)
    percentage_ticks = range(45, 100, 5)
    ax2.set_yticks(percentage_ticks) 
    ax2.set_ylim([45, 100])

    # Set y-axis for the left axis (temperature)
    temperature_ticks = range(40, 62, 2)
    ax.set_yticks(temperature_ticks)
    ax.set_ylim([40, 62])

    # Draw horizontal lines across the entire plot for temperature ticks
    for i in temperature_ticks:
        ax.axhline(y=i, color='grey', linestyle='-', lw=1, zorder=-1)

    # Set y-axis labels
    ax.set_ylabel('Temperature (°C)')
    ax2.set_ylabel('Percentage (%)')

    # Create and place the legend
    gpu_temp_legend = mpatches.Patch(color='green', label='GPU Temperature') 
    gpu_perc_legend = mpatches.Patch(color='red', label='GPU Utilization [%]') 
    mem_perc_legend = mpatches.Patch(color='blue', label='Memory Utilization [%]') 

    plt.legend(bbox_to_anchor=(0.5, -0.25), loc='lower center', handles=[gpu_temp_legend, gpu_perc_legend, mem_perc_legend], frameon=False, ncol=3)

    # Create secondary x-axis at the top
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())  # Ensure the secondary x-axis has the same limits as the primary x-axis
    ax3.set_xticks(ax.get_xticks())  # Ensure the secondary x-axis has the same ticks as the primary x-axis
    ax3.set_xticklabels(ax.get_xticklabels())  # Set the same labels as the primary x-axis
    ax3.set_xlabel('Time (in hours)')  # Set the same x-label if needed

    plt.show()



def draw_graph_corrected(device, gpu_temp_split, gpu_perc_split, mem_perc_split):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.set_title(f'{device} Performance Result', pad=20)
    ax1.set_xlabel('Time (in minutes)')

    width = 0.3

    # Plot the bar charts first
    bar_width = 0.4
    x = range(len(gpu_temp_split))

    ax2.bar([p + bar_width for p in x], gpu_perc_split, color='red', width=bar_width, label="GPU Utilization")
    ax2.bar(x, mem_perc_split, color='blue', width=bar_width, label="Memory Utilization")

    # Plot the area graph
    ax1.fill_between(x, gpu_temp_split, color='green', alpha=0.5, label="GPU Temperature")

    # Set x-axis labels.
    new_labels = range(5, 70, 5)  # This will create a list [5, 10, 15, ..., 60]
    ax1.set_xticks(range(len(new_labels)))
    ax1.set_xticklabels(new_labels)

    # Set y-axis for the right axis (percentages)
    percentage_ticks = range(10, 50, 5)
    ax2.set_yticks(percentage_ticks)
    ax2.set_ylim([0, 50])

    # Set y-axis for the left axis (temperature)
    temperature_ticks = range(60, 80, 2)
    ax1.set_yticks(temperature_ticks)
    ax1.set_ylim([60, 80])

    ax1.set_ylabel('Temperature (°C)')
    ax2.set_ylabel('Percentage (%)')

    gpu_temp_legend = mpatches.Patch(color='green', label='GPU Temperature')
    gpu_perc_legend = mpatches.Patch(color='red', label='GPU Utilization [%]')
    mem_perc_legend = mpatches.Patch(color='blue', label='Memory Utilization [%]')

    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', handles=[gpu_temp_legend, gpu_perc_legend, mem_perc_legend], frameon=False, ncol=3)

    plt.show()
# Call the function with the data
draw_graph_corrected('Personal Computer', gpu_temp_split, gpu_perc_split, mem_perc_split)
