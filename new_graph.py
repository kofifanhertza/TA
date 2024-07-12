import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the CSV file
file_path = 'WIN_8H.csv'
data = pd.read_csv(file_path)

# Convert the timestamp to a datetime object
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Resample the data to hourly averages
data.set_index('Timestamp', inplace=True)
hourly_data = data.resample('H').mean()

# Drop the last hour to ensure only 8 hours are plotted
hourly_data = hourly_data.head(8)

# Prepare the data for plotting
gpu_temp_split = hourly_data['GPU Temperature'].tolist()
gpu_perc_split = (hourly_data['GPU Memory']/1600*100).tolist()
mem_perc_split = hourly_data['RAM Percentage'].tolist()

df_gpu_temp = pd.DataFrame({'Value': gpu_temp_split})
df_gpu_perc = pd.DataFrame({'Value': gpu_perc_split})
df_mem_perc = pd.DataFrame({'Value': mem_perc_split}) 

print(df_gpu_temp.mean())
print(df_gpu_perc.mean())
print(df_mem_perc.mean())


def draw_graph_corrected(device, gpu_temp_split, gpu_perc_split, mem_perc_split):
    fig, ax1 = plt.subplots(figsize =(10,10))

    ax2 = ax1.twinx()

    ax1.set_title(f'{device} Performance Result', pad=20)
    ax1.set_xlabel('Time (in hours)')


    # Plot the bar charts first
    bar_width = 0.3
    x = range(len(gpu_temp_split))

    ax2.bar([p + bar_width for p in x], gpu_perc_split, color='red', width=bar_width, label="GPU Utilization")
    ax2.bar(x, mem_perc_split, color='blue', width=bar_width, label="Memory Utilization")

    # Plot the area graph
    ax1.fill_between(x, gpu_temp_split, color='green', alpha=0.5, label="GPU Temperature")

    # Set x-axis labels.
    new_labels = range(1, len(gpu_temp_split) + 1)  # This will create labels for each hour
    ax1.set_xticks(range(len(new_labels)))
    ax1.set_xticklabels(new_labels)

    # Set y-axis for the right axis (percentages)
    percentage_ticks = range(0, 55, 5)
    ax2.set_yticks(percentage_ticks)
    ax2.set_ylim([0, 50])

    # Set y-axis for the left axis (temperature)
    temperature_ticks = range(60, 70, 1)
    ax1.set_yticks(temperature_ticks)
    ax1.set_ylim([60, 70])

    ax1.set_ylabel('Temperature (°C)')
    ax2.set_ylabel('Percentage (%)')

    gpu_temp_legend = mpatches.Patch(color='green', label='GPU Temperature')
    gpu_perc_legend = mpatches.Patch(color='red', label='GPU Utilization [%]')
    mem_perc_legend = mpatches.Patch(color='blue', label='Memory Utilization [%]')

    plt.legend(handles=[gpu_temp_legend, gpu_perc_legend, mem_perc_legend], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plt.show()

# Call the function with the resampled data
draw_graph_corrected('Personal Computer', gpu_temp_split, gpu_perc_split, mem_perc_split)
