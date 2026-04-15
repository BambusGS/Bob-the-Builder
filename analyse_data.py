import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv



def compute_analytics(script_name:str):
    data_dir = f"output/{script_name}/"
    gen_df = pd.read_csv(data_dir + "general_performance.csv")
    jacobi_df = pd.read_csv(data_dir + "jacobi_performance.csv")

    jacobi_perf = jacobi_df['compute_time'].mean()
    mean_sim_time = gen_df["t_sim"].mean()
    wall_time = sum(gen_df['t_sim'] + gen_df['t_load'] + gen_df['t_disp'])
    return jacobi_perf, mean_sim_time, wall_time



def analyse_all():
    with open('script_list.csv', newline='') as file:
        reader = csv.reader(file)
        script_name = list(reader)[0]
    
    analesys_resoult = np.empty([len(script_name),3])
    header = ['jacobi_perf', 'mean_sim_time', 'wall_time']
    i = 0
    for name_of_experiment in script_name:
        analesys_resoult[i, :] = compute_analytics(name_of_experiment)
        i += 1
    df = pd.DataFrame(analesys_resoult, columns=header)
    df["experiment"] = script_name
    df = df.set_index("experiment")
    return df

def bar_plot_all(df:pd.DataFrame):
    for x in df.columns:
        df = df.sort_values(by=x, ascending=False)
        plt.figure()
        plt.bar(df.index, df[x])
        plt.xlabel("Experiment")
        plt.ylabel('time[s]')
        plt.title(x)
        plt.xticks(rotation=45)  
        plt.tight_layout()
        plt.show()






df = analyse_all()
#bar_plot_all(df)




import platform
import psutil
"""
script name, OS-release, node, machine, CPU, p_cores, t_cores, OS.


CPU usage per core, ram usage, current freq
add computer hardware information, CPU, GPU, total RAM usage (if possible), OS"""



def get_system_info():
    info = {}

    uname = platform.uname()
    info['OS'] = uname.system + uname.release
    info['node'] = uname.node
    if uname.processor == '':
        info['CPU'] = 'none'
    else:
        info["CPU"] = uname.processor
    info['machine'] = uname.machine

    info['p_cores'] = psutil.cpu_count(logical=False)
    info['t_cores'] = psutil.cpu_count(logical=True)


    # CPU cache (approximate via psutil if available)
    try:
        cpu_freq = psutil.cpu_freq()
        info["CPU Frequency"] = f"{cpu_freq.current:.2f} MHz"
    except:
        info["CPU Frequency"] = "Unknown"


    # Format into string
    result = (
        f"OS:{info['OS']},"
        f'node:{info['node']},'
        f'machine:{info['machine']},'
        f"CPU: {info['CPU']}, "
        f'p_threads: {info['p_cores']},'
        f't_threads: {info['t_cores']}'
        )

    return result


print(get_system_info())