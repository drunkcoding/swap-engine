import pandas as pd
import os

DATA_PATH = "plots/prefetch_bucket_size.csv"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,48" size 30cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style line 5 lt rgb "cyan" lw 3 pt 6
set style fill solid border -1

set output "plots/%s.pdf"
set xlabel "#GPUs"
set ylabel "Latency (s)" offset 2,0.5
# set xtics rotate by -5

# set boxwidth 0.9
set xtics 6 offset 0,0.5
set yrange [0:%s]
set xrange [-0.5:3.5]
set grid ytics
# set ytics 0,200,3000
set key outside top center vertical maxrows 1 spacing 3 samplen 1.5
set key at 1,1.35

# set xticks font 36
set ytics font "Times New Roman,36"
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"

set style line 1 lc rgb '#e6f2ff'
set style line 2 lc rgb '#80bdff'
set style line 3 lc rgb '#007bff'
set style line 4 lc rgb '#003e80'

set margin 6.5,1,1,1
# unset key
plot 'plots/%s.csv' using 2:xtic(1) title '1GB' ls 1, '' using 3 title '2GB' ls 2, '' using 4 title '4GB' ls 3, '' using 5 title '8GB' ls 4
"""

def normalize(series):
    # min_val = series.min()
    max_val = series.max()
    return series / max_val
    # return (series - min_val) / (max_val - min_val)

# Name,Size,Latency
# switch-base-128,1,668
# switch-base-128,2,358

df = pd.read_csv(DATA_PATH, sep=",")
df["Latency"] = df["Latency"] / 1000  # to seconds

models = df["Name"].unique()

df['Latency'] = df.groupby('Name', group_keys=False)['Latency'].apply(normalize)

max_latency = df["Latency"].max()

# pivot table on Size
name = "bucket_size"

df = df.pivot_table(index=["Name"], columns="Size", values="Latency")
df.to_csv("plots/bucket_size.csv", sep="\t", index=True, header=False)

with open("plots/bucket_single.gnuplot", "w") as f:
    f.write(GNU_SCRIPT % (name, max_latency * 1.05, name))

os.system("gnuplot plots/bucket_single.gnuplot")



# for model in models:

#     name = f"bucket_single_{model}"

#     df_single_model = df[df["Name"] == model]
#     max_latency = df_single_model["Latency"].max()
#     df_single_model = df_single_model.pivot_table(
#         index=["Name"], columns="Size", values="Latency"
#     )
#     df_single_model.to_csv(f"plots/{name}.csv", index=True, header=False, sep="\t")

#     with open("plots/bucket_single.gnuplot", "w") as f:
#         f.write(GNU_SCRIPT % (name, max_latency * 1.05, name))

#     os.system("gnuplot plots/bucket_single.gnuplot")

