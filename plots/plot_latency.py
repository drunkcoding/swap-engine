import pandas as pd
import os

DATA_PATH = "plots/e2e_lat.csv"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,48" size 15cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set style fill solid

set output "%s.pdf"
set xlabel "#GPUs"
set ylabel "Latency (s)" offset 2,0.5
# set xtics rotate by -5

# set boxwidth 0.9
set xtics 6 offset 0,0.5
set yrange [0:%d]
set xrange [-0.5:%d.5]
set grid ytics
# set ytics 0,200,3000
set key outside top center vertical maxrows 1 spacing -6 samplen 1.5
# set key at 0,0

# set xticks font 36
set ytics font "Times New Roman,36"
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"

set style line 1 lc rgb '#afd5ff'
set style line 2 lc rgb '#003671'

set margin 6.5,1,1,1

plot '%s.csv' using 3:xtic(%d) title 'DeepSpeed' ls 1, '' using 4 title 'Archer' ls 2
"""

df = pd.read_csv(DATA_PATH, sep=",")
models = df["Model"].unique()
df["DeepSpeed"] = df["DeepSpeed"] / 1000
df["Archer"] = df["Archer"] / 1000
# print(df)

# We first plot e2e latency on single GPU

df_single = df[df["GPU"] == 1]
df_single.to_csv("plots/e2e_lat_single.csv", index=False, header=False)

for model in models:

    name = f"e2e_lat_single_{model}"

    df_single_model = df_single[df_single["Model"] == model]
    df_single_model.to_csv(f"plots/{name}.csv", index=False, header=False, sep="\t")

    with open("plots/e2e_lat_single.gnuplot", "w") as f:
        f.write(
            GNU_SCRIPT
            % (
                f"plots/{name}",
                df_single_model["DeepSpeed"].max() * 1.05,
                2,
                f"plots/{name}",
                5,
            )
        )

    os.system("gnuplot plots/e2e_lat_single.gnuplot")
