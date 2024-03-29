import pandas as pd
import os

DATA_PATH = "plots/e2e_lat_cpu.csv"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,48" size 15cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style line 5 lt rgb "cyan" lw 3 pt 6
set style fill solid border -1

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
# set key outside top center vertical maxrows 2 spacing 1 samplen 1.5
# set key at 5,5

# set xticks font 36
set ytics font "Times New Roman,36"
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"

set style line 1 lc rgb '#e6f2ff'
set style line 2 lc rgb '#80bdff'
set style line 3 lc rgb '#007bff'
set style line 4 lc rgb '#003e80'

set margin 6.5,1,1,1
unset key
plot '%s.csv' using 3:xtic(%d) title 'DS(SSD)' ls 1, '' using 4 title 'DS(CPU)' ls 2, '' using 5 title 'CUDA-UM(CPU)' ls 3, '' using 6 title 'Archer(CPU+SSD)' ls 4
"""

df = pd.read_csv(DATA_PATH, sep=",")
models = df["Model"].unique()

# devide all values by 1000
df["DS(SSD)"] = df["DS(SSD)"] / 1000
df["DS(CPU)"] = df["DS(CPU)"] / 1000
df["CUDA-UM(CPU)"] = df["CUDA-UM(CPU)"] / 1000
df["Archer(CPU+SSD)"] = df["Archer(CPU+SSD)"] / 1000

# drop column BMINf
df = df.drop(columns=["BMInf(CPU)"])
df = df.fillna(0)
# print(df)

# We first plot e2e latency on single GPU

df_single = df[df["GPU"] == 1]
print(df_single)
df_single.to_csv("plots/e2e_lat_cpu_single.csv", index=False, header=False)

for model in models:

    name = f"e2e_lat_cpu_single_{model}"

    df_single_model = df_single[df_single["Model"] == model]
    df_single_model.to_csv(f"plots/{name}.csv", index=False, header=False, sep="\t")

    with open("plots/e2e_lat_cpu_single.gnuplot", "w") as f:
        f.write(
            GNU_SCRIPT
            % (
                f"plots/{name}",
                df_single_model["DS(SSD)"].max() * 1.05,
                2,
                f"plots/{name}",
                7,
            )
        )

    os.system("gnuplot plots/e2e_lat_cpu_single.gnuplot")
