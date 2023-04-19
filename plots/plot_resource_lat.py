import pandas as pd
import os

DATA_PATH = "plots/e2e_multi.csv"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,48" size 15cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set style fill solid

set output "%s.pdf"
set xlabel "#GPUs" offset 0,1
set ylabel "Latency (s)" offset 2.5,0.5
# set xtics rotate by -5

# set boxwidth 0.9
set xtics 8 offset 0,0.5
set xrange [-0.5:3.5]
set grid ytics
# set yrange [0:3000]
# set ytics 0,200,3000
set key outside top center vertical maxrows 1 spacing -6 samplen 1.5
# set key at 0,0

# set xticks font 36
set ytics font "Times New Roman,36" offset 0.5,0
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"

set style line 1 lc rgb '#e6f2ff'
set style line 2 lc rgb '#80bdff'
set style line 3 lc rgb '#007bff'
set style line 4 lc rgb '#003e80'

set margin 5,1,2,0.5
unset key
set arrow from graph 0, first %s to graph 1, first %s nohead lt 1 lw 5 lc "#003e80" front
plot '%s.csv' using 2:xtic(1) title 'DS(SSD)' ls 1, '' using 3 title 'DS(CPU)' ls 2
"""

df_multi = pd.read_csv("plots/e2e_multi.csv", sep=",")
models = df_multi["Model"].unique()

for model in models:
    name = f"e2e_resource_{model}"

    df_multi_model = df_multi[df_multi["Model"] == model]
    df_multi_model = df_multi_model.drop(columns=["Model"])
    df_multi_model.to_csv(f"plots/{name}.csv", index=False, header=False, sep="\t")

    with open("plots/e2e_resource.gnuplot", "w") as f:
        f.write(
            GNU_SCRIPT
            % tuple(
                [
                    f"plots/{name}",
                    df_multi_model["Archer(CPU+SSD)"].to_numpy()[0],
                    df_multi_model["Archer(CPU+SSD)"].to_numpy()[0],
                    f"plots/{name}",
                ]
            )
        )

    os.system("gnuplot plots/e2e_resource.gnuplot")
