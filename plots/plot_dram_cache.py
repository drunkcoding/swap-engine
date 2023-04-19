import pandas as pd
import os

DATA_PATH = "plots/dram_cache.csv"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,36" size 15cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style line 5 lt rgb "cyan" lw 3 pt 6
set style fill solid border -1

set output "%s.pdf"
# set xlabel "#GPUs"
set ylabel "Normalized Latency" offset 2,0.5 


# set boxwidth 0.5
set xtics 6 offset 1,5 rotate by 90
set tics front
set yrange [0:%d]
set xrange [-0.5:3.5]

set ytics font "Times New Roman,36"
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"
set key outside top center vertical maxrows 1 spacing 0 samplen 1.5 width 3

set style line 1 lc rgb '#e6f2ff'
set style line 2 lc rgb '#80bdff'
set style line 3 lc rgb '#007bff'
set style line 4 lc rgb '#003e80'

set margin 6.5,1,.5,1
plot '%s.csv' using 2:xtic(%d) title 'SSD' ls 1, '' using 3 title '+DRAM' ls 2
"""

df = pd.read_csv(DATA_PATH, sep=",")
models = df["Model"].unique()

df["+DRAM-Cache"] = df["+DRAM-Cache"] / df["SSD-Only"]
name = f"dram_cache_norm"

df.to_csv(f"plots/{name}.csv", index=False, header=False, sep="\t")

with open(f"plots/{name}.gnuplot", "w") as f:
    f.write(
        GNU_SCRIPT
        % (
            f"plots/{name}",
            1.05,
            f"plots/{name}",
            1,
        )
    )

os.system(f"gnuplot plots/{name}.gnuplot")
    
