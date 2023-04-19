import pandas as pd
import os

DATA_PATH = "breakdown"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,36" size 10cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style line 5 lt rgb "cyan" lw 3 pt 6
set style fill solid border -1

set output "plots/%s.pdf"
set ylabel "Normalized Latency" offset 1,0.5

set boxwidth 0.5
set xtics 6 offset 0,0.4
set tics front
set yrange [0:%d]
set xrange [-0.5:4.5]
set grid ytics
# set logscale y 10
# set key outside top center vertical maxrows 1 spacing -6 samplen 1.5

set ytics font "Times New Roman,36"
set xtics font "Times New Roman,24"
set key font "Times New Roman,36"

set style line 1 lc rgb '#e6f2ff'
set style line 2 lc rgb '#80bdff'
set style line 3 lc rgb '#007bff'
set style line 4 lc rgb '#003e80'

set margin 6.5,0.2,1.1,0.5

unset key
plot 'plots/%s.csv' using 3:2:xtic(1) with boxes ls 1
"""

# plot 'plots/%s.csv' using 2:xtic(1) title 'switch-base-128' ls 1, \
#         '' using 3 title 'switch-base-256' ls 2, \
#         '' using 4 title 'switch-large-128' ls 3, \
#         '' using 5 title 'switch-xxl-128' ls 4

df = pd.read_csv(
    "plots/%s.csv" % DATA_PATH,
)
print(df)
models = df["model"].unique()
df["latency"] = df["latency"] / 1000

df = df.pivot_table(
    index="breakdown",
    columns="model",
    values="latency",
).reset_index()
df = df.sort_values(by="switch-base-128", ascending=False)

print(df)

DATA_PATH = "breakdown_norm"

for model in models:
    df_model = df[["breakdown", model]]
    df_model["model"] = [0,1,2,3,4]
    print(df_model)
    df_model[model] = df_model[model] / df_model[model].max()
    df_model.to_csv("plots/%s.csv" % DATA_PATH, index=False, header=False, sep="\t")

    with open("plots/breakdown.gnuplot", "w") as f:
        f.write(GNU_SCRIPT % (DATA_PATH+ "_" + model, 
                              df_model[model].max() * 1.05,
                              DATA_PATH))

    os.system("gnuplot plots/breakdown.gnuplot")
