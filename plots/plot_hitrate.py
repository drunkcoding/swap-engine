import pandas as pd
import os

DATA_PATH = "recall"

GNU_SCRIPT = """
set terminal pdf font "Times New Roman,48" size 15cm, 10cm

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set style fill solid

set output "plots/%s.pdf"
set xlabel "Prefetch Algorithms"
set ylabel "Recall Rate" offset 2,0.5

set boxwidth 0.5
set xtics 6 offset 0,0.5
set yrange [0:1]
set xrange [-0.5:4.5]
set grid ytics
# set key outside top center vertical maxrows 1 spacing -6 samplen 1.5

set ytics font "Times New Roman,36"
set xtics font "Times New Roman,36"
set key font "Times New Roman,36"

set style line 1 lc rgb '#ffffff'
set style line 2 lc rgb '#e6f2ff'
set style line 3 lc rgb '#80bdff'
set style line 4 lc rgb '#007bff'
set style line 5 lc rgb '#003e80'

set margin 6.5,1,1,1

unset key
plot 'plots/%s.csv' every ::0::0 using 1:3:xtic(2) with boxes ls 1, \
    '' every ::1::1 using 1:3:xtic(2) with boxes ls 2, \
    '' every ::2::2 using 1:3:xtic(2) with boxes ls 3, \
    '' every ::3::3 using 1:3:xtic(2) with boxes ls 4, \
    '' every ::4::4 using 1:3:xtic(2) with boxes ls 5
"""

# plot '%s' using 4 title 'Stride' ls 1, '' using 3 title 'LFU' ls 2, '' using 2 title 'LRU' ls 3, '' using 1 title 'Archer' ls 4

with open("plots/miou.gnuplot", "w") as f:
    f.write(GNU_SCRIPT % (DATA_PATH, DATA_PATH))

os.system("gnuplot plots/miou.gnuplot")
