#!/usr/bin/env gnuplot
# GNUPlot script to visualize chunk-by-chunk PSNR performance
# Usage: gnuplot plot_chunk_psnr.gp

# Set output format (can be changed to png, pdf, eps, etc.)
set terminal pngcairo enhanced color font 'Arial,12' size 1200,800
set output 'chunk_psnr_comparison.png'

# Set labels and title
set xlabel 'Transmitted Chunk Number' font 'Arial,14'
set ylabel 'PSNR (dB)' font 'Arial,14'
set title 'PSNR vs Transmitted Chunk Number (Packet Size = 32)' font 'Arial,16'

# Set grid
set grid linetype 1 linecolor rgb '#cccccc' linewidth 0.5
set grid linetype 2 linecolor rgb '#cccccc' linewidth 0.5

# Set key (legend) position
set key top left font 'Arial,11'
set key box linestyle 1

# Set axis ranges (optional, uncomment to set manually)
# set xrange [1:10]
# set yrange [10:45]

# Define line styles for each mode
set style line 1 lc rgb '#1f77b4' lw 2 pt 7 ps 1.2  # blue - off
set style line 2 lc rgb '#ff7f0e' lw 2 pt 9 ps 1.2  # orange - alm
set style line 3 lc rgb '#2ca02c' lw 2 pt 5 ps 1.2  # green - mrl
set style line 4 lc rgb '#d62728' lw 2 pt 13 ps 1.2 # red - rand_mask_1
set style line 5 lc rgb '#9467bd' lw 2 pt 11 ps 1.2 # purple - rand_mask_2

# Plot data files (check if files exist)
plot \
    'plot_data/off_sequential.dat' using 1:2 with linespoints ls 1 title 'Off (Sequential)', \
    'plot_data/alm.dat' using 1:2 with linespoints ls 2 title 'ALM (Exponential)', \
    'plot_data/mrl.dat' using 1:2 with linespoints ls 3 title 'MRL', \
    'plot_data/rand_mask_1.dat' using 1:2 with linespoints ls 4 title 'Rand Mask 1', \
    'plot_data/rand_mask_2.dat' using 1:2 with linespoints ls 5 title 'Rand Mask 2'

# Alternative: Plot from combined file (uncomment to use)
# plot 'plot_data/all_modes.dat' using 1:2 with linespoints ls 1 title 'Off', \
#      '' using 1:3 with linespoints ls 2 title 'ALM', \
#      '' using 1:4 with linespoints ls 3 title 'MRL', \
#      '' using 1:5 with linespoints ls 4 title 'Rand Mask 1', \
#      '' using 1:6 with linespoints ls 5 title 'Rand Mask 2'

