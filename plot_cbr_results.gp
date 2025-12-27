#!/usr/bin/bin/gnuplot
# GNUPlot script for CBR vs Performance plots
# Usage: gnuplot -e "data_dir='plot_data'" plot_cbr_results.gp

# Get data directory from command line or use default
if (!exists("data_dir")) data_dir = "plot_data"
if (!exists("output_dir")) output_dir = "plots"

# Create output directory
system(sprintf("mkdir -p %s", output_dir))

# Set terminal
set terminal pngcairo enhanced color font 'Arial,12' size 1200,800

# PSNR plot
set output sprintf("%s/cbr_vs_psnr.png", output_dir)
set title "CBR vs PSNR"
set xlabel "Channel Bandwidth Ratio (CBR)"
set ylabel "PSNR (dB)"
set grid
set key top left
set xrange [0:0.16]

# Get list of files and plot
files = system(sprintf("ls %s/*.dat 2>/dev/null | grep -v snr_", data_dir))
plot for [file in files] file using 1:2 with linespoints \
     title system(sprintf("basename %s .dat", file))

# MS-SSIM plot
set output sprintf("%s/cbr_vs_ssim.png", output_dir)
set title "CBR vs MS-SSIM"
set ylabel "MS-SSIM"
set yrange [0:1]

plot for [file in files] file using 1:3 with linespoints \
     title system(sprintf("basename %s .dat", file))

# LPIPS plot
set output sprintf("%s/cbr_vs_lpips.png", output_dir)
set title "CBR vs LPIPS"
set ylabel "LPIPS"
unset yrange

plot for [file in files] file using 1:4 with linespoints \
     title system(sprintf("basename %s .dat", file))

print "CBR plots saved to ", output_dir
