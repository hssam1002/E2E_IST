#!/usr/bin/bin/gnuplot
# GNUPlot script for SNR vs Performance plots
# Usage: gnuplot -e "data_dir='plot_data'" plot_snr_results.gp

# Get data directory from command line or use default
if (!exists("data_dir")) data_dir = "plot_data"
if (!exists("output_dir")) output_dir = "plots"

# Create output directory
system(sprintf("mkdir -p %s", output_dir))

# Set terminal
set terminal pngcairo enhanced color font 'Arial,12' size 1200,800

# PSNR plot
set output sprintf("%s/snr_vs_psnr.png", output_dir)
set title "SNR vs PSNR"
set xlabel "SNR (dB)"
set ylabel "PSNR (dB)"
set grid
set key top left

# Get list of SNR files and plot
snr_files = system(sprintf("ls %s/snr_*.dat 2>/dev/null", data_dir))
plot for [file in snr_files] file using 1:2 with linespoints \
     title system(sprintf("basename %s .dat | sed 's/snr_//'", file))

# MS-SSIM plot
set output sprintf("%s/snr_vs_ssim.png", output_dir)
set title "SNR vs MS-SSIM"
set ylabel "MS-SSIM"
set yrange [0:1]

plot for [file in snr_files] file using 1:3 with linespoints \
     title system(sprintf("basename %s .dat | sed 's/snr_//'", file))

# LPIPS plot
set output sprintf("%s/snr_vs_lpips.png", output_dir)
set title "SNR vs LPIPS"
set ylabel "LPIPS"
unset yrange

plot for [file in snr_files] file using 1:4 with linespoints \
     title system(sprintf("basename %s .dat | sed 's/snr_//'", file))

print "SNR plots saved to ", output_dir
