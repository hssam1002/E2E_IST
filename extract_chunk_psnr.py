#!/usr/bin/env python3
"""
Extract chunk-by-chunk PSNR data from log files and create visualization.
Supports both GNUPlot data export and matplotlib plotting.
"""

import re
import os
import glob

def extract_psnr_from_log(log_file):
    """
    Extract chunk-by-chunk PSNR data from log file.
    
    Args:
        log_file (str): Path to log file
    
    Returns:
        dict: {'chunk': [1, 2, ...], 'psnr': [psnr1, psnr2, ...], 'mode': mode_name}
    """
    chunks = []
    psnrs = []
    mode = None
    in_sequential = False
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract mode name
            if 'Testing Mode:' in line:
                match = re.search(r'Testing Mode: (\w+)', line)
                if match:
                    mode = match.group(1)
            
            # For off mode, look for "Strategy 1: Sequential" section
            if mode == 'off':
                # Check if we're in Sequential section
                if 'Strategy 1: Sequential' in line:
                    in_sequential = True
                    continue
                elif 'Strategy 2: Variance' in line:
                    in_sequential = False
                    break  # Stop after sequential section
                
                # Extract from Sequential section only
                # Format: "Chunk  1: PSNR=14.67 dB, MS-SSIM=0.8068"
                if in_sequential:
                    match = re.search(r'Chunk\s+(\d+):\s+PSNR=([\d.]+)\s*dB', line)
                    if match:
                        chunk_num = int(match.group(1))
                        psnr = float(match.group(2))
                        chunks.append(chunk_num)
                        psnrs.append(psnr)
            else:
                # For other modes: "Chunk  1/10: PSNR=27.56 dB, MS-SSIM=0.0555"
                match = re.search(r'Chunk\s+(\d+)[/\|].*?PSNR[=\s]+([\d.]+)\s*dB', line)
                if match:
                    chunk_num = int(match.group(1))
                    psnr = float(match.group(2))
                    chunks.append(chunk_num)
                    psnrs.append(psnr)
    
    return {
        'chunk': chunks,
        'psnr': psnrs,
        'mode': mode
    }


def extract_all_results(results_dir='/data4/hongsik/E2E_IST/results'):
    """
    Extract PSNR data from all test log files.
    
    Args:
        results_dir (str): Base results directory
    
    Returns:
        dict: {mode_name: {'chunk': [...], 'psnr': [...]}}
    """
    all_results = {}
    
    # Find all test log files
    log_pattern = os.path.join(results_dir, 'test_Kodak_*_packet32/*/Log_*.log')
    log_files = glob.glob(log_pattern)
    
    for log_file in sorted(log_files):
        data = extract_psnr_from_log(log_file)
        if data['mode'] and data['chunk']:
            mode_name = data['mode']
            # Handle off mode separately (use sequential)
            if mode_name == 'off':
                mode_name = 'off_sequential'
            
            all_results[mode_name] = {
                'chunk': data['chunk'],
                'psnr': data['psnr']
            }
    
    return all_results


def save_data_for_gnuplot(results, output_dir='./plot_data'):
    """
    Save extracted data in GNUPlot-friendly format.
    
    Args:
        results (dict): Extracted results
        output_dir (str): Output directory for data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual files for each mode
    for mode_name, data in results.items():
        output_file = os.path.join(output_dir, f'{mode_name}.dat')
        with open(output_file, 'w') as f:
            f.write('# Chunk PSNR(dB)\n')
            for chunk, psnr in zip(data['chunk'], data['psnr']):
                f.write(f'{chunk} {psnr:.2f}\n')
        print(f"Saved: {output_file}")
    
    # Save combined file (all modes)
    combined_file = os.path.join(output_dir, 'all_modes.dat')
    max_chunks = max(len(data['chunk']) for data in results.values())
    
    with open(combined_file, 'w') as f:
        # Header
        header = '# Chunk'
        for mode_name in sorted(results.keys()):
            header += f' {mode_name}'
        f.write(header + '\n')
        
        # Data rows
        for chunk_idx in range(max_chunks):
            row = f'{chunk_idx + 1}'
            for mode_name in sorted(results.keys()):
                data = results[mode_name]
                if chunk_idx < len(data['chunk']):
                    row += f' {data["psnr"][chunk_idx]:.2f}'
                else:
                    row += ' -'
            f.write(row + '\n')
    
    print(f"Saved combined file: {combined_file}")




if __name__ == '__main__':
    print("Extracting chunk-by-chunk PSNR data from log files...")
    results = extract_all_results()
    
    if not results:
        print("No results found! Check log file paths.")
    else:
        print(f"\nFound {len(results)} modes:")
        for mode_name in sorted(results.keys()):
            print(f"  - {mode_name}: {len(results[mode_name]['chunk'])} chunks")
        
        print("\nSaving data files for GNUPlot...")
        save_data_for_gnuplot(results)
        
        print("\nDone! Data files saved in ./plot_data/")
        print("Now run: gnuplot plot_chunk_psnr.gp")

