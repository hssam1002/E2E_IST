"""
Extract model information for debugging
"""

import os
import glob
import json
import re
import torch

def extract_model_info(model_dir):
    """
    Extract all relevant information from a model directory
    
    Args:
        model_dir (str): Directory containing model and log files
    
    Returns:
        dict: Model information
    """
    info = {
        'model_dir': model_dir,
        'log_file': None,
        'model_file': None,
        'architecture_from_log': {},
        'state_dict_keys': [],
        'state_dict_shapes': {},
        'model_architecture': {}
    }
    
    # Find log file
    log_files = glob.glob(os.path.join(model_dir, 'Log_*.log'))
    if log_files:
        info['log_file'] = log_files[0]
        try:
            with open(log_files[0], 'r') as f:
                log_content = f.read()
                
                # Extract architecture from log
                embed_match = re.search(r"'embed_dims':\s*\[([\d,\s]+)\]", log_content)
                depths_match = re.search(r"'depths':\s*\[([\d,\s]+)\]", log_content)
                heads_match = re.search(r"'num_heads':\s*\[([\d,\s]+)\]", log_content)
                
                if embed_match and depths_match and heads_match:
                    info['architecture_from_log'] = {
                        'embed_dims': embed_match.group(1).strip(),
                        'depths': depths_match.group(1).strip(),
                        'num_heads': heads_match.group(1).strip()
                    }
        except Exception as e:
            info['log_error'] = str(e)
    
    # Find model file - check multiple locations
    best_model = None
    # 1. In models subdirectory
    models_dir = os.path.join(model_dir, 'models')
    candidate = os.path.join(models_dir, 'best_model.pth')
    if os.path.exists(candidate):
        best_model = candidate
    else:
        # 2. Directly in model_dir
        candidate = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(candidate):
            best_model = candidate
        else:
            # 3. Look for any .pth file
            pth_files = glob.glob(os.path.join(model_dir, '**/*.pth'), recursive=True)
            if pth_files:
                best_model = pth_files[0]
                info['model_file_note'] = f"Found {len(pth_files)} .pth files, using first one"
    
    if best_model:
        info['model_file'] = best_model
        try:
            state_dict = torch.load(best_model, map_location='cpu')
            info['state_dict_keys'] = list(state_dict.keys())[:100]  # First 100 keys
            info['state_dict_total_keys'] = len(state_dict.keys())
            
            # Get shapes for attn_mask buffers
            attn_mask_info = {}
            for k, v in state_dict.items():
                if 'attn_mask' in k:
                    attn_mask_info[k] = {
                        'shape': list(v.shape),
                        'dtype': str(v.dtype)
                    }
            info['attn_mask_buffers'] = attn_mask_info
            info['num_attn_mask_buffers'] = len(attn_mask_info)
            
            # Get encoder/decoder architecture info from weights
            encoder_info = {}
            decoder_info = {}
            
            # Encoder patch embed
            if 'encoder.patch_embed.proj.weight' in state_dict:
                encoder_info['patch_embed_dim'] = int(state_dict['encoder.patch_embed.proj.weight'].shape[0])
            
            # Encoder layers
            encoder_layers = {}
            for i in range(4):  # 4 stages
                layer_key = f'encoder.layers.{i}.blocks.0.norm1.weight'
                if layer_key in state_dict:
                    encoder_layers[f'layer_{i}_dim'] = int(state_dict[layer_key].shape[0])
            encoder_info['layers'] = encoder_layers
            
            # Decoder layers
            decoder_layers = {}
            for i in range(4):  # 4 stages
                layer_key = f'decoder.layers.{i}.blocks.0.norm1.weight'
                if layer_key in state_dict:
                    decoder_layers[f'layer_{i}_dim'] = int(state_dict[layer_key].shape[0])
            decoder_info['layers'] = decoder_layers
            
            info['encoder_from_weights'] = encoder_info
            info['decoder_from_weights'] = decoder_info
            
            # Get some key weight shapes
            key_shapes = {}
            key_names = [
                'encoder.patch_embed.proj.weight',
                'encoder.layers.0.blocks.0.norm1.weight',
                'encoder.layers.1.blocks.0.norm1.weight',
                'encoder.layers.2.blocks.0.norm1.weight',
                'encoder.layers.3.blocks.0.norm1.weight',
                'decoder.layers.0.blocks.0.norm1.weight',
                'decoder.layers.1.blocks.0.norm1.weight',
                'decoder.layers.2.blocks.0.norm1.weight',
                'decoder.layers.3.blocks.0.norm1.weight'
            ]
            for k in key_names:
                if k in state_dict:
                    key_shapes[k] = list(state_dict[k].shape)
            info['key_weight_shapes'] = key_shapes
            
        except Exception as e:
            import traceback
            info['model_load_error'] = str(e)
            info['model_load_traceback'] = traceback.format_exc()
    
    return info


def main():
    """Extract information from all models"""
    results_dir = "/data4/hongsik/E2E_IST/results"
    model_pattern = os.path.join(results_dir, "**/best_model.pth")
    model_files = glob.glob(model_pattern, recursive=True)
    model_files = [f for f in model_files if 'OLD' not in f]
    
    all_info = []
    for model_path in model_files:
        model_dir = os.path.dirname(model_path)
        parent_dir = os.path.dirname(model_dir)  # Parent directory with log
        
        info = extract_model_info(parent_dir)
        all_info.append(info)
    
    # Save to JSON
    output_path = os.path.join(results_dir, 'model_debug_info.json')
    with open(output_path, 'w') as f:
        json.dump(all_info, f, indent=2)
    
    print(f"Debug information saved to: {output_path}")
    print(f"Total models: {len(all_info)}")
    
    # Also create info for specific model
    specific_model = "results/DIV2K_SNR-10to10_rand_mask_2/20251230_163136"
    if os.path.exists(specific_model):
        specific_info = extract_model_info(specific_model)
        specific_output = os.path.join(results_dir, 'model_debug_info_specific.json')
        with open(specific_output, 'w') as f:
            json.dump(specific_info, f, indent=2)
        print(f"Specific model info saved to: {specific_output}")


if __name__ == '__main__':
    main()

