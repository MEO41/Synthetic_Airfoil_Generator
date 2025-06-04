import os
import pandas as pd
import numpy as np
from pathlib import Path
import neuralfoil as nf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_airfoil_coordinates(dat_file_path):
    """Load airfoil coordinates from .dat file"""
    try:
        # Try different common formats for airfoil .dat files
        coords = np.loadtxt(dat_file_path, skiprows=1)  # Skip header line
        return coords[:, 0], coords[:, 1]  # x, y coordinates
    except:
        try:
            coords = np.loadtxt(dat_file_path)  # No header
            return coords[:, 0], coords[:, 1]
        except Exception as e:
            print(f"Error loading {dat_file_path}: {e}")
            return None, None

def analyze_airfoil(x_coords, y_coords, reynolds_numbers, alpha_range=(-30, 30, 1)):
    """Analyze airfoil using NeuralFoil at different Reynolds numbers"""
    results = []
    
    # Create angle of attack range
    alphas = np.arange(alpha_range[0], alpha_range[1], alpha_range[2])
    
    for re_num in reynolds_numbers:
        try:
                        
            for alpha in alphas:
                try:
                    aero = nf.get_aero_from_coordinates(
                        coordinates=np.column_stack([x_coords, y_coords]),
                        alpha=alpha,
                        Re=re_num,
                        model_size="large"    # choose model complexity as desired
                    )
                    # aero is now a dict with keys 'CL','CD','CM', etc.
                    result = {
                        'reynolds_number': re_num,
                        'angle_of_attack': alpha,
                        'cl': aero['CL'],
                        'cd': aero['CD'],
                        'cm': aero['CM'],
                        'cl_cd_ratio': aero['CL'] / aero['CD'] if aero['CD'] != 0 else np.inf
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error analyzing at alpha={alpha}, Re={re_num}: {e}")
                    continue
        except Exception as e:
            print(f"Error creating airfoil or analyzing at Re={re_num}: {e}")
            continue
    
    return results

def process_airfoil_folder(folder_path, output_csv, reynolds_numbers=[100000, 500000, 1000000]):
    """Process all .dat files in folder and create dataset"""
    
    # Get all .dat files
    dat_files = list(Path(folder_path).glob("*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {folder_path}")
        return
    
    print(f"Found {len(dat_files)} .dat files")
    print(f"Reynolds numbers: {reynolds_numbers}")
    
    all_results = []
    
    # Process each airfoil file
    for dat_file in tqdm(dat_files, desc="Processing airfoils"):
        print(f"\nProcessing: {dat_file.name}")
        
        # Load airfoil coordinates
        x_coords, y_coords = load_airfoil_coordinates(dat_file)
        
        if x_coords is None or y_coords is None:
            print(f"Skipping {dat_file.name} - could not load coordinates")
            continue
        
        # Analyze airfoil
        airfoil_results = analyze_airfoil(x_coords, y_coords, reynolds_numbers)
        
        # Add airfoil name to each result
        for result in airfoil_results:
            result['airfoil_name'] = dat_file.stem  # filename without extension
            result['airfoil_file'] = dat_file.name
        
        all_results.extend(airfoil_results)
        print(f"Generated {len(airfoil_results)} data points for {dat_file.name}")
    
    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'airfoil_name', 'airfoil_file', 'reynolds_number', 
            'angle_of_attack', 'cl', 'cd', 'cm', 'cl_cd_ratio'
        ]
        df = df[column_order]
        
        # Sort by airfoil name, reynolds number, and angle of attack
        df = df.sort_values(['airfoil_name', 'reynolds_number', 'angle_of_attack'])
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nDataset saved to: {output_csv}")
        print(f"Total data points: {len(df)}")
        print(f"Unique airfoils: {df['airfoil_name'].nunique()}")
        
        # Display summary statistics
        print("\nDataset Summary:")
        print(df.groupby(['airfoil_name', 'reynolds_number']).size().head(10))
        
        return df
    else:
        print("No data generated. Check your .dat files and NeuralFoil installation.")
        return None

def main():
    # Configuration
    AIRFOILS_FOLDER = "generated_airfoils"  # Change this to your folder path
    OUTPUT_CSV = "generated_airfoil_dataset1.csv"
    REYNOLDS_NUMBERS = [1e5, 2.5e5, 5e5, 7.5e5, 1e6]  # 100k, 500k, 1M
    
    # Check if folder exists
    if not os.path.exists(AIRFOILS_FOLDER):
        print(f"Folder '{AIRFOILS_FOLDER}' not found!")
        print("Please make sure the airfoils folder exists and contains .dat files")
        return
    
    print("Starting airfoil dataset generation...")
    print(f"Input folder: {AIRFOILS_FOLDER}")
    print(f"Output file: {OUTPUT_CSV}")
    
    # Process airfoils and create dataset
    dataset = process_airfoil_folder(
        folder_path=AIRFOILS_FOLDER,
        output_csv=OUTPUT_CSV,
        reynolds_numbers=REYNOLDS_NUMBERS
    )
    
    if dataset is not None:
        print(f"\nSuccess! Dataset with {len(dataset)} records saved to {OUTPUT_CSV}")
    else:
        print("\nFailed to generate dataset.")

if __name__ == "__main__":
    main()