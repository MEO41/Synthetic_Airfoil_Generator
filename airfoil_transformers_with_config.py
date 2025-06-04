import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Centralized configuration for VAE Airfoil Generator"""
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 1. DATA CONFIGURATION
    # ──────────────────────────────────────────────────────────────────────────────
    DATA_FOLDER         = 'airfoils'       # Path to folder containing .dat files
    TARGET_POINTS       = 50             # Number of (x,y) samples per airfoil (100 is usually enough)
    TRAIN_TEST_SPLIT    = 0.20             # 80/20 train/validation
    RANDOM_SEED         = 42               # For reproducibility (NumPy, PyTorch, etc.)

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. MODEL ARCHITECTURE
    # ──────────────────────────────────────────────────────────────────────────────
    LATENT_DIM          = 48               # 48–64 dims gives good capacity on ~1,600 airfoils
    HIDDEN_DIM          = 256              # Hidden‐layer size in MLP encoder/decoder
    DROPOUT_RATE        = 0.2              # Dropout to regularize and avoid overfitting

    # If you want to try a small Conv1D front‐end, you could set a flag here:
    USE_CONV_FRONTEND   = False            # Set True if you add Conv1D layers to encoder

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. TRAINING PARAMETERS
    # ──────────────────────────────────────────────────────────────────────────────
    EPOCHS              = 1000             # Train up to 1000 epochs; early stopping will usually stop earlier
    BATCH_SIZE          = 64               # ~26 updates/epoch; balances stability and GPU usage
    LEARNING_RATE       = 5e-4             # 5e-4 with a scheduler is a good “medium‐fast” LR
    BETA                = 1.0              # Start at β=1.0 to encourage wider latent spread
    PATIENCE            = 100              # Wait 100 epochs of no‐improvement before stopping

    # ──────────────────────────────────────────────────────────────────────────────
    # 4. LR SCHEDULER
    # ──────────────────────────────────────────────────────────────────────────────
    SCHEDULER_PATIENCE  = 10               # When val_loss hasn’t improved for 10 epochs → LR *= SCHEDULER_FACTOR
    SCHEDULER_FACTOR    = 0.5              # Halve the LR when plateau is detected

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. GENERATION PARAMETERS
    # ──────────────────────────────────────────────────────────────────────────────
    DEFAULT_TEMPERATURE   = 1.5            # Typical “temp” to sample from N(0,1)*1.5
    NUM_GENERATED_AIRFOILS = 6           # Generate a large pool (e.g. 100) so you can filter & pick top k
    #   (You can later down‐select to PLOT_AIRFOILS_COUNT or whatever you need.)

    DIVERSITY_METHOD     = 'mixed'         # Combined strategy: 'temperature', 'spherical', 'clustered', or 'mixed'
    # If you prefer to emphasize “k‐means” clustering on real μ’s, you can set DIVERSITY_METHOD='clustered'

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. TEMPERATURE SAMPLING RANGE (for the 'mixed' or 'temperature' methods)
    # ──────────────────────────────────────────────────────────────────────────────
    TEMP_MIN             = 0.8             # Avoid sampling too close to origin (overly “average” shapes)
    TEMP_MAX             = 2.5             # Avoid sampling too far out (too many invalid shapes)
    # You can grid‐search among a few ranges, e.g., (0.7,2.2), (0.8,2.5), (1.0,3.0) to see which yields highest diversity.

    # ──────────────────────────────────────────────────────────────────────────────
    # 7. SPHERICAL SAMPLING (for the 'mixed' or 'spherical' methods)
    # ──────────────────────────────────────────────────────────────────────────────
    SPHERICAL_RADIUS_MIN = 1.5             # Minimum radial distance from origin in latent space
    SPHERICAL_RADIUS_MAX = 3.0             # Maximum radial distance
    # Sampling: z = normalize(randn(latent_dim)) * uniform(SPHERICAL_RADIUS_MIN, SPHERICAL_RADIUS_MAX)

    # ──────────────────────────────────────────────────────────────────────────────
    # 8. HIGH‐VARIANCE (GAUSSIAN) SAMPLING (for the 'mixed' or 'clustered' methods)
    # ──────────────────────────────────────────────────────────────────────────────
    HIGH_VAR_STD         = 2.0             # Standard deviation when sampling z ∼ N(0, HIGH_VAR_STD²)

    # ──────────────────────────────────────────────────────────────────────────────
    # 9. QUALITY VALIDATION THRESHOLDS
    # ──────────────────────────────────────────────────────────────────────────────
    MIN_X_RANGE          = 0.5             # Require chordwise span ≥ 0.5 before deeming “valid”
    MIN_Y_RANGE          = 0.03            # Slightly lower than 0.05, since some thin foils might only be 0.03c
    MAX_JUMP_THRESHOLD   = 0.5             # No two adjacent points should jump more than 0.5 (in normalized units)
    # You can also add a monotonicity check on x: ensure np.all(np.diff(xs) >= -1e-6) in validate().

    # ──────────────────────────────────────────────────────────────────────────────
    # 10. DIVERSITY ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────────
    DIVERSITY_SAMPLE_SIZE   = 200          # Compare generated → original against a random sample of 200 originals
    NOVELTY_THRESHOLD_RATIO = 0.10         # A generated airfoil is “novel” if its nearest‐neighbor distance > 10% of orig_mean_dist

    # ──────────────────────────────────────────────────────────────────────────────
    # 11. OUTPUT SETTINGS
    # ──────────────────────────────────────────────────────────────────────────────
    OUTPUT_FOLDER       = 'generated_airfoils'       # Where to save .dat files
    MODEL_SAVE_PATH     = 'best_airfoil_model.pt'    # Where to checkpoint the best VAE
    PLOT_AIRFOILS_COUNT = 6                         # How many to visualize (subplot grid: 2×3)
    INTERPOLATION_STEPS = 6                         # Number of steps when showing latent interpolation

    # ──────────────────────────────────────────────────────────────────────────────
    # 12. DEVICE SELECTION
    # ──────────────────────────────────────────────────────────────────────────────
    USE_CUDA              = True            # If False, force torch.device('cpu')
    # You can override at runtime: torch.device('cuda' if USE_CUDA and cuda_available else 'cpu')

    # ──────────────────────────────────────────────────────────────────────────────
    # 13. OPTIONAL TUNING GRID (for automated search)
    # ──────────────────────────────────────────────────────────────────────────────
    # Below is an example of how you could organize a small grid‐search over key hyperparams.
    # You don’t have to use it, but it’s provided as a template if you want to systematically search.
    #
    # HYPERPARAM_GRID = {
    #     'LATENT_DIM': [32, 48, 64],
    #     'BETA': [0.5, 1.0, 2.0],
    #     'LEARNING_RATE': [1e-3, 5e-4, 1e-4],
    #     'TEMP_RANGE': [(0.8,2.2), (1.0,2.5), (1.2,3.0)]
    # }
    #
    # For each combination, you would re‐train the VAE (on the same train/val split) and record:
    #   - Validation MSE
    #   - Latent variance profile (how many dims are active)
    #   - Sampling yield (what % of z’s produce valid curves)
    #   - Mean pairwise diversity among the top‐k valid shapes
    # Then choose the combination that best balances those metrics.
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=== CONFIGURATION ===")
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 21)

class AirfoilDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

class AirfoilVAE(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None, hidden_dim=None, dropout_rate=None):
        super(AirfoilVAE, self).__init__()
        
        # Use config values if not specified
        input_dim = input_dim or Config.TARGET_POINTS * 2
        latent_dim = latent_dim or Config.LATENT_DIM
        hidden_dim = hidden_dim or Config.HIDDEN_DIM
        dropout_rate = dropout_rate or Config.DROPOUT_RATE
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim//4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//4, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Ensures output is bounded
        )
        
    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def resample_airfoil(coords, target_points=None):
    """Resample airfoil to exactly target_points using arc length parameterization"""
    target_points = target_points or Config.TARGET_POINTS
    
    if len(coords) < 3:
        return None
    
    # Calculate cumulative arc length
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Normalize arc length to [0, 1]
    total_length = arc_lengths[-1]
    if total_length == 0:
        return None
    arc_lengths = arc_lengths / total_length
    
    # Create interpolation functions
    try:
        fx = interpolate.interp1d(arc_lengths, coords[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
        fy = interpolate.interp1d(arc_lengths, coords[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        # Sample at uniform intervals
        uniform_s = np.linspace(0, 1, target_points)
        resampled_x = fx(uniform_s)
        resampled_y = fy(uniform_s)
        
        return np.column_stack([resampled_x, resampled_y])
    except:
        return None

class AirfoilGenerator:
    def __init__(self, config=None):
        # Allow custom config or use default
        self.config = config or Config
        
        self.data_folder = self.config.DATA_FOLDER
        self.target_points = self.config.TARGET_POINTS
        self.latent_dim = self.config.LATENT_DIM
        self.scaler = MinMaxScaler()
        self.model = None
        
        # Device selection
        if self.config.USE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
    def load_dat_file(self, filepath):
        """Load airfoil coordinates from .dat file"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            start_idx = 0
            for i, line in enumerate(lines):
                if re.match(r'^[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+', line.strip()):
                    start_idx = i
                    break
            
            coords = []
            for line in lines[start_idx:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            coords.append([x, y])
                        except ValueError:
                            continue
            
            return np.array(coords)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_and_preprocess_data(self):
        """Load all .dat files and resample to consistent point count"""
        data_path = Path(self.data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"Folder '{self.data_folder}' not found")
        
        dat_files = list(data_path.glob('*.dat'))
        if not dat_files:
            raise FileNotFoundError(f"No .dat files found in '{self.data_folder}'")
        
        print(f"Loading and resampling {len(dat_files)} airfoil files...")
        
        airfoils = []
        for filepath in dat_files:
            coords = self.load_dat_file(filepath)
            if coords is not None and len(coords) > 5:
                resampled = resample_airfoil(coords, self.target_points)
                if resampled is not None:
                    airfoils.append(resampled)
        
        if not airfoils:
            raise ValueError("No valid airfoil data loaded")
        
        print(f"Successfully loaded {len(airfoils)} airfoils, each with {self.target_points} points")
        
        # Normalize data
        all_coords = np.vstack(airfoils).reshape(-1, 2)
        self.scaler.fit(all_coords)
        
        scaled_airfoils = []
        for airfoil in airfoils:
            scaled = self.scaler.transform(airfoil)
            scaled_airfoils.append(scaled)
        
        return np.array(scaled_airfoils)
    
    def vae_loss_function(self, recon_x, x, mu, logvar, beta=None):
        """VAE loss with adjustable beta for diversity control"""
        beta = beta or self.config.BETA
        MSE = nn.MSELoss(reduction='sum')(recon_x, x.view(x.size(0), -1))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + beta * KLD
    
    def evaluate_model(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                recon_batch, mu, logvar = self.model(batch)
                loss = self.vae_loss_function(recon_batch, batch, mu, logvar)
                total_loss += loss.item()
        return total_loss / len(dataloader.dataset)
    
    def train(self, custom_config=None):
        """Train VAE with configuration parameters"""
        # Use custom config if provided, otherwise use instance config
        config = custom_config or self.config
        
        # Load and preprocess data
        airfoils = self.load_and_preprocess_data()
        
        # Train/validation split
        train_data, val_data = train_test_split(
            airfoils, 
            test_size=config.TRAIN_TEST_SPLIT, 
            random_state=config.RANDOM_SEED
        )
        
        train_dataset = AirfoilDataset(train_data)
        val_dataset = AirfoilDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        input_dim = self.target_points * 2
        self.model = AirfoilVAE(
            input_dim=input_dim, 
            latent_dim=config.LATENT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=config.SCHEDULER_PATIENCE, 
            factor=config.SCHEDULER_FACTOR
        )
        
        print(f"Training on {self.device}")
        print(f"Dataset: {len(train_data)} train, {len(val_data)} validation")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training with early stopping
        best_val_loss = float('inf')
        no_improve = 0
        
        for epoch in range(config.EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.model(batch)
                loss = self.vae_loss_function(recon_batch, batch, mu, logvar, config.BETA)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            val_loss = self.evaluate_model(val_loader)
            scheduler.step(val_loss)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    def generate_airfoil(self, z=None, temperature=None):
        """Generate airfoil from latent vector with temperature scaling"""
        temperature = temperature or self.config.DEFAULT_TEMPERATURE
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(1, self.latent_dim).to(self.device) * temperature
            else:
                z = torch.FloatTensor(z).unsqueeze(0).to(self.device)
            
            generated = self.model.decode(z)
            generated_np = generated.cpu().numpy().reshape(self.target_points, 2)
            denormalized = self.scaler.inverse_transform(generated_np)
            
            return denormalized
    
    def generate_diverse_airfoils(self, num_airfoils=None, diversity_method=None):
        """Generate diverse airfoils using multiple strategies"""
        num_airfoils = num_airfoils or self.config.NUM_GENERATED_AIRFOILS
        diversity_method = diversity_method or self.config.DIVERSITY_METHOD
        
        airfoils = []
        
        if diversity_method == 'temperature':
            temperatures = np.linspace(self.config.TEMP_MIN, self.config.TEMP_MAX, num_airfoils)
            for temp in temperatures:
                airfoil = self.generate_airfoil(temperature=temp)
                airfoils.append(airfoil)
                
        elif diversity_method == 'spherical':
            for _ in range(num_airfoils):
                z = torch.randn(self.latent_dim)
                z = z / torch.norm(z) * np.random.uniform(
                    self.config.SPHERICAL_RADIUS_MIN, 
                    self.config.SPHERICAL_RADIUS_MAX
                )
                airfoil = self.generate_airfoil(z.numpy())
                airfoils.append(airfoil)
                
        elif diversity_method == 'clustered':
            n_clusters = min(4, num_airfoils)
            cluster_centers = [np.random.randn(self.latent_dim) * 2 for _ in range(n_clusters)]
            
            for i in range(num_airfoils):
                center = cluster_centers[i % n_clusters]
                z = center + np.random.randn(self.latent_dim) * 0.5
                airfoil = self.generate_airfoil(z)
                airfoils.append(airfoil)
                
        else:  # mixed approach
            methods_per_airfoil = num_airfoils // 3
            
            # Temperature sampling
            temps = np.linspace(self.config.TEMP_MIN, self.config.TEMP_MAX, methods_per_airfoil)
            for temp in temps:
                airfoil = self.generate_airfoil(temperature=temp)
                airfoils.append(airfoil)
            
            # Spherical sampling
            for _ in range(methods_per_airfoil):
                z = torch.randn(self.latent_dim)
                z = z / torch.norm(z) * np.random.uniform(
                    self.config.SPHERICAL_RADIUS_MIN, 
                    self.config.SPHERICAL_RADIUS_MAX
                )
                airfoil = self.generate_airfoil(z.numpy())
                airfoils.append(airfoil)
            
            # High variance sampling
            remaining = num_airfoils - len(airfoils)
            for _ in range(remaining):
                z = np.random.randn(self.latent_dim) * self.config.HIGH_VAR_STD
                airfoil = self.generate_airfoil(z)
                airfoils.append(airfoil)
        
        return airfoils
    
    def generate_multiple_airfoils(self, num_airfoils=100):
        """Generate multiple diverse airfoils (legacy wrapper)"""
        return self.generate_diverse_airfoils(num_airfoils, 'mixed')
    
    def interpolate_airfoils(self, z1, z2, steps=None):
        """Generate airfoils by interpolating between two latent vectors"""
        steps = steps or self.config.INTERPOLATION_STEPS
        airfoils = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            airfoil = self.generate_airfoil(z_interp)
            airfoils.append(airfoil)
        return airfoils
    
    def validate_airfoil_quality(self, airfoil):
        """Basic quality checks for generated airfoils"""
        x_range = airfoil[:, 0].max() - airfoil[:, 0].min()
        y_range = airfoil[:, 1].max() - airfoil[:, 1].min()
        
        diffs = np.diff(airfoil, axis=0)
        max_jump = np.max(np.sqrt(np.sum(diffs**2, axis=1)))
        
        return {
            'x_range': x_range,
            'y_range': y_range,
            'max_jump': max_jump,
            'is_valid': (x_range > self.config.MIN_X_RANGE and 
                        y_range > self.config.MIN_Y_RANGE and 
                        max_jump < self.config.MAX_JUMP_THRESHOLD)
        }
    
    def plot_airfoils(self, airfoils, title="Generated Airfoils", validate=True):
        """Plot multiple airfoils with quality assessment"""
        plot_count = min(len(airfoils), self.config.PLOT_AIRFOILS_COUNT)
        plt.figure(figsize=(15, 10))
        
        for i in range(plot_count):
            plt.subplot(2, 3, i + 1)
            plt.plot(airfoils[i][:, 0], airfoils[i][:, 1], 'b-', linewidth=2)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            if validate:
                quality = self.validate_airfoil_quality(airfoils[i])
                status = "✓" if quality['is_valid'] else "✗"
                plt.title(f'Airfoil {i+1} {status}')
            else:
                plt.title(f'Airfoil {i+1}')
            
            plt.xlabel('x/c')
            plt.ylabel('y/c')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
    
    def save_airfoil(self, airfoil, filename):
        """Save generated airfoil to .dat file"""
        with open(filename, 'w') as f:
            f.write(f"Generated Airfoil - VAE Model\n")
            for x, y in airfoil:
                f.write(f"{x:.6f} {y:.6f}\n")
    
    def calculate_airfoil_distance(self, airfoil1, airfoil2):
        """Calculate Euclidean distance between two airfoils"""
        return np.sqrt(np.sum((airfoil1 - airfoil2)**2))
    
    def calculate_variance_matrix(self, airfoils):
        """Calculate pairwise distances between airfoils"""
        n = len(airfoils)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.calculate_airfoil_distance(airfoils[i], airfoils[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def analyze_diversity(self, generated_airfoils, original_dataset=None):
        """Analyze diversity within generated airfoils and against original dataset"""
        print("\n=== DIVERSITY ANALYSIS ===")
        
        gen_distances = self.calculate_variance_matrix(generated_airfoils)
        gen_mean_dist = np.mean(gen_distances[np.triu_indices_from(gen_distances, k=1)])
        gen_std_dist = np.std(gen_distances[np.triu_indices_from(gen_distances, k=1)])
        
        print(f"Generated Airfoils Internal Diversity:")
        print(f"  Mean pairwise distance: {gen_mean_dist:.4f}")
        print(f"  Std pairwise distance:  {gen_std_dist:.4f}")
        print(f"  Distance range: {gen_distances.max():.4f} - {gen_distances.min():.4f}")
        
        gen_to_orig_distances = None
        if original_dataset is not None:
            print(f"\nComparing against original dataset ({len(original_dataset)} airfoils):")
            
            sample_size = min(self.config.DIVERSITY_SAMPLE_SIZE, len(original_dataset))
            orig_sample = np.random.choice(len(original_dataset), sample_size, replace=False)
            sampled_originals = [original_dataset[i] for i in orig_sample]
            
            gen_to_orig_distances = []
            for gen_airfoil in generated_airfoils:
                distances_to_orig = [self.calculate_airfoil_distance(gen_airfoil, orig) 
                                   for orig in sampled_originals]
                min_dist = min(distances_to_orig)
                gen_to_orig_distances.append(min_dist)
            
            mean_nearest_dist = np.mean(gen_to_orig_distances)
            
            orig_distances = self.calculate_variance_matrix(sampled_originals)
            orig_mean_dist = np.mean(orig_distances[np.triu_indices_from(orig_distances, k=1)])
            
            print(f"  Mean distance to nearest original: {mean_nearest_dist:.4f}")
            print(f"  Original dataset internal diversity: {orig_mean_dist:.4f}")
            print(f"  Novelty ratio (gen/orig diversity): {gen_mean_dist/orig_mean_dist:.2f}")
            
            novelty_threshold = orig_mean_dist * self.config.NOVELTY_THRESHOLD_RATIO
            novel_count = sum(1 for d in gen_to_orig_distances if d > novelty_threshold)
            print(f"  Novel airfoils (>{novelty_threshold:.4f} from originals): {novel_count}/{len(generated_airfoils)}")
        
        return {
            'internal_mean_distance': gen_mean_dist,
            'internal_std_distance': gen_std_dist,
            'distance_matrix': gen_distances,
            'gen_to_orig_distances': gen_to_orig_distances
        }

def main():
    # Print current configuration
    Config.print_config()
    
    # You can modify config values here or create a custom config class
    # Example: Config.LATENT_DIM = 64  # Increase latent dimensions
    # Example: Config.EPOCHS = 300     # Reduce training epochs
    
    # Initialize generator
    generator = AirfoilGenerator()
    
    try:
        print("Starting VAE training...")
        generator.train()
        
        # Store original dataset for comparison
        original_airfoils = generator.load_and_preprocess_data()
        original_denormalized = []
        for airfoil in original_airfoils:
            denorm = generator.scaler.inverse_transform(airfoil)
            original_denormalized.append(denorm)
        
        print("\nGenerating diverse airfoils...")
        
        # Test different diversity methods
        methods = ['temperature', 'spherical', 'clustered', 'mixed']
        all_generated = []
        
        for method in methods:
            print(f"\nTesting {method} method...")
            diverse_airfoils = generator.generate_diverse_airfoils(
                num_airfoils=Config.NUM_GENERATED_AIRFOILS // 4, 
                diversity_method=method
            )
            all_generated.extend(diverse_airfoils)
            
            if len(diverse_airfoils) > 1:
                method_distances = generator.calculate_variance_matrix(diverse_airfoils)
                method_mean_dist = np.mean(method_distances[np.triu_indices_from(method_distances, k=1)])
                print(f"  Mean pairwise distance: {method_mean_dist:.4f}")
        
        # Generate best diverse set
        best_airfoils = generator.generate_diverse_airfoils()
        
        # Filter for valid airfoils
        valid_airfoils = []
        for i, airfoil in enumerate(best_airfoils):
            quality = generator.validate_airfoil_quality(airfoil)
            if quality['is_valid']:
                valid_airfoils.append(airfoil)
            print(f"Airfoil {i+1}: Valid={quality['is_valid']}, "
                  f"X-range={quality['x_range']:.3f}, "
                  f"Y-range={quality['y_range']:.3f}")
        
        print(f"\nGeneration success rate: {len(valid_airfoils)}/{len(best_airfoils)} ({len(valid_airfoils)/len(best_airfoils)*100:.1f}%)")
        
        # Diversity Analysis
        diversity_results = generator.analyze_diversity(valid_airfoils, original_denormalized)
        
        # Plot results
        generator.plot_airfoils(valid_airfoils, "AI-Generated Airfoils (VAE)")
        
        # Save valid airfoils
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        for i, airfoil in enumerate(valid_airfoils):
            filename = f'{Config.OUTPUT_FOLDER}/vae_airfoil_{i+1}.dat'
            generator.save_airfoil(airfoil, filename)
        
        print(f"Saved {len(valid_airfoils)} valid airfoils to '{Config.OUTPUT_FOLDER}' folder")
        
        # Demonstrate latent space interpolation
        print("\nDemonstrating latent space interpolation...")
        z1 = np.random.randn(generator.latent_dim)
        z2 = np.random.randn(generator.latent_dim)
        interpolated = generator.interpolate_airfoils(z1, z2)
        generator.plot_airfoils(interpolated, "Latent Space Interpolation")
        
        # Diversity visualization
        if len(valid_airfoils) > 1:
            print("\nCreating diversity heatmap...")
            distance_matrix = diversity_results['distance_matrix']
            plt.figure(figsize=(8, 6))
            plt.imshow(distance_matrix, cmap='viridis')
            plt.colorbar(label='Distance')
            plt.title('Pairwise Distance Matrix (Generated Airfoils)')
            plt.xlabel('Airfoil Index')
            plt.ylabel('Airfoil Index')
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()