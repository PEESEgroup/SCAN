# Import necessary libraries
import numpy as np

# Constants
q = 1.602e-19  # Elementary charge in Coulombs
k_B = 1.38e-23  # Boltzmann constant in J/K
NA = 6.02e23

# Inputs
D = 0.1548
print(D)
D_cm2_s = (D * 1e-5)  # Diffusion coefficient in cm^2/s
box_size_A = 48.4326
num_ions = 23
T = 300
z = 1  # Charge of Li+ ions

# Convert units
D_m2_s = D_cm2_s * 1e-4  # Convert cm^2/s to m^2/s
box_size_m = box_size_A * 1e-10  # Convert Å to meters
volume_m3 = box_size_m**3  # Volume in m^3

# Check for zero or negative volume
if volume_m3 <= 0:
    raise ValueError("Calculated volume is invalid. Please check box size input.")

# Calculate ionic conductivity
conductivity = (q**2 * num_ions * D_m2_s * z**2) / (k_B * T * volume_m3)  # In S/m

# Convert to mS/cm
conductivity_mS_cm = conductivity * 1e3 / 1e2  # Convert S/m to mS/cm

# Output result
print(f"Ionic conductivity: {conductivity:.6e} S/m")
print(f"Ionic conductivity: {conductivity_mS_cm:.6f} mS/cm")
