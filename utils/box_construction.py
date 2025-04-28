import numpy as np

def calculate_molecule_numbers(box_size, density, salt_concentration, salt_molar_mass, solvent1_molar_mass, solvent2_molar_mass, solvent_mass_ratio):
    """
    Calculate the number of molecules for lithium salt and two solvents in the box.

    Parameters:
    - box_size (float): Length of the cubic box in Å (angstrom).
    - density (float): Overall density in g/cm³.
    - salt_concentration (float): Concentration of lithium salt in mol/L.
    - salt_molar_mass (float): Molar mass of lithium salt in g/mol.
    - solvent1_molar_mass (float): Molar mass of solvent 1 in g/mol.
    - solvent2_molar_mass (float): Molar mass of solvent 2 in g/mol.
    - solvent_mass_ratio (float): Mass ratio of solvent 1 to solvent 2.

    Returns:
    - A dictionary with the number of molecules for lithium salt, solvent 1, and solvent 2.
    """
    # Constants
    NA = 6.02214076e23  # Avogadro's number (molecules/mol)

    #
    volume = (box_size * 1e-9)**3  # Convert Å to 0.1m and compute volume

    # Calculate salt mass based on its concentration (g)
    salt_moles = salt_concentration * (volume) * NA  # Convert volume to L


    # Calculate total mass in the box (g)
    total_mass_g = density * volume

    salt_mass_g = salt_moles * salt_molar_mass / NA

    # Calculate solvent masses
    solvent_total_mass_g = total_mass_g - salt_mass_g
    solvent1_mass_g = solvent_total_mass_g * (solvent_mass_ratio / (1 + solvent_mass_ratio))
    solvent2_mass_g = solvent_total_mass_g * (1 / (1 + solvent_mass_ratio))

    # Calculate number of molecules
    salt_molecules = salt_moles
    solvent1_molecules = solvent1_mass_g / solvent1_molar_mass * NA
    solvent2_molecules = solvent2_mass_g / solvent2_molar_mass * NA

    return {
        "Lithium Salt Molecules": int(salt_molecules),
        "Solvent 1 Molecules": int(solvent1_molecules),
        "Solvent 2 Molecules": int(solvent2_molecules)
    }


# Example usage
box_size = 45  # Box size in Å
density = 1500 # Overall density in g/dm3

salt_concentration = 0.6  # Lithium salt concentration in mol/L

salt_molar_mass = 151.91  # Molar mass of lithium salt in g/mol (e.g., LiPF6)

solvent1_molar_mass = 86.09  #Molar mass of solvent 1 in g/mol (e.g., EC)
solvent2_molar_mass = 73.09   # Molar mass of solvent 2 in g/mol (e.g., AN)

solvent_mass_ratio = 1/9  # Mass ratio of solvent 1 to solvent 2 (1:1)


result = calculate_molecule_numbers(box_size, density, salt_concentration, salt_molar_mass, solvent1_molar_mass, solvent2_molar_mass, solvent_mass_ratio)
print("Molecule counts:")
for key, value in result.items():
    print(f"{key}: {value}")
