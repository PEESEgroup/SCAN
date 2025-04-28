from rdkit.Chem import AllChem as Chem
# from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
#
smiles = 'O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2'
mol = Chem.MolFromSmiles(smiles)

Chem.EmbedMolecule(mol)
mol = Chem.MolToMolBlock(mol)
print(mol)


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Aromatic Rings": Descriptors.NumAromaticRings(mol),
        "NHOH Count": Descriptors.NHOHCount(mol),
        "Radical Electrons": Descriptors.NumRadicalElectrons(mol),
        "Valence Electrons": Descriptors.NumValenceElectrons(mol),
        "Heavy Atom MolWt": Descriptors.HeavyAtomMolWt(mol),
        "Heavy Atom Count": Descriptors.HeavyAtomCount(mol),
        "Number of H-Bond Donors": Descriptors.NumHDonors(mol),
        "Number of H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Chi0": Descriptors.Chi0(mol),
        "Chi1": Descriptors.Chi1(mol),
    }
    return properties

# Example usage
smiles = "[Li+].O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2"  # Replace with your SMILES string
properties = calculate_properties(smiles)
print(f"Properties for {smiles}:")
print(properties.values())
# for prop, value in properties.items():
#     print(f"  {prop}: {value}")