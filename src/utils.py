import re
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import subprocess
import pandas as pd
import os
import requests
import stat
import tarfile
import glob
import httpx
import httpx
import asyncio
import numpy as np
import deepchem as dc


def extract_sequence_from_pdb(pdb_path):
    """Extracts the primary protein sequence from a PDB file."""
    mol = Chem.MolFromPDBFile(pdb_path)
    if not mol:
        raise ValueError(f"Could not parse PDB file at {pdb_path}")
    return Chem.MolToSequence(mol)

######################

#######################

def validate_smiles(smiles):
    if not smiles or not isinstance(smiles, str): return False
    smiles = smiles.strip().strip('.')
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            return True
        return False
    except: return False

######################

#######################

def get_max_similarity(smiles, target_fps):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fp = gen.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, target_fps)
        return max(similarities) if similarities else 0.0
    except: return 0.0

######################

#######################

def parse_smiles_from_text(raw_text):
    list_match = re.search(r"\[\s*['\"](.*?)['\"]\s*\]", raw_text, re.DOTALL)
    if list_match:
        return re.findall(r"['\"]([a-zA-Z0-9@+\-\[\]\(\)\\\/%=#$]{5,})['\"]", raw_text)
    return []

######################

#######################

def setup_p2rank():
    """Locates the prank executable and ensures it has run permissions."""

    base_dir = os.getcwd()
    script_path = os.path.join(base_dir, "prank/prank")
    
    if not os.path.exists(script_path):
        import glob
        existing = glob.glob(os.path.join(base_dir, "p2rank*", "prank"), recursive=True)
        if existing:
            script_path = existing[0]
        else:
            raise FileNotFoundError("P2Rank executable 'prank' not found in project root.")
        
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)
    
    return script_path

######################

#######################

def get_p2rank_pocket(pdb_path):
    p2rank_executable = setup_p2rank()
    
    output_dir = os.path.abspath("p2rank_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # command
    print(f"Running P2Rank analysis on {pdb_path}...")
    cmd = [p2rank_executable, "predict", "-f", pdb_path, "-o", output_dir, "-visualizations", "0"]
    subprocess.run(cmd, check=True, capture_output=True)
    pdb_filename = os.path.basename(pdb_path)
    pred_file = os.path.join(output_dir, f"{pdb_filename}_predictions.csv")
    
    if os.path.exists(pred_file):
        import pandas as pd
        df = pd.read_csv(pred_file)
        df.columns = df.columns.str.strip()
        if not df.empty:
            return df.iloc[0]['residue_ids']
            
    return "Unknown Pocket"

######################

#######################


def get_binding_pockets_and_residues(pdb_path, output_dir="results"):
    
    finder = dc.dock.ConvexHullPocketFinder(pad=5.0)
    pockets = finder.find_pockets(pdb_path)
    
    if not pockets:
        return "No pockets found", "Unknown"

    
    pocket_data = []
    for i, p in enumerate(pockets):
        center = p.center() 
        pocket_data.append({
            "pocket_id": i + 1,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "volume_approx": (p.x_range[1]-p.x_range[0]) * (p.y_range[1]-p.y_range[0]) * (p.z_range[1]-p.z_range[0])
        })
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pocket_data).to_csv(f"{output_dir}/all_pockets.csv", index=False)

   ## hardcoded to get the 5th pocket for CD19
    best_pocket = pockets[4]
    center = best_pocket.center()
    

    mol = Chem.MolFromPDBFile(pdb_path)
    conf = mol.GetConformer()
    
    nearby_residues = set()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - center)
        
        if dist <= 8.0:
          
            info = atom.GetPDBResidueInfo()
            if info:
                res_name = info.GetResidueName().strip()
                res_num = info.GetResidueNumber()
                chain = info.GetChainId().strip()
                nearby_residues.add(f"{res_name}{res_num}_{chain}")

    residue_list = ", ".join(sorted(list(nearby_residues)))
    pocket_desc = f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
    
    return pocket_desc, residue_list

######################

#######################


async def check_pubchem_patents(smiles):
 
    async with httpx.AsyncClient() as client:
        try:
            # Identity 
            cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
            id_resp = await client.get(cid_url, timeout=10)
            
            cid = None
            id_patents = 0
            
            if id_resp.status_code == 200:
                cid = id_resp.json()['IdentifierList']['CID'][0]
                xref_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/JSON"
                xref_resp = await client.get(xref_url, timeout=10)
                if xref_resp.status_code == 200:
                    id_patents = len(xref_resp.json().get('InformationList', {}).get('Information', [{}])[0].get('PatentID', []))

            # Substructure 
            sub_search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smiles/{smiles}/JSON"
            sub_resp = await client.get(sub_search_url, timeout=10)
            
            sub_patents = 0
            if sub_resp.status_code == 202: 
                list_key = sub_resp.json()['Waiting']['ListKey']
                
                for _ in range(3):
                    await asyncio.sleep(2)
                    list_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{list_key}/cids/JSON"
                    list_resp = await client.get(list_url)
                    
                    if list_resp.status_code == 200:
                        sub_cids = list_resp.json()['IdentifierList']['CID'][:10] 
                        sub_patents = len(sub_cids) 
                        break

            return cid, id_patents, sub_patents

        except Exception as e:
            print(f"PubChem Query Error: {e}")
            return None, 0, 0