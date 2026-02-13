
import httpx
import asyncio
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import sascorer  

class BoltzClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.public_url = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
        self.status_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"

    async def make_nvcf_call(self, smiles, sequence, pocket_residues=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "NVCF-POLL-SECONDS": "300",
            "Content-Type": "application/json"
        }
        
        polymers = [{
            "id": "A", 
            "molecule_type": "protein", 
            "sequence": sequence,
            "msa": {"uniref90": {"a3m": {"alignment": f">seq1\n{sequence}", "format": "a3m"}}}
        }]

        ligands = [{"smiles": smiles, "id": "L1", "predict_affinity": True}]
        
        constraints = []
        if pocket_residues:
            constraints.append({
                "constraint_type": "pocket",
                "binder": "L1",  
                "contacts": [{"id": "A", "residue_index": res} for res in pocket_residues]
            })

        data = {
            "polymers": polymers,
            "ligands": ligands,
            "constraints": constraints,
            "recycling_steps": 3, 
            "sampling_steps": 200, 
            "diffusion_samples": 1,
            "step_scale": 1.2, 
            "without_potentials": True
        }
      
        async with httpx.AsyncClient() as client:
            response = await client.post(self.public_url, json=data, headers=headers, timeout=400)
            if response.status_code == 202:
                task_id = response.headers.get("nvcf-reqid")
                while True:
                    await asyncio.sleep(5)
                    status_res = await client.get(self.status_url.format(task_id=task_id), headers=headers)
                    if status_res.status_code == 200: 
                        return status_res.json()
                    elif status_res.status_code >= 400: 
                        return None
            return response.json() if response.status_code == 200 else None

    async def compute_properties(self, smiles_list, sequence, pocket_residues=None):
        results = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: 
                continue
            
            print(f"\nEvaluating SMILES: {smiles}")
            data = await self.make_nvcf_call(smiles, sequence, pocket_residues=pocket_residues) 
            
            if data:
                #  structural confidence scores
                ptm = data.get("ptm_scores", [0])[0]
                iptm = data.get("iptm_scores", [0])[0]
                conf = data.get("confidence_scores", [0])[0]
                plddt = data.get("complex_plddt_scores", [0])[0]
                
                #  ligand-specific affinity data
                affinities = data.get("affinities", {})
                lig_key = next(iter(affinities)) if affinities else None
                lig_data = affinities[lig_key] if lig_key else {}
                
                prob = lig_data.get("affinity_probability_binary", [0])[0]
                pic50 = lig_data.get("affinity_pic50", [0])[0]
                ic50_um = pow(10, -pic50) * 1e6 

            
                print(f"  > Predicted pTM:           {ptm:.3f}")
                print(f"  > Predicted ipTM:          {iptm:.3f}")
                print(f"  > Confidence Score:        {conf:.3f}")
                print(f"  > Average pLDDT:           {plddt:.3f}")
                print(f"  > Affinity Probability:    {prob:.3f}")
                print(f"  > Predicted pIC50:         {pic50:.3f}")
                print(f"  > Predicted IC50 (µM):     {ic50_um:.3f}")

              
                results.append({
                    'SMILES': smiles,
                    'pTM': ptm,
                    'ipTM': iptm,
                    'Confidence': conf,
                    'pLDDT': plddt,
                    'Affinity_Prob': prob,
                    'pIC50': pic50,
                    'IC50_uM': ic50_um,
                    'MolWt': round(Descriptors.MolWt(mol), 2),
                    'LogP': round(Descriptors.MolLogP(mol), 2),
                    'QED': round(QED.qed(mol), 3),
                    'SAS': round(sascorer.calculateScore(mol), 3),
                    'TPSA': round(Descriptors.TPSA(mol), 2),
                    'H_Acceptors': Descriptors.NumHAcceptors(mol),
                    'H_Donors': Descriptors.NumHDonors(mol),
                    'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol)
                })
              
        return pd.DataFrame(results)