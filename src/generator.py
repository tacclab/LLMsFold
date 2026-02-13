import os
import re
import pandas as pd
from groq import Groq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from src.utils import (
    validate_smiles, get_max_similarity, parse_smiles_from_text, 
    get_binding_pockets_and_residues, check_pubchem_patents
)

async def generate_molecules_unified(
    pdb_path, boltz_client, output_dir, protein_sequence, few_shot_csv,
    max_iterations=3, max_samples=5, use_pocket_data=True 
):
    # 1. Setup Data and Fingerprints
    few_shot_data = pd.read_csv(few_shot_csv, sep=';')
    positives = few_shot_data["Smiles"].dropna().tolist()[:5]
    
    gen_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    target_fps = [gen_fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in positives if Chem.MolFromSmiles(s)]
    
    global_registry = []
    context_leads = positives.copy()
    groq_client = Groq()
    
    # 2. Pocket Processing
    pocket_coords = None
    pocket_residues = None
    clean_indices = []
    
    if use_pocket_data:
        pocket_coords, pocket_residues = get_binding_pockets_and_residues(pdb_path, output_dir)
        
        # Handle string or list inputs for residues
        residue_list = pocket_residues.split(',') if isinstance(pocket_residues, str) else (pocket_residues or [])
        
        # Extract numeric indices safely
        for r in residue_list:
            match = re.search(r'\d+', str(r))
            if match:
                clean_indices.append(int(match.group()))

        print(f"Targeting Pocket at {pocket_coords}")
        print(f"Cleaned Residue Indices: {clean_indices}")
    else:
        print("Running in Few-Shot mode (Ignoring pocket constraints).")

    # 3. Iterative Generation Loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- ITERATION {iteration} ---")
        leads_text = ", ".join(context_leads[-5:])
        
        base_system = (
            "You are an expert medicinal chemist. Your goal is to generate novel, chemically valid SMILES strings "
            "as a Python list: ['SMILES1', 'SMILES2']. "
            "CONSTRAINTS: Satisfy Lipinski's Rule of Five, ensure synthetic feasibility, and avoid PAINS. "
            "TECHNICAL RULES: 1. Ensure all rings are explicitly closed. 2. Maintain valid valency. "
            "3. Use [nH] for aromatic nitrogen. 4. Specify stereochemistry (@/@@) where relevant."
        )

        if use_pocket_data:
            user_content = (
                f"Design {max_samples} drug-like molecules for a binding pocket containing: {pocket_residues}. "
                f"Strategy: Create fragments for H-bonds with these residues while inspired by: {leads_text}. "
                "Return ONLY the Python list."
            )
        else:
            user_content = (
                f"Generate {max_samples} bioisosteres or analogs of: {leads_text}. "
                "Improve drug-likeness and novelty. Return ONLY the Python list."
            )

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.8 
            )

            # Robust Parsing
            raw_content = completion.choices[0].message.content
            new_smiles = parse_smiles_from_text(raw_content)
            valid_smiles = [s for s in new_smiles if validate_smiles(s)]
            
            if not valid_smiles:
                print(f"No valid SMILES parsed in iteration {iteration}. Skipping...")
                continue

            # 4. Property Computation (Physics-based/ML Scoring)
            results_df = await boltz_client.compute_properties(
                valid_smiles, 
                protein_sequence, 
                pocket_residues=clean_indices if use_pocket_data else None
            )
            
            if not results_df.empty:
                results_df['MaxSim'] = results_df['SMILES'].apply(lambda x: get_max_similarity(x, target_fps))
                # Balanced scoring: 70% Predicted Affinity, 30% Structural Similarity
                results_df['score'] = (results_df['Affinity_Prob'] * 0.7) + (results_df['MaxSim'] * 0.3)
                
                global_registry.extend(results_df.to_dict('records'))
                
                # Update context with best performers for the next iteration
                top_performers = pd.DataFrame(global_registry).sort_values(by='score', ascending=False)
                context_leads = top_performers['SMILES'].head(3).tolist()

        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            continue

    # 5. Final Report & IP Analysis
    if not global_registry:
        print("No molecules were successfully generated. Check LLM connectivity or SMILES validation.")
        return

    final_hits = pd.DataFrame(global_registry).drop_duplicates(subset='SMILES')
    
    print(f"\nVerifying IP status for {len(final_hits)} unique candidates...")
    
    ip_results = []
    for smiles in final_hits['SMILES']:
        cid, id_patents, sub_patents = await check_pubchem_patents(smiles)
        
        is_novel = "Yes" if (cid is None or cid == 0) else "No"
        ip_results.append({
            'PubChem_CID': cid if cid else "N/A",
            'Identity_Patents': id_patents,
            'Substructure_Patents': sub_patents,
            'Is_Novel': is_novel
        })

    # Merge IP data back
    ip_df = pd.DataFrame(ip_results)
    final_hits = pd.concat([final_hits.reset_index(drop=True), ip_df], axis=1)

    # Final Sort: Novelty first, then highest score
    final_hits = final_hits.sort_values(by=['Is_Novel', 'score'], ascending=[False, False])

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "unified_report.csv")
    final_hits.to_csv(report_path, index=False)
    
    print(f"Workflow Complete. Results in {report_path}")