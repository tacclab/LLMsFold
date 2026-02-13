import os
import sys 
import re
import pandas as pd
from groq import Groq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, QED, RDConfig, Descriptors, FilterCatalog
from src.utils import (
    get_max_similarity, parse_smiles_from_text , calculate_reward,passes_lipinski,
    get_binding_pockets_and_residues, check_pubchem_patents
)


params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
FILTER_CATALOG = FilterCatalog.FilterCatalog(params)


async def generate_molecules_unified(
    pdb_path, boltz_client, output_dir, protein_sequence, few_shot_csv,
    max_iterations=3, max_samples=5, use_pocket_data=True 
):
    
    few_shot_data = pd.read_csv(few_shot_csv, sep=';')
    positives = few_shot_data["Smiles"].dropna().tolist()[:5]
    
    gen_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    target_fps = [gen_fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in positives if Chem.MolFromSmiles(s)]
    
    global_registry = []
    context_leads = positives.copy()
    groq_client = Groq()
    
    #  Pocket Processing
    pocket_coords = None
    pocket_residues = None
    clean_indices = []
    
    if use_pocket_data:
        pocket_coords, pocket_residues = get_binding_pockets_and_residues(pdb_path, output_dir)
        residue_list = pocket_residues.split(',') if isinstance(pocket_residues, str) else (pocket_residues or [])
        for r in residue_list:
            match = re.search(r'\d+', str(r))
            if match:
                clean_indices.append(int(match.group()))

        print(f"Targeting Pocket at {pocket_coords}")
        print(f"Cleaned Residue Indices: {clean_indices}")
    else:
        print("Running in Few-Shot mode (Ignoring pocket constraints).")

    # Generation Loop
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

            raw_content = completion.choices[0].message.content
            new_smiles = parse_smiles_from_text(raw_content)

            valid_smiles = []
            for s in new_smiles:
                mol = Chem.MolFromSmiles(s, sanitize=True) 
                if mol:
                    # Check for PAINS and Lipinski 
                    if not FILTER_CATALOG.HasMatch(mol) and passes_lipinski(mol):
                        valid_smiles.append(s)
                else:
                    print(f"Skipping invalid/unkekulizable SMILES: {s}")

            #  Property Computation 
            results_df = await boltz_client.compute_properties(
                valid_smiles, 
                protein_sequence, 
                pocket_residues=clean_indices if use_pocket_data else None
            )


            if not results_df.empty:
                def calculate_metrics(smi):
                    mol = Chem.MolFromSmiles(smi)
                    return {
                        'MaxSim': get_max_similarity(smi, target_fps)
                    }

                metrics = results_df['SMILES'].apply(calculate_metrics).apply(pd.Series)
                results_df = pd.concat([results_df, metrics], axis=1)
                results_df['adj_affinity'] = results_df['Affinity_Prob'].apply(lambda x: x if x > 0.6 else 0)
                results_df['score'] = results_df.apply(calculate_reward, axis=1)
                results_df = results_df[results_df['SAS'] <= 6] 
                
                global_registry.extend(results_df.to_dict('records'))

        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            continue

    # Final Report & IP Analysis
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

    ip_df = pd.DataFrame(ip_results)
    final_hits = pd.concat([final_hits.reset_index(drop=True), ip_df], axis=1)

    # Final Sort
    final_hits = final_hits.sort_values(by=['Is_Novel', 'score'], ascending=[False, False])

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "unified_report.csv")
    final_hits.to_csv(report_path, index=False)
    
    print(f"Workflow Complete. Results in {report_path}")