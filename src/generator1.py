import os
import pandas as pd
from groq import Groq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from src.utils import validate_smiles, get_max_similarity, parse_smiles_from_text , get_binding_pockets_and_residues,check_pubchem_patents
import re
# async def generate_molecules_unified(pdb_path,
#     boltz_client, output_dir, protein_sequence, few_shot_csv,
#     max_iterations=3, max_samples=5
# ):
#     few_shot_data = pd.read_csv(few_shot_csv, sep=';')
#     positives = few_shot_data["Smiles"].dropna().tolist()[:5]
    
#     gen_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2)
#     target_fps = [gen_fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in positives if Chem.MolFromSmiles(s)]
    
#     global_registry = []
#     context_leads = positives.copy()
#     groq_client = Groq()
    
#     ## Pocket Finder
#     #pocket_residues = get_p2rank_pocket(pdb_path) 
#     pocket_coords, pocket_residues = get_binding_pockets_and_residues(pdb_path, output_dir)

#     print(f"Targeting Pocket at {pocket_coords}")
#     print(f"Residues within 8A: {pocket_residues}")
#     ## Generation
#     for iteration in range(1, max_iterations + 1):
#         print(f"\n--- ITERATION {iteration} ---")
#         leads_text = ", ".join(context_leads[-5:])
        
#         completion = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": (
#                         "You are an expert medicinal chemist specializing in structure-based drug design. "
#                         "Your task is to generate chemically valid SMILES strings. "
#                         "Rules:\n"
#                         "1. Return ONLY a valid Python list of strings.\n"
#                         "2. Ensure all rings are explicitly closed (check digit pairs like 1...1).\n"
#                         "3. Maintain valid valency for all atoms.\n"
#                         "4. Focus on fragments that complement the provided binding pocket residues."
#                     )
#                 },
#                 {
#                     "role": "user", 
#                     "content": (
#                         f"Design {max_samples} molecules to bind at {pocket_coords}. "
#                         f"Critical residues within 8A: {pocket_residues}. "
#                         f"Reference leads: {leads_text}. "
#                         f"Generate {max_samples} novel, drug-like SMILES. "
#                         "Review each SMILES to ensure no unclosed rings before providing the list."
#                     )
#                 }
#             ],
          
#             temperature=0.9 
#         )
#     # for iteration in range(1, max_iterations + 1):
#     #     print(f"\n--- ITERATION {iteration} ---")
#     #     leads_text = ", ".join(context_leads[-5:])
        
#     #     completion = groq_client.chat.completions.create(
#     #         model="llama-3.3-70b-versatile",
#     #         messages=[{"role": "system", "content": "Return only a Python list of SMILES: ['S1', 'S2']"},
#     #                   {"role": "user", "content": f"Generate {max_samples} novel molecules similar to: {leads_text}"}],
#     #         temperature=1.0
#     #     )

#         new_smiles = parse_smiles_from_text(completion.choices[0].message.content)
#         valid_smiles = [s for s in new_smiles if validate_smiles(s)]
        
#         results_df = await boltz_client.compute_properties(valid_smiles, protein_sequence)

#########################################################
### New Logic With Pockets Flag

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
    
   
    pocket_coords = None
    pocket_residues = None
    clean_indices = []
    if use_pocket_data:
        pocket_coords, pocket_residues = get_binding_pockets_and_residues(pdb_path, output_dir)
        if isinstance(pocket_residues, str):
            residue_list = [r.strip() for r in pocket_residues.split(',')]
        else:
            residue_list = pocket_residues
        clean_indices = []
        for r in residue_list:
            match = re.search(r'\d+', r)
            if match:
                clean_indices.append(int(match.group()))

        print(f"Cleaned Indices: {clean_indices}")
 

        print(f"Targeting Pocket at {pocket_coords}")
        print(f"Pocket Residues {pocket_residues}")
    else:
        print("Running in Few-Shot mode (Ignoring pocket constraints).")

    # for iteration in range(1, max_iterations + 1):
    #     print(f"\n--- ITERATION {iteration} ---")
    #     leads_text = ", ".join(context_leads[-5:])
        
       
    #     if use_pocket_data:
    #        if use_pocket_data:
    #         sys_content = (
    #             "You are an expert medicinal chemist. Generate chemically valid SMILES. "
    #             "The target pocket is defined by these residues: {pocket_residues}. \n"
    #             "Focus on: \n"
    #             "- Proper ring closures (ensure every '1' has a matching '1').\n"
    #             "- Correct valency (Carbon must have 4 bonds).\n"
    #             "- Using [nH] for aromatic nitrogen atoms to ensure kekulization."
    #         )
            
    #         user_content = (
    #             f"Design {max_samples} drug-like molecules. \n"
    #             f"Target Site: A pocket containing {pocket_residues}. \n"
    #             f"Strategy: Create fragments that form H-bonds with these residues while remaining "
    #             f"structurally similar to these leads: {leads_text}. \n"
    #             "Double-check that all aromatic rings are valid before outputting."
    #         )
    #         # sys_content = "Focus on generating analogs and bioisosteres of the provided reference leads."
    #         # user_content = f"Generate {max_samples} novel molecules similar to the lead compounds: {leads_text}."
    #     else:
    #         sys_content = "Focus on generating analogs and bioisosteres of the provided reference leads."
    #         user_content = f"Generate {max_samples} novel molecules similar to the lead compounds: {leads_text}."

    #     completion = groq_client.chat.completions.create(
    #         model="llama-3.3-70b-versatile",
    #         messages=[
    #             {"role": "system", "content": f"You are an expert medicinal chemist. {sys_content} Return ONLY a Python list of strings."},
    #             {"role": "user", "content": user_content}
    #         ],
    #         temperature=0.8
    #     )



    for iteration in range(1, max_iterations + 1):
        print(f"\n--- ITERATION {iteration} ---")
        leads_text = ", ".join(context_leads[-5:])
        
        
        base_system = (
            "You are an expert medicinal chemist. Your goal is to generate novel, chemically valid SMILES strings "
            "as a Python list: ['SMILES1', 'SMILES2']. "
            "CONSTRAINTS: You must satisfy Lipinski's Rule of Five, ensure synthetic feasibility, "
            "and strictly avoid PAINS (Pan-Assay Interference Compounds). "
            "TECHNICAL RULES: 1. Ensure all rings are explicitly closed. 2. Maintain valid valency. "
            "3. Use [nH] for aromatic nitrogen to ensure proper kekulization."
        )

       
        if use_pocket_data:
            user_content = (
                f"Design {max_samples} drug-like molecules tailored for a binding pocket containing: {pocket_residues}. "
                f"Strategy: Create fragments that form H-bonds with these residues while remaining "
                f"structurally inspired by these leads: {leads_text}. "
                f"Ensure the generated SMILES are novel and obey the chemical constraints provided."
            )
        else:
            user_content = (
                f"Generate {max_samples} novel molecules that are bioisosteres or analogs of: {leads_text}. "
                "Focus on maintaining the core scaffold while improving drug-likeness and novelty."
            )

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": base_system},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8 
        )


        new_smiles = parse_smiles_from_text(completion.choices[0].message.content)
        valid_smiles = [s for s in new_smiles if validate_smiles(s)]
        
        
        results_df = await boltz_client.compute_properties(
            valid_smiles, 
            protein_sequence, 
            pocket_residues=clean_indices if use_pocket_data else None
        )
        
            
        if not results_df.empty:
            results_df['MaxSim'] = results_df['SMILES'].apply(lambda x: get_max_similarity(x, target_fps))
            results_df['score'] = (results_df['Affinity_Prob'] * 0.7) + (results_df['MaxSim'] * 0.3)
            global_registry.extend(results_df.to_dict('records'))
            top_leads = pd.DataFrame(global_registry).sort_values(by='score', ascending=False)
            context_leads = top_leads['SMILES'].head(3).tolist()

    final_hits = pd.DataFrame(global_registry).drop_duplicates(subset='SMILES')
    
    if not final_hits.empty:
        print(f"\nVerifying IP status (Identity & Substructure) for {len(final_hits)} unique candidates...")
        
        p_cids = []
        id_patent_counts = []
        sub_patent_counts = []
        p_novel = []

        for smiles in final_hits['SMILES']:
            cid, id_patents, sub_patents = await check_pubchem_patents(smiles)
            
            if cid is None or cid == 0:
                p_cids.append("N/A")
                id_patent_counts.append(0)
                sub_patent_counts.append(sub_patents)
                p_novel.append("Yes") 
            else:
                p_cids.append(str(cid))
                id_patent_counts.append(id_patents)
                sub_patent_counts.append(sub_patents)
                p_novel.append("No") 

      
        final_hits['PubChem_CID'] = p_cids
        final_hits['Identity_Patents'] = id_patent_counts
        final_hits['Substructure_Patents'] = sub_patent_counts
        final_hits['Is_Novel'] = p_novel
        final_hits = final_hits.sort_values(
            by=['Is_Novel', 'score'], 
            ascending=[False, False]
        )

        os.makedirs(output_dir, exist_ok=True)
        final_hits.to_csv(f"{output_dir}/unified_report.csv", index=False)  
    print(f"Workflow Complete. Results in {output_dir}/unified_report.csv")   
    final_hits = pd.DataFrame(global_registry).drop_duplicates(subset='SMILES')


