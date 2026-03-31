import pandas as pd
import torch
import numpy as np
from analysis_core.core import Core, calculate_B_C, evaluate_quantum_results_with_uncertainties, build_histograms, build_results
import vector

vector.register_awkward()
import awkward as ak
from pathlib import Path

from analysis_core.unfold import main
import pandas as pd
import os

# Global vars
bin_nums = 20 + 1
bin_edges = {
    "m_tt": np.array([0, 400, 500, 800, np.inf]),
    "B_Ak": np.linspace(-1, 1, bin_nums),
    "B_An": np.linspace(-1, 1, bin_nums),
    "B_Ar": np.linspace(-1, 1, bin_nums),
    "B_Bk": np.linspace(-1, 1, bin_nums),
    "B_Bn": np.linspace(-1, 1, bin_nums),
    "B_Br": np.linspace(-1, 1, bin_nums),
    "C_kk": np.linspace(-1, 1, bin_nums),
    "C_kn": np.linspace(-1, 1, bin_nums),
    "C_kr": np.linspace(-1, 1, bin_nums),
    "C_nk": np.linspace(-1, 1, bin_nums),
    "C_nn": np.linspace(-1, 1, bin_nums),
    "C_nr": np.linspace(-1, 1, bin_nums),
    "C_rk": np.linspace(-1, 1, bin_nums),
    "C_rn": np.linspace(-1, 1, bin_nums),
    "C_rr": np.linspace(-1, 1, bin_nums),
}
    

def classify_TT2L(point_cloud, assignment_target, event_selection=None):
    """
    Classify particles in a TT2L topology.

    Parameters:
        point_cloud: (N, num_particles, num_features)
        assignment_target: tuple/list with 2 index arrays (each (N, 2)) for the two groups

    Returns:
        Dict with b1, b2, l1, l2 reconstructions.
        :param event_selection:
    """

    idx = np.arange(point_cloud.shape[0])[:, None]

    # Two targets (e.g., top1 and top2)
    t1_target = assignment_target[0]  # (N, 2)
    t2_target = assignment_target[1]  # (N, 2)

    # Gather candidates
    t1_recon_tmp = point_cloud[idx, t1_target, :].numpy()  # (N, 2, F)
    t2_recon_tmp = point_cloud[idx, t2_target, :].numpy()

    N = t1_recon_tmp.shape[0]

    if event_selection is None:
        event_selection = np.ones(N, dtype=bool)
    
    def select_object(recon_tmp, mask_feature_idx, threshold=0.5):
        """
        For each event, select the candidate if feature > threshold.
        At most one is expected to be True. Return (N, F) with NaN for none.
        """
        mask = recon_tmp[:, :, mask_feature_idx] > threshold  # (N, 2)
        idx_first = np.argmax(mask, axis=1)  # (N,)
        has_true = np.any(mask, axis=1)  # (N,)
        valid = has_true & event_selection  # only keep events that pass both conditions
        
        result = recon_tmp[np.arange(N), idx_first]  # (N, F)
        result[~valid] = np.nan  # Set to NaN if no valid candidate or not selected

        result = vector.zip({
            'pt': np.expm1(result[:, 1]),
            'eta': result[:, 2],
            'phi': result[:, 3],
            # 'mass': result[:, 4],
            'energy': np.expm1(result[:, 0]),
            'charge': result[:, 6],
        })

        return result

    # B-jet: feature[4] > 0.5
    b1_recon = select_object(t1_recon_tmp, 4)
    b2_recon = select_object(t2_recon_tmp, 4)

    # Lepton: feature[5] > 0.5
    l1_recon = select_object(t1_recon_tmp, 5)
    l2_recon = select_object(t2_recon_tmp, 5)

    return {
        'b1_recon': b1_recon,
        'b2_recon': b2_recon,
        'l1_recon': l1_recon,
        'l2_recon': l2_recon,
    }


def zip_two_neutrinos(neutrino_dict):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    log_pt = to_numpy(neutrino_dict['log_pt'])  # (N, 2)
    eta = to_numpy(neutrino_dict['eta'])  # (N, 2)
    phi = to_numpy(neutrino_dict['phi'])  # (N, 2)

    pt = np.expm1(log_pt)

    nu1 = vector.zip({
        "pt": pt[:, 0],
        "eta": eta[:, 0],
        "phi": phi[:, 0],
        "mass": np.zeros_like(pt[:, 0]),  # neutrinos are massless
    })

    nu2 = vector.zip({
        "pt": pt[:, 1],
        "eta": eta[:, 1],
        "phi": phi[:, 1],
        "mass": np.zeros_like(pt[:, 1]),  # neutrinos are massless
    })

    return nu1, nu2

def sanity_and_merge(pairs, data):
    merged = {}
    for a, b, new_key in pairs:
        valid_a = ~np.isnan(data[a]).all(axis=1)
        valid_b = ~np.isnan(data[b]).all(axis=1)
        if np.any(valid_a & valid_b):
            raise ValueError(f"Conflict: both {a} and {b} present in same event.")
        valid_a = valid_a[:, None]  # broadcast
        merged[new_key] = np.where(valid_a, data[a], data[b])
    return merged

def extract_batch_assignments(batch, classify_fn, process="TT2L", truth_ass=False, truth_nu=False):
    pred = batch['assignment_prediction']
    target = batch['assignment_target']
    target_mask = batch['assignment_target_mask']

    process_match = {
        'num_lepton': batch['full_input_point_cloud'].sum(axis=1)[:, 5].numpy().astype(np.int32),
        'num_bjet': batch['full_input_point_cloud'].sum(axis=1)[:, 4].numpy().astype(np.int32),
        'total_charge': batch['full_input_point_cloud'].sum(axis=1)[:, 6].numpy().astype(np.int32),
    }

    common_selection = (
            (process_match['num_bjet'] == 2) &
            (process_match['num_lepton'] == 2) &
            (process_match['total_charge'] == 0)
    )

    target_list = target[process]
    if truth_ass:
        pred_process = target_list
    else:
        pred_process = pred[process]['best_indices']
    mask_process = target_mask[process]

    process_match.update({
        **classify_fn(batch['full_input_point_cloud'], pred_process, event_selection=common_selection),
    })

    nu1_true, nu2_true = zip_two_neutrinos(batch['neutrinos']['target'])
    if truth_nu:
        nu1_pred, nu2_pred = nu1_true, nu2_true
    else:
        nu1_pred, nu2_pred = zip_two_neutrinos(batch['neutrinos']['predict'])
    process_match.update({
        "nu1_recon": nu1_pred,
        "nu2_recon": nu2_pred,
        "nu1_truth": nu1_true,
        "nu2_truth": nu2_true,
    })

    for p_idx, (assignment_target, assignment_prediction, assignment_target_mask) in enumerate(
            zip(target_list, pred_process, mask_process)):
        assignment_target = assignment_target.numpy()
        assignment_prediction = assignment_prediction.numpy()
        assignment_target_mask = assignment_target_mask.numpy()

        # Matching: true if all particles in the group are correctly assigned
        matched = (assignment_target == assignment_prediction)
        matched = matched.all(axis=1)  # along particle axis

        process_match[f"{process}_{p_idx}"] = matched
        process_match[f"{process}_{p_idx}_mask"] = assignment_target_mask

    return process_match

def mask_invalid(recon_data, base_recon_cut):
    # Mask invalid recon_data based on base_recon_cut
    trans_items = ["l1_recon", "l2_recon", "t1_recon", "t2_recon"]
    nan_array = ak.zeros_like(recon_data.l1_recon.pt) + np.nan
    mask_particles = vector.zip({
        'pt': nan_array,
        'eta': nan_array,
        'phi': nan_array,
        'energy': nan_array,
        'charge': nan_array,
    })
    for key in recon_data.fields:
        if key in trans_items:
            recon_data = ak.with_field(recon_data, ak.where(base_recon_cut, recon_data[key], mask_particles), key)
        
    return recon_data

def Get_ana_data(raw_data, truth_data, truth_ass=False, truth_nu=False):
    dfs = []
    if truth_ass:
        print("[INFO] With truth assignment")
    if truth_nu:
        print("[INFO] Use truth nu info")
    for batch in raw_data:
        out = extract_batch_assignments(batch, classify_fn=classify_TT2L,
                                        truth_ass=truth_ass, truth_nu=truth_nu)
        dfs.append(out)

    # Instead of pd.concat, build one big awkward.Array
    recon_data = ak.zip({
        k: ak.concatenate([out[k] for out in dfs])
        for k in dfs[0].keys()
    })

    truth_particle = {
        'b2': truth_data['b~'],
        'b1': truth_data['b'],
        'l2': truth_data['l-'],
        'l1': truth_data['l+'],
        't2': truth_data['t~'],
        't1': truth_data['t'],
    }

    for p, v in truth_particle.items():
        recon_data = ak.with_field(recon_data, v, f'{p}_truth')

    recon_data = ak.with_field(recon_data, recon_data.l1_recon + recon_data.nu1_recon + recon_data.b1_recon, f't1_recon')
    recon_data = ak.with_field(recon_data, recon_data.l2_recon + recon_data.nu2_recon + recon_data.b2_recon, f't2_recon')

    base_recon_cut = (recon_data["num_bjet"] > 0) & (recon_data["num_lepton"] == 2) & (recon_data["total_charge"] == 0) & \
        (recon_data.l1_recon.pt > 25) & (recon_data.l2_recon.pt > 25) & \
        (abs(recon_data.l1_recon.eta) < 2.47) & (abs(recon_data.l2_recon.eta) < 2.47) & \
        (recon_data.l1_recon.charge + recon_data.l2_recon.charge == 0) & \
        (recon_data.b1_recon.pt > 25) & (recon_data.b2_recon.pt > 25) & \
        (abs(recon_data.b1_recon.eta) < 2.5) & (abs(recon_data.b2_recon.eta) < 2.5)  
    recon_data = mask_invalid(recon_data, base_recon_cut)
    
    recon_data = ak.with_field(recon_data, base_recon_cut, "base_recon_cut")
    
    truth_result = Core(
        main_particle_1=recon_data.t2_truth,
        main_particle_2=recon_data.t1_truth,
        child1=recon_data.l2_truth,
        child2=recon_data.l1_truth,
    ).analyze()

    recon_result = Core(
        main_particle_1=recon_data.t2_recon,
        main_particle_2=recon_data.t1_recon,
        child1=recon_data.l2_recon,
        child2=recon_data.l1_recon,
    ).analyze()
    
    full = build_results(truth_result, recon_result)
    
    return recon_data, full, base_recon_cut

def truth_ana(raw_data):
    raw_truth = {
    key.replace('EXTRA/lhe/', ''): torch.cat([item[key] for item in raw_data]).numpy()
        for key in raw_data[0].keys()
    if key.startswith('EXTRA/lhe/')
    }

    # Define pairs to merge: (key_a, key_b, merged_key)
    pairs_to_merge = [
        ('e+', 'mu+', 'l+'),
        ('e-', 'mu-', 'l-'),
        ('nu(e)', 'nu(mu)', 'v'),
        ('nu(e)~', 'nu(mu)~', 'v~'),
    ]

    # Run merging with sanity check
    truth_data = sanity_and_merge(pairs_to_merge, raw_truth)
    for k in ['W+', 'W-', 'b', 'b~', 't', 't~']:
        truth_data[k] = raw_truth[k]

    for k in truth_data.keys():
        truth_data[k] = vector.zip({
            'pt': truth_data[k][:, 0],
            'eta': truth_data[k][:, 1],
            'phi': truth_data[k][:, 2],
            'mass': truth_data[k][:, 3],
        })

    truth_core = Core(
        main_particle_1=truth_data['t'],
        main_particle_2=truth_data['t~'],
        child1=truth_data['l+'],
        child2=truth_data['l-'],
    )

    truth_result = truth_core.analyze()
    truth_result = truth_result.query('mass < 400')
    truth_hists = build_histograms(truth_result)

    result, result_up, result_down = calculate_B_C(truth_hists, kappas=(1.0, -1.0))
    D = -(result['C_nn'] + result['C_rr'] + result['C_kk'])
    final = evaluate_quantum_results_with_uncertainties(result, result_up, result_down)
    return D, final, truth_data

def Efficiency_calculation(recon_data, base_recon_cut, save_path="results"):
    mask_all_true = (
            recon_data["TT2L_0"] &
            recon_data["TT2L_0_mask"] &
            recon_data["TT2L_1"] &
            recon_data["TT2L_1_mask"] &
            base_recon_cut
    )

    valid_events = (
            recon_data.TT2L_0_mask &
            recon_data.TT2L_1_mask &
            base_recon_cut
    )

    all_true = (
            recon_data["TT2L_0"] &
            recon_data["TT2L_1"] 
    )

    # Count how many events have all four True
    count = ak.sum(mask_all_true)
    all_count = len(recon_data)
    print("Number of events with all four True:", count, "out of", ak.sum(valid_events), "percentage:",
            count / ak.sum(valid_events) * 100)
    print("Number of events with all Events:", ak.sum(all_true), "out of", all_count, "percentage:",
            ak.sum(all_true) / all_count * 100)

    # Save the counts and percentages to a CSV file

    results = {
    "Metric": ["All Four True", "All Events"],
    "Count": [int(count), int(ak.sum(all_true))],
    "Total": [int(ak.sum(valid_events)), all_count],
    "Percentage": [float(count / ak.sum(valid_events) * 100), float(ak.sum(all_true) / all_count * 100)]
    }

    results_df = pd.DataFrame(results)
    output_dir = Path(f"{save_path}/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "event_counts.csv", index=False)

def unfolding_process(full, Num_raw_ents, save_path="results"):

    full['weight'] = 19.29e3 * 140 / Num_raw_ents
    if save_path != "results" and os.path.exists(Path("results/unfolding")):
        os.rename(Path("results/unfolding"), Path("results/unfolding_"))
    unfolded = main(full, bin_edges=bin_edges, weight_col='weight')

    # 1) Reshape all unfolded arrays for each variable (except m_tt)
    mtt_nbins = len(bin_edges['m_tt']) - 1

    unfolded_temp = {
        key: {
            'edges': edges,
            'counts': unfolded[f'{key}_recon_unfold_content'].to_numpy().reshape(mtt_nbins, len(edges) - 1),
            'errors': unfolded[f'{key}_recon_unfold_error'].to_numpy().reshape(mtt_nbins, len(edges) - 1),
        }
        for key, edges in bin_edges.items()
        if key != 'm_tt'
    }

    # 2) Split by m_tt bins using clean dict comprehension
    unfolded_hists = {
        f"m_tt < {mtt_right}": {
            key: {
                'edges': data['edges'],
                'counts': data['counts'][idx],
                'errors': data['errors'][idx],
            }
            for key, data in unfolded_temp.items()
        }
        for idx, mtt_right in enumerate(bin_edges['m_tt'][1:])
    }

    result, result_up, result_down = calculate_B_C(unfolded_hists['m_tt < 400.0'], kappas=(1.0, -1.0))
    # D = -(result['C_nn'] + result['C_rr'] + result['C_kk'])
    final = evaluate_quantum_results_with_uncertainties(result, result_up, result_down)
    
    if save_path != "results" and os.path.exists(Path("results/unfolding_")):
        os.rename(Path("results/unfolding"), Path(f"{save_path}/unfolding"))
        os.rename(Path("results/unfolding_"), Path("results/unfolding"))
        
    return final, result, result_up, result_down, unfolded

def save_unfold_res(final, result, result_up, result_down, file_tag, save_path="results"):
    base_df = pd.DataFrame({
    'value': result,
    'uncertainty_up': result_up,
    'uncertainty_down': result_down
    })
    base_df['uncertainty_up'] = base_df['uncertainty_up'] - base_df['value']
    base_df['uncertainty_down'] = base_df['value'] - base_df['uncertainty_down']

    # Step 2: Combine with `final` entries (like 'Concurrence', 'Ckk + Cnn', etc.)
    final_df = pd.DataFrame.from_dict(final, orient='index')
    final_df.index.name = 'name'

    # Step 3: Concatenate both
    combined_df = pd.concat([base_df, final_df])

    # Optional: Reset index if you prefer flat structure
    combined_df.index.name = 'name'
    combined_df = combined_df.reset_index()
    if not os.path.exists(Path(os.getcwd()) / "results" / "csv"):
            os.makedirs(Path(os.getcwd()) / "results" / "csv")
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(f'{save_path}/csv/{file_tag}.csv', index=False)


def analysis(input_file: str):
    file_tag = Path(input_file).stem
    raw_data = torch.load(input_file)
    Num_raw_ents = 0
    for b in raw_data:
        Num_raw_ents += b['full_input_point_cloud'].shape[0]
    truth_D, truth_final, truth_data = truth_ana(raw_data)
    named_configs = {
        "neutrino": {
            "variables": ["pt", "eta", "phi", "dR"],
            "kin_range": {"pt": (0, 350), "eta": (-np.pi * 1.5, np.pi * 1.5), "phi": (-np.pi, np.pi), "dR": (0, 4)}, 
            "columns": ['nu'],
        },
        "top": {
            "variables": ["pt", "eta", "phi", "mass"],
            "kin_range": {"pt": (0, 600), "eta": (-np.pi * 1.5, np.pi * 1.5), "phi": (-np.pi, np.pi), "mass": (100, 240)},
            "columns": ['t'],
        },
    }
    recon_data, full, base_recon_cut = Get_ana_data(raw_data, truth_data)
    Efficiency_calculation(recon_data, base_recon_cut)
    final, result, result_up, result_down, unfolded = unfolding_process(full, Num_raw_ents)
    save_unfold_res(final, result, result_up, result_down, file_tag)
    
    new_dir = str(Path(os.getcwd()) / "results") + "_" + file_tag
    print(new_dir)
    os.rename(str(Path(os.getcwd()) / "results"), new_dir)
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] # The pt file generated from evenet
    analysis(input_file)
