"""Parallel CPU geometry ablation after all controlled GPU runs are complete."""
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def ablate(item):
    from .spine_angle_geometry import analyze_strict, bisect_dividers
    from .experimental_spine_axes import analyze_frame_axes
    raw, geometry, spacing = item
    strict = analyze_strict(raw, geometry, spacing)
    weighted = analyze_frame_axes(raw, geometry, spacing, neighbor_scale=.25, weight_basis='length')
    rebuilt, info = bisect_dividers(geometry, strict, spacing)
    refit = analyze_strict(raw, rebuilt, spacing) if rebuilt is not None else None
    return {'strict': strict['global_angle_deg'], 'weighted': weighted['global_angle_deg'],
            'bisectors': refit['global_angle_deg'] if refit else None,
            'info': info, 'lines': rebuilt['spine']['disc_lines'] if rebuilt else None}


def run(partial=False):
    import numpy as np
    import pydicom
    from .data import load_records, load_augmented_records
    from .spine_angle_study import ROOT, DEST, save, angle_summary, render
    protocol = json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    refs = json.loads((DEST/'reference_axes.json').read_text(encoding='utf-8'))
    originals = load_records(ROOT)
    records = originals + load_augmented_records(ROOT, ROOT/'dxa_project/outputs/augmented_15000_20260929', originals)
    by = {r.relative_path: r for r in records}
    groups = {k: [by[p] for p in paths] for k, paths in protocol['partitions'].items()}
    geometries = {r.relative_path: json.loads(r.geometry_path.read_text(encoding='utf-8')) for r in groups['test']}
    raws = {r.relative_path: np.squeeze(pydicom.dcmread(r.source_path).pixel_array) for r in groups['test']}
    results = {k: json.loads((DEST/(k+'_result.json')).read_text(encoding='utf-8')) for k in protocol['configs'] if (DEST/(k+'_result.json')).exists()}
    if not partial and len(results) != len(protocol['configs']): raise RuntimeError('Controlled variants have not all completed.')
    previous = json.loads((ROOT/'dxa_project/outputs/spine_brightness_study_20260929/predicted_lines.json').read_text(encoding='utf-8'))
    methods = {k: v['test_lines'] for k, v in results.items()}
    methods.update({'previous_raw_old_loss': previous['raw_old_loss'], 'previous_raw_new_loss': previous['raw_new_loss']})
    comparisons = {}; diagnostic = {}; predictions = {}
    progress = DEST/'geometry_comparison_progress.json'
    if progress.exists():
        # Progress contains aggregates only; recompute rows consistently.
        print('Recomputing full geometry rows with six CPU workers.', flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name, lines in methods.items():
            import hashlib
            fingerprint = hashlib.sha256(json.dumps(lines, sort_keys=True).encode()).hexdigest()
            cached_path = DEST/('geometry_cache_'+name+'.json')
            if cached_path.exists():
                cached = json.loads(cached_path.read_text(encoding='utf-8'))
                if cached['lines_sha256'] == fingerprint:
                    comparisons[name] = cached['comparison']; diagnostic[name] = cached['rows']; predictions[name] = cached['bisector_lines']
                    print(json.dumps({'geometry_cache': name}), flush=True); continue
            tasks = []
            for r in groups['test']:
                g = copy.deepcopy(geometries[r.relative_path]); g['spine']['disc_lines'] = lines[r.relative_path]
                tasks.append((raws[r.relative_path], g, r.spacing_mm))
            rows = {k: [] for k in ['strict', 'weighted', 'bisectors']}; rejected = 0; equal = []; predictions[name] = {}
            for i, (r, v) in enumerate(zip(groups['test'], pool.map(ablate, tasks, chunksize=2))):
                ref = refs[r.relative_path]
                base = {'relative_path': r.relative_path, 'study': r.study, 'reference_angle': ref['angle'], 'flag': ref['flag']}
                for k in rows: rows[k].append({**base, 'angle': v[k]})
                if v['lines'] is None: rejected += 1
                else:
                    predictions[name][r.relative_path] = v['lines']; equal.append(v['info']['max_equal_angle_error_deg'])
                if (i+1)%40 == 0: print(json.dumps({'stage': 'parallel_geometry', 'variant': name, 'processed': i+1}), flush=True)
            comparisons[name] = {'weighted_terminals': angle_summary(rows['weighted']), 'strict_terminals': angle_summary(rows['strict']),
                                 'strict_plus_bisectors_refit': angle_summary(rows['bisectors']), 'bisector_rejections': rejected,
                                 'max_equal_angle_error_deg': max(equal, default=None)}
            diagnostic[name] = rows; save(progress, comparisons)
            save(cached_path, {'lines_sha256': fingerprint, 'comparison': comparisons[name], 'rows': rows, 'bisector_lines': predictions[name]})
    if partial:
        print(json.dumps({'stage': 'partial_geometry_complete', 'methods': list(comparisons)}), flush=True); return
    winner = min(results, key=lambda k: results[k]['validation']['angle']['failure_penalized_mae_deg'])
    log = ROOT/'analysis_20260929/spine_angle_training.log'
    report = {'seconds': time.time()-log.stat().st_ctime, 'protocol': protocol,
              'prior_audit': json.loads((DEST/'prior_audit.json').read_text(encoding='utf-8')), 'winner_validation': winner,
              'results': {k: {a:b for a,b in v.items() if a not in ['rows','head_rows','history','line_rows','test_lines','validation_lines','validation_rows']} for k,v in results.items()},
              'geometry_comparisons': comparisons,
              'reference_caveat': 'Strict manual-divider contour axes are a pseudo-reference, not independent clinical measurements. Test is diagnostic; selection uses validation.',
              'geometry_execution': 'Six CPU processes; same geometry functions and inputs as sequential ablation.'}
    save(DEST/'report.json', report); save(DEST/'geometry_comparison_rows.json', diagnostic); save(DEST/'bisector_lines.json', predictions)
    render(report, results, groups, geometries)
    print(json.dumps({'stage':'complete','winner_validation':winner}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true')
    run(parser.parse_args().partial)
