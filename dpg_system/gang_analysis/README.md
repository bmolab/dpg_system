# gang_analysis

The scripts that produced `../CHARACTERIZATION_RESULTS.md` — the measurement of
the torque field over 20.4 M frames of AMASS that the gang work rests on.

**These are working scripts, not a library.** They were written to answer
specific questions once, they hardcode paths, and several are single-use
probes. They are here because the alternative was the numbers in
CHARACTERIZATION_RESULTS.md being unreproducible, which is what happened to the
original gang test suite (see GANG_NOTES "Verified" — those scripts are gone).

Run them from this directory. Repo root is added to `sys.path` relative to the
file, so they work from any checkout, but the **data** paths below are absolute
and machine-specific.

## Paths you will have to edit

    ROOT / ARCH / ARCHIVE   /Users/drokeby/dpg_system/AMASS_Dynamic[/SMPL_H]
    NOISE                   ../noise_estimation/noise_results_lenses_2026_06_17
    V1 / V2                 AMASS_Torque (v1, deleted) / AMASS_Dynamic

`MODEL_PATH` resolves relative to this file and should need no editing.

## The pipeline, in order

    build_noise_index.py      noise verdicts + clean_segments -> noise_index.json
                              (needed by everything that filters; run first)

    characterize.py           the main pass. Conditioning histograms, per-gang
                              stats, 66x66 cross-products for correlation.
                              --regime SEG applies clean_segments per frame.
                              ~80-120 s on 8 workers over the full corpus.

    analyze.py <dir>          reads characterize's output; produces the
                              conditioning table, PCA/dimensionality, preset
                              scoring, and uncovered channel pairs.

    power_analysis.py         mechanical power: torque x angular velocity.
                              Per joint, per gang, generating/absorbing split,
                              rectification by window. ~40 s.

    filtered_compare.py       the same headline numbers under ALL / KEEP / SEG
                              / SEGP / CLEAN quality regimes at once. This is
                              what established that filtering changes nothing
                              structural. ~100 s.

    build_torque_prior.py     mean + covariance + whitening + the distance
                              distribution -> torque_prior.npz. Two passes.
                              --ridge is the eigenvalue floor; 1e-3 was swept
                              and chosen.

    validate_surprise.py      the four checks on a prior: calibration, noise
                              sensitivity, subset discrimination, and the
                              gang-span decomposition.

## Archive tooling

    inspect_torque_archive.py   what is in an unknown archive: arrays, shapes,
                                ranges, which torque streams, full vectors vs
                                reduced. Handles npz/npy/hdf5/pickle/torch.
    verify_v2.py                sanity checks on a torque archive: stream
                                decomposition identity, structural zeros,
                                max_torque agreement, gang bank evaluation.
    fingerprint_archive.py      sha + percentiles per stream, before a reprocess
    compare_fingerprint.py      after a reprocess: did exactly the intended
                                streams change, at exactly the intended rates?

The fingerprint pair exists because the archive is 31 GB and the disk cannot
hold two copies, so a before/after comparison has to be captured in advance.

## Probes and one-offs

    probe_torque.py           streaming vs batch timing; also holds make_options(),
                              the option block the others import. Not standalone.
    probe_archive.py          second-pass probe: dead channels, effort/torque ratio
    probe_available.py        what process_frame actually exposes per frame
    acc_smooth_rate_test.py   the controlled same-motion test behind commit
                              8896ffa (acceleration smoothing in ms, not frames)
    test_gang_surprise_node.py end-to-end test of the surprise outlet through the
                              real node path, headless

## Known duplication

`surprise_core.py` here and `../gang_prior.py` in the package both implement
whitening against the prior. They are not the same thing and neither is
redundant:

- `gang_prior.py` is the **runtime** path — minimal, no scipy, caches per-gang
  directions, degrades to reporting 0 when the prior is missing.
- `surprise_core.py` is the **analysis** path — carries `gang_basis()`,
  `decompose()` and `shape_surprise()`, used for the section 12e decomposition
  and the magnitude-correlation check.

If the whitening maths is ever changed, both need it.

## Caveats worth knowing before trusting output

- Percentiles come from log-magnitude histograms with 0.05-dex bins, so the
  resolution is about 12%. Changes smaller than that quantize to zero or one
  bin — which is how a real 3% effect first showed up as a string of identical
  -10.9% steps. For small effects, measure directly.
- `rate_hz` in `per_file.npz` is computed with `np.diff` across the whole
  masked array, so under `--regime SEG` it differences across segment joins.
  0.18% of samples, harmless for the median, wrong for any tail statistic.
- `filtered_compare.py` and `characterize.py` take the index path as an
  argument because macOS multiprocessing uses spawn: globals set in `main()`
  do not reach workers.
