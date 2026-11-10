# Mustard multi-source campaign

Output: `outputs/mustard_multisource_valid100`.

The published DexYCB right-hand `006_mustard_bottle` data contains 22 distinct
demonstrations. Robot retargetings of the same task are deduplicated; preference
is Allegro, Sharpa, Xhand, Inspire. Other datasets are excluded because matching
names do not establish matching object geometry or coordinates.

Each demonstration independently needs 100 accepted augmentations (2,200 total).
The previously completed 100 are reused after checking both physical reports
and all exported artifacts. New source references retain their published timing
and geometry. Each source receives its own deterministic augmentation seed.

The existing IK, original SPIDER sampling optimizer, 512 samples, 200-iteration
limit, convergence stopping, gravity compensation and independent GPU replay
acceptance criteria are unchanged. Failed candidates do not count. GPU work is
limited globally to ten jobs using file locks and the private CUDA MPS daemon;
each source does not receive ten simultaneous jobs. Candidate IK is limited to
eight jobs globally. Initial baseline preparation uses up to eight CPU workers.

`selected_sources.json` records source provenance and per-source directories.
`summary.json`, `accepted.json`, and `timing.json` aggregate progress. Each group
retains individual states, checkpoints, physical reports, trajectories and videos.
Campaign wall time starts with this campaign; it does not include the earlier
run that supplied the reused group.

Run with the spider-repro Python:

```sh
python -u -m examples.arm_hand.mustard_multisource
```

Source coordinators retry interrupted GPU work from checkpoints. The supervisor
retries a failed source coordinator up to three times, then records that source
as needing attention while other groups continue. Do not restart the supervisor
while its detached source coordinators are still running.
