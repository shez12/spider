# Right Sharpa Wave model

Copied with its license from the local MuJoCo Menagerie-derived asset at
`/home/szc/work/primavista-ns/pv/models/hands/third_party/sharpa_wave/`.
The accompanying README describes the upstream Sharpa source, right-hand frame
conventions, and the model-generation process. This is a standalone right-hand
robot model; no dataset trajectory reflection is required.

Source references:

- https://github.com/google-deepmind/mujoco_menagerie/tree/main/sharpa_wave
- https://github.com/sharpa-robotics/sharpa-urdf-usd-xml

The original XML and mesh assets are preserved. KUKA attachment, palm tracking
site and environment contact pairs are added by `prepare_kuka_right.py` in the
generated scene. The attachment transform is a simulation fixture assumption,
not a measurement of a physical KUKA/right-Sharpa mounting adapter.
