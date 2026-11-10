# Supplied left Sharpa HA4

Source: `/home/yimu12/work/simtoolreal/assets/urdf/left_sharpa_ha4/`.

`original/` contains byte-preserved copies of the base MuJoCo XML, matching URDF, and meshes. The base XML is used, not the offset XML (which adds an application-specific root pose and changes some exclusions). A missing XML attribute whitespace is normalized only while parsing the generated adapter; the original copy remains unchanged.

`left.xml` is a fixed-base adapter with 22 scalar joints and 22 independent finger motors, no coupling constraints, added palm/finger tracking sites, and motor effort limits taken from the matching URDF. It is not a floating-base SPIDER model and is not registered in the existing `run_arm_hand.py` choices. The separate test entry point is `python -m examples.arm_hand.try_sharpa`. No existing method source is edited.

Native hand mass: 1.307687 kg. Joint ranges, axes, armature, friction loss, damping, meshes, inertias, and native position-servo settings are retained. The supplied MuJoCo XML has no force limits; the adapter uses the supplied URDF effort limits (approximately 0.189–3.3 Nm depending on joint). Velocity ratings are recorded but not enforced by an additional speed limiter.

Canonical tracking wrist is inferred at native position `[0,-0.01,0.06]` m with identity rotation. Fingertip tracking points are inferred from the elastomer mesh bounds: midpoint in local x/z and maximum local y on each distal pad. These are geometric choices, not calibrated hardware frames.

The hand is mounted to the existing generic seven-joint arm for the coffee-can test. The inferred mount aligns the canonical palm 0.22 m along the last arm body. Because the supplied model is left-handed and the dataset episode uses a right Allegro hand, the test mirrors world y and object-local y consistently, including object mesh geometry and quaternion transforms. Object mass and shape are preserved up to reflection. The source video remains the original right-hand episode and is labeled accordingly.

Results and reproducible commands: `outputs/allegro_to_sharpa_left/README.md`. Existing passive-joint compatibility edits from earlier work remain in the checkout; no further changes to them are made for Sharpa.
