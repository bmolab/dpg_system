# smpl_ragdoll tests

Scripts that exercise `dpg_system/smpl_ragdoll.py` and `dpg_system/smpl_bullet.py`
without the GUI: `repro_trans.py` holds a stub of the node with every property
as a plain value, and everything else builds on it.  Run any script directly
with the project's python from any directory; the walk and cartwheel scripts
read `assets/motion_capture_files/`.

    python dpg_system/tests/ragdoll/test_cycle.py

## Pass/fail tests

| script | checks |
|---|---|
| test_cycle | arm → release → catch → release again through the node's control surface; release speed matches the capture |
| test_weights | per-joint weights, groups, `weight`/`release`/`catch` messages, the `weights` input |
| test_node | pose/trans plumbing, format detection, outputs |
| test_active | the 20-joint Shadow quaternion layout in and out |
| test_support | the support measure and auto-release on a handstand |
| test_bullet_frames | Bullet joint/velocity frame conventions (probes that fixed the URDF) |
| test_bullet_release | momentum carried through a release: velocity, spin, parabola, flight turn |
| test_bullet_drop | a whole body dropped settles on the floor |
| test_bullet_arm | a limp arm hangs and swings like one |
| test_ragdoll, test_ramp, test_root, test_conserve, test_seed, test_robust, test_limb, test_self, test_limits, test_land, test_cycle_native | the native (non-Bullet) core |

## Characterisation sweeps (print numbers, no assertions)

| script | shows |
|---|---|
| spring | worst-joint deflection and pelvis sag vs blend weight on the walk, root free and driven |
| wpart | the same, shorter |
| spine_probe | deflection per body region with and without gravity compensation |
| ring, nod3 | ring-down of a shoved elbow / head at a partial weight — the damping tests |
| heli | wind-up check: everything partial on the cartwheel for 900 frames |
| osc4 (→ osc → heli) | per-joint chatter: excess motion, sign flips, limit occupancy |
| rev3 (→ rev_sweep) | release everything at every handstand frame; early joint reversals |
| flick4 | Walk B17 frames 500–600 release flicks |
| leg_release, leg_sweep, fall_through, w999 | a released leg vs the support measure; the 0.999 case |
| bounce | pelvis bobbing and horizontal offset vs weight |
