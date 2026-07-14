# Legacy design documents — SUPERSEDED. Do not implement.

These two files describe the **pre-MSP** system: a DenseFusion-style RGB-D network trained with
an **ADD-S pose loss**, a 3-phase freeze/unfreeze curriculum, TSDF-based next-best-view, and
weight-space TTA via differentiable rendering.

**This is the thesis MSP was written to refute.** The formalization states plainly:

> "There is no reconstruction or pose loss anywhere in (10)."

They are kept for provenance only. A contributor who implements them will build the wrong system.
The current design is `../../Object Pose and Grasping/MSP_formalization_2026_07_13.md`.
