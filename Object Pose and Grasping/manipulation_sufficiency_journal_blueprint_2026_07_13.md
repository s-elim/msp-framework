# A Q1 Journal Paper Blueprint: Reformulating Object Pose and Grasp Planning as Manipulation Sufficiency

Target class: IEEE T-RO / T-ASE / RCIM (IF > 10). This is a problem-reformulation paper, not an architecture paper.

Honesty note carried throughout: the generic idea "compress perception to task-relevant features via an information bottleneck" is not new. Pacelli and Majumdar formalized task-driven representations via information bottlenecks and even demonstrated grasping (RSS 2020 and their task-driven estimation work); Bai et al. applied an information bottleneck to behavior cloning (2025). The contribution below is defined precisely against what those cannot do, and I state the delta explicitly in Steps 3 and 10. All numeric "expected results" are mechanism-based hypotheses, labeled [speculation]; no fabricated citations or benchmark numbers appear.

---

## STEP 1 - Twenty Wrong Assumptions the Community Has Been Making

Each is stated, then shown to be not merely limiting but wrong in its premise.

1. **Pose must be estimated before grasping.** Sequencing perception before action assumes the object has a task-independent state that action later consumes. But the only pose information that changes any action outcome is a small, task-defined subspace; estimating the full pose first wastes capacity and injects errors on axes no grasp is sensitive to.

2. **Pose is a well-defined, observer-independent quantity.** SE(3) pose requires a canonical frame that does not exist for a symmetric or partially observed object. Pose is defined only up to the object's symmetry group and the observation's information, so "the pose" is a category error for a large fraction of real objects.

3. **Geometric accuracy of the reconstruction is the right objective.** Chamfer or ADD-S error weights every millimetre equally. Grasp success depends on geometry only through contact normals, local curvature, and the wrench response; a mesh can be globally accurate and locally wrong exactly where it matters, or globally wrong and locally sufficient.

4. **Pose and grasp are statistically independent stages.** They are treated as separable modules, yet the conditional distribution of successful grasps given pose is sharply peaked and the mutual information runs both ways: knowing the intended grasp constrains which pose degrees of freedom must be resolved.

5. **Shape completion is sufficient for grasp planning.** Completing the occluded geometry assumes the missing surface is the bottleneck. The true bottleneck is the physical response (mass distribution, friction, deformability), which shape completion does not recover and which decides slip and torque failures.

6. **A single static observation is enough.** One RGB-D frame is treated as the input. For most manipulation-relevant ambiguities (scale, back-surface, mass), a single view is provably insufficient, and the missing information is recoverable only by moving or touching.

7. **Depth is metric and trustworthy.** Pipelines consume depth as ground truth. On transparent, specular, and dark materials depth is systematically wrong, and the error is correlated with exactly the objects where grasping is hard.

8. **Objects are rigid.** Rigidity underlies both meshing and force closure. Deformables and articulated objects violate it, and the assumption silently excludes a large part of the manipulation world.

9. **A canonical object frame transfers across instances.** Category-level pose assumes a shared frame. Intra-category shape variation makes the canonical frame ill-defined, so the estimand itself drifts across instances.

10. **Perception uncertainty is aleatoric noise to be averaged out.** Systems report a point estimate. The dangerous uncertainty is epistemic and structured (which of two discrete configurations), and averaging produces a physically impossible mean pose.

11. **Force closure computed on the estimated geometry is a valid success criterion.** It is an analytic proxy evaluated on a hallucinated surface. A self-consistent but wrong reconstruction passes the check and fails the lift, so the criterion is not grounded in outcome.

12. **More reconstruction accuracy monotonically improves grasping.** The community assumes a monotone map. Beyond the resolution at which the outcome changes, extra accuracy yields zero task benefit while costing compute and latency; the map saturates.

13. **The representation should be reusable and task-agnostic.** A single geometry is assumed to serve all downstream tasks. Different manipulation goals induce different sufficiency requirements; one geometry is either over-complete for a given task or under-informative for another.

14. **Grasping is a one-shot selection problem.** Sampling then filtering assumes no interaction. In practice a light probe or partial close resolves the decisive uncertainty, and one-shot framing forbids this cheap information.

15. **Evaluation on pose metrics predicts manipulation performance.** BOP-style pose accuracy is used as a proxy. Pose error and grasp success are only weakly coupled once error is below the task-sensitivity threshold, so the proxy misranks methods.

16. **The gripper and object are the only relevant physics.** Contact is modeled as ideal point contact with one friction coefficient. Real success depends on contact patch mechanics, compliance, and mass, which the idealization discards.

17. **Semantics and geometry are separate channels.** Language grounding is bolted on after reconstruction. But the task specified in language changes which geometric details are sufficient, so semantics should define the estimand, not post-process it.

18. **Generalization means recognizing more object categories.** Open-set is framed as category coverage. The harder generalization is across physical-response regimes (a heavy mug vs an identical-looking empty one), which category recognition does not address.

19. **Sim-to-real gap is a rendering gap.** Domain randomization targets appearance. The decisive gap is in the physical-response distribution (friction, mass, compliance), which photorealism does not close.

20. **Perception should minimize its own error, independently of the actor.** The objective is defined inside the perception module. Optimality of a representation is only definable relative to the family of actions and their outcomes; a representation has no task-free notion of "correct."

Common root: representational realism, the belief that perception should recover a canonical, accurate, observer-independent 3D state through which all action must pass. Steps 2 and 3 reject this root.

---

## STEP 2 - Five New Research Philosophies

Philosophies, not architectures. Each states what it changes about the field.

**P1. Sufficiency over accuracy.** The estimand of perception is not the accurate state but the minimal statistic that preserves all and only the information that changes action outcomes. This changes the field by replacing geometric error with a task-outcome-defined objective, so "how good is the perception" becomes answerable only jointly with the action family. It reframes evaluation, supervision, and architecture at once.

**P2. Indistinguishability defines the estimand.** Two world states are the same for manipulation if every action yields the same outcome distribution. Perception should estimate the equivalence class, not a representative point. This changes the field by making pose and shape emergent readouts identifiable only up to an outcome-equivalence, dissolving the ill-posedness of canonical frames and symmetry into a well-posed quotient.

**P3. Interaction is a perceptual operator.** A probe, nudge, or partial close is not a fallback when vision fails; it is a first-class measurement that collapses the decisive uncertainty. This changes the field by unifying perception and light interaction under one information-gain objective, so active touch and active view are the same operation on the belief.

**P4. Uncertainty is structured and actionable, not noise.** The output of perception is a calibrated belief over the sufficient statistic, with a certificate. This changes the field by making abstention and information-seeking principled, and by tying safety guarantees to perception rather than to the controller.

**P5. Language sets the sufficiency budget.** The task, expressed in language, selects which distinctions must be resolved and to what resolution. This changes the field by making semantics define the estimand up front rather than annotate a pre-built geometry, so the same scene induces different sufficient statistics under different instructions.

These are not five methods; they are five reformulations of what perception for manipulation is for. P1, P2, and P4 are mutually reinforcing and jointly the strongest.

---

## STEP 3 - The Single Strongest Philosophy

Selected: P2, indistinguishability defines the estimand, with P1 and P4 as its objective and its output, and P3 and P5 as consequences.

Statement. Define a manipulation outcome operator M that maps a world state x and an action a to an outcome distribution M(x, a) over success, stability, and slip. Two states x, x' are manipulation-indistinguishable, written x ~ x', if M(x, a) = M(x', a) for every admissible action a. The correct estimand of perception is the equivalence class [x] in the quotient X / ~, equivalently any sufficient statistic z(o) of the observation o such that the outcome family factors through z: M(x, a) = M'(z(o), a) for all a. Perception should estimate a calibrated belief over z, not a pose.

Why highest novelty. The established task-driven information bottleneck (Pacelli and Majumdar) compresses state to features relevant to a single scalar task reward under a fixed, usually RL-trained, policy. The estimand here is different in kind: sufficiency with respect to the entire action-conditioned outcome family M(., a) over all a, which is policy-free and task-general within manipulation, and the object of estimation is a quotient equivalence class with an explicit geometry, not a policy-conditioned feature vector. This yields a theorem class those works do not state: pose and shape are recoverable only up to ~, so metric accuracy beyond the sufficiency resolution is both unrecoverable and unnecessary. That is a formulation result, not a method.

Why highest scientific contribution. It converts an ill-posed problem (estimate a canonical pose that may not exist) into a well-posed one (estimate an equivalence class defined by physics), and it gives a single principle from which supervision (match outcomes), active perception (separate outcome-distinguishable classes), test-time adaptation (tighten class membership from proprioception), and efficiency (the quotient is lower-dimensional) all derive. One definition, four capabilities.

Why highest journal impact. T-RO and T-ASE reward principled reformulations validated on hardware more than they reward architecture deltas. A definition that reorganizes pose estimation, grasp planning, active perception, and uncertainty under one physically grounded quotient, with a real-robot demonstration and calibration guarantees, is a flagship-type contribution for these venues.

Why highest long-term influence. It changes the dependent variable of the subfield. If adopted, papers stop reporting Chamfer and ADD-S as ends and start reporting outcome-sufficiency, and the community stops chasing accuracy past the point it changes behavior. Changing what people measure has longer reach than any single network.

Delta stated plainly for reviewers. Over Pacelli-Majumdar: policy-free, action-family sufficiency; explicit quotient geometry with an identifiability theorem; calibrated set-valued output; unified active perception and test-time adaptation; no reinforcement learning. Over affordance and feature-field lines (ReKep, RAM, F3RM, GraspSplats): those attach task features to geometry but still assume geometry as the substrate and give no sufficiency definition, no identifiability limit, and no calibrated equivalence-class belief.

---

## STEP 4 - One Complete Framework (MSP, first form)

MSP, Manipulation-Sufficient Perception. I present the framework in its first, deliberately over-complete form, then strip it in Step 5. The estimand is the manipulation-sufficient statistic z whose belief q(z|o) encodes the equivalence class of Step 3. There is no pose target, no reconstruction loss, and no learned forward-dynamics rollout, which is what separates this from a pose-plus-grasp-plus-world-model pipeline.

Module A, probabilistic sufficiency encoder.
Input: observation o = (RGB-D, optional language c, camera intrinsics). Output: belief q_theta(z | o) over the sufficient statistic z in R^d. Purpose: map raw observation to a distribution over manipulation-sufficient codes, carrying epistemic uncertainty. Innovation: the code is trained to be minimal sufficient for the action-outcome family, not to reconstruct geometry. Mathematical intuition: rate-distortion where distortion lives in outcome space. Training objective: the information bottleneck of Step 7. Inference: one forward pass yields q(z|o). Complexity: O(F), one foundation-model backbone pass. Expected failure mode: if the training outcome operator misrepresents real contact physics, z is sufficient for simulated outcomes but not real ones.

Module B, action-outcome head.
Input: (z, action a) where a in SE(3) x gripper-config. Output: outcome distribution M'_psi(y | z, a) over success, stability margin, and slip. Purpose: realize and enforce sufficiency; also serve as the action scorer. Innovation: outcomes are physics-grounded wrench-space quantities, so supervision never needs a pose or mesh label. Mathematical intuition: M'_psi is the likelihood term whose fidelity defines sufficiency. Training objective: match the physics outcome operator M. Inference: score sampled actions, select the best. Complexity: O(K h) for K candidate actions through a light head. Expected failure mode: the action space fixes the embodiment; a new gripper redefines M.

Module C, epistemic-uncertainty and calibration layer.
Input: q(z|o) and M'_psi. Output: calibrated outcome belief and a conformal action set C_alpha. Purpose: turn spread in z into a distribution-free certificate and an abstention rule. Innovation: the certificate is located in perception, not the controller. Training: a held-out calibration set. Inference: build C_alpha, abstain if empty. Complexity: negligible. Failure mode: exchangeability breaks under shift, loosening the guarantee.

Module D, active-perception selector.
Input: current belief, a set of admissible sensing actions (next view, light probe). Output: the sensing action that most reduces outcome-class ambiguity. Purpose: acquire the one measurement that separates outcome-distinguishable classes. Innovation: information gain is computed in outcome space, not geometric coverage. Training: none beyond A and B. Inference: one-step value of information; act only when uncertainty is high. Complexity: O(candidate sensing actions x K h). Failure mode: a physical probe perturbs the scene, changing x.

Module E, test-time adaptation.
Input: proprioceptive or tactile outcome of a probe. Output: updated belief consistent with the observed outcome. Purpose: tighten class membership at deployment without labels. Innovation: adaptation is driven by outcome-consistency, the same functional as D. Training: meta-trained for few-step updates. Inference: k gradient steps on z. Complexity: O(k h). Failure mode: miscalibrated uncertainty misdirects the update.

Module F, optional pose and shape readout.
Input: z. Output: a representative SE(3) pose and coarse shape. Purpose: interpretability and interfacing with classical planners. Innovation: the readout is the maximum-entropy representative of the equivalence class, honest about what is unidentifiable. Training: a small decoder on top of frozen z. Inference: one pass. Complexity: O(h). Failure mode: users misread the representative as the true metric pose.

---

## STEP 5 - Minimization: Remove Every Removable Module

The exercise is to delete until each survivor is necessary.

Module F, pose readout: removable. It plays no role in action selection, uncertainty, or adaptation; it exists only to interface with legacy planners. Cut from the core, offered as an optional adaptor.

Module E, test-time adaptation: not a separate learned module. It is Module B's likelihood applied to a probe outcome plus a belief update. It is an inference procedure over A and B, not a third network. Fold in.

Module D, active perception: likewise not a module. It is the value of information of Module B's outcome prediction under Module A's epistemic spread. Fold in as an inference procedure.

Module C, calibration: the conformal wrapper is a post-hoc procedure on Module B, not a learned network. Fold in as an inference-time wrapper.

Module B, action-outcome head: necessary. It defines sufficiency (without it, z has no objective) and it is the action scorer. Keep.

Module A, probabilistic encoder: necessary. It produces the belief over z. Keep. Its epistemic uncertainty (via a variational posterior or a small ensemble) is required so that D and E have a signal, so uncertainty is intrinsic to A, not a separate module.

Minimal MSP. Exactly two learned modules: the probabilistic sufficiency encoder A and the action-outcome head B. Grasp selection, calibrated abstention, active view-or-touch, and test-time adaptation are all inference procedures that reuse A and B. Pose and shape are an optional readout. This is the strongest possible answer to "keep simplifying": one encoder, one outcome head, one principle, and every capability the field currently builds separate modules for falls out of the two.

---

## STEP 6 - Four Algorithmic Contributions

All four are algorithmic, not implementation, engineering, or dataset.

C1. The manipulation-sufficiency objective. A policy-free, action-conditioned information bottleneck that trains a perception encoder to be a minimal sufficient statistic for the outcome operator M(., a) across the whole admissible action family, with distortion measured in outcome space rather than geometry. This is a new estimand and a new training objective for perception, distinct from task-driven IB that compresses toward a single policy or scalar reward.

C2. An identifiability theorem and estimator. A proof that under MSP, pose and shape are recoverable only up to manipulation-indistinguishability, that the residual ambiguity is exactly the null space of the outcome Jacobian dM/dx, and a maximum-entropy estimator that returns the honest representative of the equivalence class. This turns the ill-posedness of canonical pose into a characterized, bounded quantity.

C3. A single value-of-information functional unifying active perception and test-time adaptation. Both are shown to optimize the same expected outcome-class ambiguity reduction, computed by propagating the encoder's epistemic spread through the outcome head, so a next view, a light touch, and a belief update are one operation in three guises.

C4. A distribution-free calibrated outcome predictor with abstention. A conformal construction over the outcome head that returns an action set containing a truly-successful grasp with probability at least 1 minus alpha, and abstains when none is certifiable, placing a safety guarantee inside perception.

---

## STEP 7 - The Mathematics

Notation. Observation o; latent sufficient statistic z in R^d; world state x (never estimated directly); action a in A = SE(3) x gripper width; outcome y = (success in {0,1}, stability margin m in R, slip s in R). Physics outcome operator M(x, a) = p(y | x, a), realized in training by a differentiable or analytic simulator (force-closure margin, Ferrari-Canny epsilon quality, perturbation slip). Encoder q_theta(z | o). Outcome head M'_psi(y | z, a).

Module A objective, variables and derivation. We want z minimal and sufficient for the family {p(y | x, a)}_{a in A}. Sufficiency is I(z; Y | A) = I(O; Y | A); minimality is the smallest I(z; O) among sufficient z. The Lagrangian is

  min_{theta, psi}  I(z; O) - beta * I(z; Y | A).

Since the mutual informations are intractable, use the variational bound with a marginal prior r(z):

  L(theta, psi) = E_o[ KL( q_theta(z|o) || r(z) ) ]  -  beta * E_{o, a, y}[ log M'_psi(y | z, a) ],   z ~ q_theta(z|o).

The first term is the rate: it upper-bounds I(z; O) and compresses the code. The second term is the negative distortion: maximizing the outcome log-likelihood lower-bounds I(z; Y | A) and enforces sufficiency. Beta sets the sufficiency budget, and this is exactly where language enters, because a task c can scale beta per outcome dimension, making the budget task-conditioned (philosophy P5).

Why this makes sense. Rate-distortion theory says the optimal code discards precisely the observation bits that do not reduce distortion. Here distortion is outcome uncertainty, so the encoder is forced to keep geometry only where it changes what actions do, and to coarsen it elsewhere. The physical reading is direct: contact regions and center-of-mass indicators are preserved to high resolution because dM/dx is large there, while task-irrelevant surface detail is compressed.

Energy formulation. Define per-sample energy E(z; o, a, y) = - log M'_psi(y | z, a) + (1/beta) * log [ q_theta(z|o) / r(z) ]. The belief concentrates on low-energy codes that jointly predict outcomes well and stay short in description length. Sufficiency is the zero of the outcome-prediction energy gap between z and o.

Information-theoretic interpretation. z is a minimal sufficient statistic for a conditional family, the Fisher-Neyman notion generalized from a single likelihood to {p(Y | X, a)}_a. The quotient X / ~ of Step 3 is the coordinate space of such statistics; d is the intrinsic dimension of manipulation-relevant variation, typically far below the dimension of a full mesh.

Module B objective. Fit M'_psi to the physics operator M by minimizing a divergence over states and actions,

  L_out(theta, psi) = E_{x ~ D, a ~ A}[ D_KL( M(x, a) || M'_psi(. | z(o_x), a) ) ],

with a calibrated regression term for the continuous margins. Because M is physics-grounded, the label is a wrench-space outcome, never a pose or mesh, which is the whole reformulation: supervision comes from what actions do, not from how the object looks.

Action selection. At inference, choose

  a* = argmax_{a}  E_{z ~ q(z|o)}[ success(M'_psi(z, a)) ]  -  lambda * Var_{z ~ q(z|o)}[ success(M'_psi(z, a)) ],

which realizes sufficiency-driven selection with epistemic risk aversion (P1 and P4 together).

Identifiability theorem (C2), stated. Let s(x, a) = E[ success | x, a ]. Two states are manipulation-indistinguishable iff s(x, a) = s(x', a) for all a. To first order, x and x + dx are indistinguishable iff dx lies in the null space N = ker( dM/dx ) evaluated over the action family. Therefore any readout of pose or shape is determined only on the complement of N, and the maximum-entropy representative

  x_hat = argmax_{x in [x]} H(x)   subject to  M(x, .) = M'(z, .)

is the honest estimate that adds no unidentifiable detail. Physical interpretation: N is the set of geometric changes that leave the wrench response invariant, for example rotations within a symmetry or back-surface variation no gripper contacts. The theorem tells us exactly how much pose accuracy is recoverable and declares the rest unmeasurable, which no accuracy-driven method acknowledges.

Value-of-information functional (C3). Define epistemic ambiguity U(o) = E_{a}[ Var_{z ~ q(z|o)} success(M'_psi(z, a)) ]. For a sensing action a_sense with predicted observation o' , the expected ambiguity reduction is

  IG(a_sense) = U(o) - E_{o'}[ U(o cup o') ],

and active perception selects a_sense* = argmax IG. Test-time adaptation, after a probe returns an outcome y_obs under action a_probe, updates the belief by

  min_{z}  - log M'_psi(y_obs | z, a_probe)  +  KL( q(z | o) || prior ),

a few gradient steps that make z consistent with what the interaction revealed. Both optimize the same quantity, the reduction of outcome-class ambiguity, one by choosing the measurement and the other by assimilating it. This is value of information computed in outcome space, which is why it targets manipulation-relevant disambiguation rather than geometric coverage.

Conformal outcomes (C4). On a calibration set of size n, compute nonconformity scores alpha_i = 1 - hat p(y_i | z_i, a_i). For tolerance alpha, the threshold q_hat is the ceil((n+1)(1-alpha))/n empirical quantile of the scores. The action set C_alpha(o) = { a : max_z hat p(success | z, a) >= 1 - q_hat } contains a successful action with probability at least 1 - alpha under exchangeability. If C_alpha is empty, MSP abstains. Physical interpretation: the robot certifies "there exists a grasp I can execute safely" or declines, a guarantee about the world located in perception rather than assumed by the controller.

Complexity summary. Training: standard backbone training with an extra light outcome head and a simulator that supplies M offline, so no reinforcement learning loop and no large real-robot dataset. Inference: O(F) for the encoder plus O(K h) for scoring K actions, plus optional O(k h) for k adaptation steps and one extra sensing step only when U is high. No meshing, no reconstruction, no dynamics rollout.

---

## STEP 8 - Reviewer #1 Attacks: Thirty Weaknesses, Rejection Risk, and Fixes

Grounding and physics.
1. The outcome operator M is simulated; real contact physics differ. Rejection risk: the whole estimand may be sufficient only for sim. Fix: calibrate M on a small real outcome dataset and report a sim-to-real sufficiency gap explicitly.
2. Friction and mass are not observable from vision, yet outcomes depend on them. Risk: z cannot be sufficient in principle. Fix: define sufficiency as outcome-distribution sufficiency, which correctly returns high epistemic uncertainty when physics is unobservable, and let active touch resolve it.
3. Differentiable simulators are inaccurate for contact-rich events. Risk: mislabeled outcomes. Fix: use analytic grasp-quality plus a validated rigid-body simulator, and bound label noise.
4. The identifiability theorem is first-order (Jacobian). Risk: large deformations violate the linearization. Fix: state it locally, and estimate the equivalence class numerically for finite perturbations.
5. Outcomes reduced to success, stability, slip may omit task-relevant modes. Risk: hidden insufficiency. Fix: make the outcome vector task-configurable and show robustness to its choice.

Novelty and positioning.
6. Task-driven information bottleneck already exists (Pacelli-Majumdar). Risk: perceived as incremental. Fix: foreground the policy-free action-family estimand, the identifiability theorem, and the unified active/TTA functional, none of which they provide.
7. Affordance learning already couples task and perception. Risk: seen as affordances rebranded. Fix: show affordances lack a sufficiency definition, an identifiability limit, and calibration.
8. End-to-end grasping is implicitly task-driven perception. Risk: "you formalized what AnyGrasp already does." Fix: demonstrate the explicit belief, abstention, and active perception that end-to-end regressors cannot express.
9. The quotient-manifold framing may read as philosophy, not method. Risk: dismissed as hand-waving. Fix: tie every claim to the two concrete learned modules and measurable quantities.
10. "No pose target" may be seen as giving up a useful signal. Risk: reviewers want geometry. Fix: offer the optional readout and show geometry helps only as a regularizer, not as the estimand.

Evaluation.
11. There is no standard benchmark for outcome-sufficiency. Risk: unfalsifiable claims. Fix: define the metric (predicted-vs-true outcome divergence) and evaluate on execution success, not pose error.
12. Comparing to pose methods on task success may be unfair to them. Risk: strawman accusation. Fix: give baselines the same action sampler and compute budget.
13. Real-robot trials are expensive and may be few. Risk: anecdotal evidence. Fix: preregister object counts, seeds, and confidence intervals; report per-attempt success.
14. Success is confounded by the gripper and controller. Risk: perception credit unclear. Fix: hold controller fixed across methods and ablate.
15. Calibration guarantees assume exchangeability that manipulation violates. Risk: guarantee is vacuous. Fix: use weighted or adaptive conformal prediction under covariate shift and report empirical coverage.

Method internals.
16. The variational bound is loose; minimality may not hold. Risk: z not actually minimal. Fix: report rate-distortion curves and compare to compressed baselines.
17. Beta is a free knob; results may be cherry-picked. Risk: tuning artifact. Fix: sweep beta and report the full frontier.
18. Epistemic uncertainty from a variational encoder is often miscalibrated. Risk: active perception and TTA misfire. Fix: use a small deep ensemble and validate calibration.
19. Active probes perturb the scene, changing x. Risk: the belief update is inconsistent. Fix: model the probe as a known state transition and update through it.
20. Action space is discretized for scoring; optimum may be missed. Risk: suboptimal grasps. Fix: coarse-to-fine action refinement, report sensitivity to K.
21. The outcome head could ignore z and memorize action priors. Risk: z is bypassed. Fix: an information-flow penalty and a z-ablation control.
22. Language-conditioned sufficiency budget is under-specified. Risk: the P5 claim is thin. Fix: either fully develop the language-to-beta mapping or scope it out of the core paper.
23. Test-time adaptation may diverge or overfit one probe. Risk: instability. Fix: trust-region update on z only, bounded steps.
24. The framework assumes a single object of interest. Risk: multi-object scenes unhandled. Fix: per-object sufficiency with a shared encoder and inter-object outcome terms.
25. Deformable and articulated objects break the fixed action-outcome map. Risk: limited scope. Fix: scope to rigid in V1, state the extension.

Scalability and systems.
26. Foundation-model encoder is heavy for a mobile platform. Risk: not deployable. Fix: report latency and a distilled encoder variant.
27. Generating M over a dense action grid at training is costly. Risk: training does not scale. Fix: importance-sample actions near the outcome decision boundary.
28. The approach may only shine on hard, ambiguous objects. Risk: marginal average gains. Fix: report stratified results and argue the frontier, not the mean.
29. Sim-to-real of physical response, not appearance, is unproven. Risk: central bet unvalidated. Fix: the real calibration study in fix 1 is the paper's decisive experiment.
30. The contribution may be seen as theory without enough hardware. Risk: T-RO wants systems. Fix: a compact but real mobile-manipulation demonstration with active touch and abstention.

The three attacks that decide acceptance: 1 and 29 (sim-to-real of physical response), 6 (delta over task-driven IB), and 11 (a fair, outcome-grounded evaluation). V2 is designed around these.

---

## STEP 9 - Framework V2

V2 keeps the two-module minimal core and hardens the three decisive fault lines.

Change 1, physics grounding with a small real calibration set. Train M'_psi on a large simulated outcome corpus, then fit a low-parameter residual on a few hundred real grasp outcomes so the outcome head is calibrated to real contact physics. This is a small labeled set of grasp results, not massive teleoperation data, and it directly answers attacks 1, 3, and 29. Report the sim-to-real sufficiency gap as a headline number.

Change 2, embodiment-parameterized sufficiency. Condition M'_psi on a gripper descriptor g_grip so the estimand is sufficiency with respect to a gripper family, not one gripper. This removes the embodiment-relativity objection (attack 8) and yields a cross-embodiment claim.

Change 3, interaction as a modeled state transition. When a probe perturbs the scene, update the belief through a known transition T(x, a_probe), so active touch remains consistent (attack 19). Interaction becomes a first-class, physics-consistent measurement (philosophy P3 realized correctly).

Change 4, calibrated and shift-robust output. Replace vanilla conformal with weighted adaptive conformal under covariate shift, and validate empirical coverage on held-out real scenes (attack 15).

Change 5, honest evaluation protocol. Primary metric is execution success per attempt under a fixed controller and action sampler shared by all baselines; secondary metric is outcome-prediction divergence; pose error is reported only as a diagnostic, never as the objective (attacks 11, 12, 14). Stratify by object difficulty and report the frontier.

Change 6, minimality evidence. Report rate-distortion frontiers over beta and z-ablation controls to prove z is used and near-minimal (attacks 16, 17, 21).

Change 7, deployability. Provide a distilled encoder and full latency accounting on a mobile base, with active touch and abstention in the loop (attacks 26, 30).

Everything else, the two learned modules and the inference-time derivation of grasp selection, active perception, TTA, and abstention, is unchanged. V2 is the same idea made falsifiable and deployable.

---

## STEP 10 - Philosophy-Level Comparison with the Last Five Years

Comparison is of scientific assumptions, not architectures.

End-to-end grasp regression (Dex-Net, GraspNet-1Billion, AnyGrasp). They assume the mapping from observation to a good grasp can be learned directly, with perception implicit and uncertainty absent. MSP assumes the estimand is an explicit calibrated belief over a sufficient statistic, so grasp selection, abstention, and active sensing are derived rather than baked into a single regressor. The difference is that MSP exposes and reasons about what the regressor hides.

Modular reconstruction then plan (the source paper, FoundationPose, NOCS, SceneComplete). They assume perception should recover accurate observer-independent geometry, and that better geometry monotonically improves grasping. MSP assumes geometry is recoverable only up to manipulation-indistinguishability and that accuracy past the sufficiency resolution is wasted. This is a direct denial of the source paper's thesis, replacing "better pose and shape yields better grasping" with "sufficient outcome-prediction yields better grasping, and accuracy is neither necessary nor sufficient."

Task-driven information bottleneck (Pacelli-Majumdar; Bai et al. for behavior cloning). They assume a fixed task or policy and compress state toward its scalar reward, typically with reinforcement learning. MSP assumes sufficiency with respect to the entire action-conditioned outcome family, policy-free and supervised by physics, and adds an identifiability theorem, a unified active-and-adaptation functional, and calibration. The estimand is broader and the object of estimation is a quotient geometry, not a policy feature.

Affordance and keypoint methods (ReKep, RAM, Grasp2Vec). They assume task-relevant structure can be attached to or retrieved for a scene, but still treat geometry or keypoints as the substrate and provide no sufficiency criterion or uncertainty guarantee. MSP defines the substrate itself by physics of action and quantifies exactly what is unidentifiable.

Feature fields (F3RM, GraspSplats, LERF-TOGO). They assume a dense geometric field decorated with semantic features is the right intermediate. MSP assumes the intermediate should be the minimal outcome-sufficient statistic, which is typically far lower-dimensional and carries calibrated uncertainty a field does not.

Vision-language-action models (OpenVLA-style and successors). They assume a single network mapping pixels and language to actions, with perception, sufficiency, and uncertainty all implicit. MSP assumes an explicit sufficiency estimand and certificate, and can be read as giving VLAs a principled perceptual bottleneck with abstention rather than an opaque one.

Diffusion grasping. It assumes a generative model over grasps captures multimodality. MSP assumes the decisive object is the outcome-sufficient belief; multimodality in grasps is a downstream consequence of ambiguity in that belief, not the primary estimand.

The single sentence a distinguished editor should take away: every line above estimates either accurate geometry or a task-specific policy feature, whereas MSP estimates the physics-defined equivalence class of world states that manipulation cannot tell apart, and proves how much of geometry that leaves undetermined.

---

## STEP 11 - Journal Paper Blueprint

Title. Perception as Sufficiency, not Accuracy: Estimating the Manipulation-Indistinguishability Class for Grasp Planning.

Abstract. Object pose and shape estimation for grasping is formulated as recovering accurate, observer-independent geometry, and recent work concludes that better geometry yields better grasps. We show this estimand is both ill-posed and wasteful: for manipulation, two world states are equivalent when every action produces the same outcome, and geometry is recoverable only up to this equivalence. We reformulate perception as estimating a calibrated belief over a manipulation-sufficient statistic, the minimal code that preserves all and only the information that changes action outcomes, trained by a policy-free action-conditioned information bottleneck whose distortion lives in outcome space rather than geometry. From this single estimand we derive grasp selection, distribution-free abstention, active perception, and test-time adaptation as inference procedures over two learned modules, an encoder and a physics-grounded outcome head, with no reconstruction, no dynamics rollout, and no reinforcement learning. We prove that pose and shape are identifiable only on the complement of the outcome Jacobian null space, quantifying exactly how much accuracy is recoverable. On simulated and real mobile-manipulation experiments the framework matches or exceeds accuracy-driven modular and end-to-end baselines at a fraction of the representation dimension and latency, with the largest gains on physically ambiguous objects where accurate geometry is unrecoverable and irrelevant.

Motivation. Accuracy-driven perception spends capacity on geometry no action is sensitive to, discards the uncertainty needed for safe action, and reports metrics that misrank methods on real manipulation. The field needs an estimand defined by what actions do.

Research gap. No existing formulation defines the perceptual estimand for manipulation as a physics-grounded equivalence class, characterizes its identifiability limit, and derives active perception, adaptation, and calibrated abstention from that single definition without reinforcement learning or large real datasets.

Scientific hypothesis. A representation that is minimal and sufficient for the action-conditioned outcome family will match or exceed accuracy-driven representations on grasp success while being lower-dimensional, better-calibrated, and cheaper, and its residual pose-shape ambiguity will coincide with the outcome Jacobian null space.

Method overview. Encode observation to a belief over a sufficient statistic; score actions with a physics-grounded outcome head; select, abstain, look, or touch by reusing these two modules.

Pipeline diagram (ASCII).
```
            +-----------------------------+
 o (RGB-D,  |  A: probabilistic           |   q(z|o)
 language) ->|     sufficiency encoder     |----------+
            +-----------------------------+          |
                                                     v
                              +---------------------------------------+
   candidate actions a  --->  |  B: physics-grounded outcome head     |
   (SE3 x gripper, g_grip)    |     M'(y | z, a)                      |
                              +---------------------------------------+
                                     |            |            |
                          argmax_a   |   conformal|   Var_z    |
                          expected   |   set C_a  |  ambiguity |
                          success    |  (abstain) |     U      |
                                     v            v            v
                                  grasp a*     certificate  act to reduce U:
                                                            next view / touch
                                                            -> update q(z|o)  (TTA)
```

Core algorithm (inference).
```
function MSP_ACT(o, actions, gripper, alpha, tau):
    q_z <- Encoder(o)                       # belief over sufficient statistic
    for a in actions:
        mu[a], var[a] <- outcome_stats(OutcomeHead, q_z, a, gripper)
    U <- mean_a var[a]                       # epistemic ambiguity
    if U > tau:                              # active perception
        a_sense <- argmax_sense InfoGain(q_z, sense_actions)
        o <- o + observe(a_sense)            # view, or probe via known transition
        return MSP_ACT(o, actions, gripper, alpha, tau)   # with TTA belief update
    C <- conformal_set(mu, var, alpha)       # calibrated feasible actions
    if C is empty: return ABSTAIN
    return argmax_{a in C} ( mu[a] - lambda * var[a] )
```

Pseudo-code (training).
```
for batch (o_x, x) in D:
    sample actions a ~ importance_near_boundary(x)
    y_true <- M(x, a)                        # physics operator: FC margin, epsilon-quality, slip
    z ~ Encoder(o_x)                         # reparameterized
    y_pred <- OutcomeHead(z, a, g_grip)
    L_suff  <- - E[ log y_pred(y_true) ]     # sufficiency (distortion)
    L_rate  <- KL( q(z|o_x) || r(z) )        # minimality (rate)
    L       <- L_suff + (1/beta) * L_rate
    step(theta, psi; grad L)
# then: fit small real residual on real grasp outcomes; calibrate conformal
```

Training procedure. Stage 1, train encoder and outcome head on a simulated outcome corpus with boundary-focused action sampling and a beta sweep. Stage 2, fit a low-parameter real residual on a few hundred real grasp outcomes. Stage 3, calibrate the weighted conformal predictor on held-out real scenes. No RL, no large teleoperation set.

Inference procedure. As in the core algorithm: encode, score, and then select, abstain, or act to reduce ambiguity with a physics-consistent belief update.

Loss functions. Sufficiency distortion (negative outcome log-likelihood), minimality rate (KL to prior), optional geometry regularizer as a weak auxiliary, and the real-residual calibration loss. The frontier over beta is reported, not a single point.

Complexity analysis. Encoder O(F) once; outcome scoring O(K h) for K actions; optional active step and O(k h) adaptation only under high ambiguity. Representation dimension d is the intrinsic manipulation-relevant dimension, far below mesh or feature-field size, giving the latency and memory advantage over reconstruction and field methods.

Theoretical justification. Rate-distortion optimality of the minimal sufficient code; the identifiability theorem locating residual pose-shape ambiguity in the outcome Jacobian null space; distribution-free coverage of the weighted conformal predictor under covariate shift.

Expected experimental results. [speculation] Grasp success matching or exceeding modular-analytic and end-to-end baselines at a fraction of representation dimension and latency; the gap widening on physically ambiguous objects (transparent, off-center mass, symmetric) where accurate geometry is unrecoverable; abstention converting most catastrophic failures into declines; active touch resolving mass and friction ambiguities that no passive view can; empirical conformal coverage close to nominal. These are hypotheses to be measured, not claims.

Ablation studies. Remove the rate term (sufficiency only), remove epistemic uncertainty (no active or TTA), remove the real residual (sim-only physics), remove abstention, vary beta across the frontier, z-ablation to confirm the outcome head uses z, action budget K sensitivity, and cross-embodiment via the gripper descriptor.

Failure analysis. Physics unobservable even after touch (truly hidden mass); simulator-to-real residual insufficient for exotic materials; conformal coverage degradation under severe shift; probes that irreversibly change the scene; multi-object contact coupling beyond the per-object outcome terms.

Future extensions. Deformable and articulated objects via a state-dependent action-outcome map; language-defined sufficiency budgets developed fully; sufficiency-aware VLA where the policy consumes the calibrated belief; long-horizon manipulation where sufficiency is defined per subgoal.

Expected journal. IEEE Transactions on Robotics as first choice for the formulation-plus-hardware profile; IEEE T-ASE or Robotics and Computer-Integrated Manufacturing as strong alternatives given the automation and deployment angle.

Expected reviewer questions. How is M validated against real physics? What is the sim-to-real sufficiency gap? How does the delta over task-driven information bottleneck hold up quantitatively? Is the evaluation fair to pose baselines under a shared controller? Does the identifiability theorem survive beyond the first-order regime? What is the latency on a real mobile base? Does calibration coverage hold under the shifts you actually encounter?

Expected novelty score. 8 of 10. It changes the estimand and proves an identifiability limit, which is rare, but it builds on an existing task-driven information-bottleneck lineage, which a careful reviewer will weigh against the top scores.

Expected acceptance probability. [speculation, calibrated] First submission to T-RO: roughly 25 to 35 percent for direct acceptance or minor revision, with major revision the most likely outcome, because the decisive real-physics sufficiency experiment carries the paper and reviewers will demand it be airtight. Conditional on a convincing real calibration study and a fair shared-controller evaluation, acceptance after major revision rises to roughly 60 to 70 percent. Without the real-physics validation, expect rejection regardless of the theory.

