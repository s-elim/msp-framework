# T-RO Review Simulation: Manipulation-Sufficient Perception (MSP)

Manuscript under review: "Perception as Sufficiency, not Accuracy: Estimating the Manipulation-Indistinguishability Class for Grasp Planning." Submitted version is MSP as specified in the blueprint (two learned modules, physics-grounded outcome operator, identifiability theorem, calibrated abstention, active perception and test-time adaptation as inference procedures).

Scientific hypothesis to be preserved across all iterations, verbatim: a representation that is minimal and sufficient for the action-conditioned outcome family matches or exceeds accuracy-driven representations on grasp success while being lower-dimensional, better calibrated, and cheaper, and its residual pose-shape ambiguity coincides with the outcome Jacobian null space.

Convention: reviewer recommendations use the T-RO scale (Accept, Minor Revision, Major Revision, Reject). Confidence is 1 to 5. All predicted experimental outcomes are [speculation]; no fabricated numbers are stated as results. Named baselines and methods are real.

---

# ITERATION 1

## Reviewer A (object pose estimation)

**Summary.** The paper argues that estimating accurate object pose and shape is the wrong objective for grasping and replaces it with estimating a calibrated belief over a manipulation-sufficient statistic, defined as a sufficient statistic for the action-conditioned outcome family. It claims pose and shape are identifiable only up to a physics-defined equivalence, with residual ambiguity equal to the null space of the outcome Jacobian, and derives grasp selection, abstention, active perception, and adaptation from two learned modules.

**Strengths.** The reframing of the estimand is principled and, unlike most grasping papers, takes identifiability seriously. Locating unrecoverable pose degrees of freedom in the null space of dM/dx is an elegant and, to my knowledge, novel statement in the manipulation-perception literature. The refusal to supervise on pose or mesh labels is intellectually consistent and sidesteps the canonical-frame problem that plagues category-level pose estimation.

**Weaknesses.** (1) The identifiability theorem is first-order. Object symmetries and large occlusions induce discrete, non-infinitesimal ambiguities that a Jacobian null space does not capture, so the central theoretical claim is incomplete. (2) Symmetry is not treated explicitly, yet it is the dominant source of pose ambiguity for the very objects grasping cares about (cylinders, boxes, bottles). (3) The paper asserts geometry beyond the sufficiency resolution is unrecoverable but never validates this against a modern pose estimator; without a diagnostic on a standard pose benchmark the claim is rhetorical. (4) The optional pose readout is described but its honesty (maximum-entropy representative) is not measured against ground truth, so the reader cannot judge what is actually lost.

**Questions.** Does the equivalence relation contain the object symmetry group as a subgroup, and if so is that made explicit in the encoder? How does the theorem extend to discrete ambiguities? What is the measured pose error of the readout on a benchmark such as YCB-Video, decomposed into the null-space and its complement?

**Required experiments.** A diagnostic on a standard 6D pose benchmark (BOP or YCB-Video) showing the readout error concentrates on the predicted null space; a symmetry stress test on symmetric objects; a controlled study relating reconstruction error to grasp success confirming the saturation claim.

**Missing baselines.** A strong modern pose-plus-grasp modular baseline (FoundationPose plus antipodal sampling, and NOCS category-level plus sampling), which is exactly the family the paper aims to displace.

**Novelty assessment.** High on the estimand and identifiability framing; moderate given the task-driven information bottleneck lineage the authors themselves cite.

**Technical correctness.** The variational bound and rate-distortion reading are correct. The identifiability theorem as stated is correct only locally; the paper overstates its generality.

**Confidence.** 4.

**Recommendation.** Major Revision.

## Reviewer B (robot grasping)

**Summary.** The manuscript predicts grasp outcomes from a compressed perceptual belief rather than planning on a reconstructed mesh, trains the outcome head against a physics operator, and adds abstention and active touch. It claims to match or beat accuracy-driven modular and end-to-end grasping at lower dimension and latency.

**Strengths.** The outcome-grounded objective is the right instinct: grading perception by what actions do rather than by geometric error is a real improvement over the modular pipelines that dominate. Abstention with a certificate and active touch to resolve mass and friction are genuinely useful for deployment, and the framework gets them from one principle. The absence of a reconstruction and meshing stage is a plausible latency win.

**Weaknesses.** (1) The outcome operator M is simulated, and contact-rich outcomes are exactly where simulators are least trustworthy; the paper's entire estimand is only as good as M, and its fidelity to real contact is unvalidated. (2) The real calibration set of a few hundred grasps is too small to cover the physical-response distribution the method claims to capture. (3) Evaluation is not on a community protocol, so results are not comparable; GraspNet-1Billion has a defined protocol and should be used, with real-robot trials reporting seeds and confidence intervals. (4) Force closure and slip depend on friction and mass that no passive view observes, so the claim that a single-view belief suffices needs the active-touch pathway to be demonstrated, not just described. (5) Clutter and bin picking, where grasping is hard, are absent.

**Questions.** What is the measured sim-to-real gap of M on real grasps? How many real trials, on how many objects, with what variance? Does the method actually trigger active touch, and how often does it help? How is success defined per attempt, and is the controller held fixed across baselines?

**Required experiments.** GraspNet-1Billion under its protocol; real-robot trials with preregistered object sets, at least tens of objects, multiple seeds, and confidence intervals; an active-touch ablation isolating its contribution; a clutter and bin-picking study; a shared-controller fair comparison.

**Missing baselines.** AnyGrasp, Contact-GraspNet, and Dex-Net 4.0 as end-to-end references, plus the modular FoundationPose-plus-sampling pipeline. Without these the comparison is not credible.

**Novelty assessment.** Moderate to high. The outcome-sufficiency estimand is new for grasping, but end-to-end grasping already learns an implicit task-driven perception, so the delta must be shown empirically.

**Technical correctness.** The formulation is sound. The empirical grounding of M is the weak link and currently unproven.

**Confidence.** 5.

**Recommendation.** Major Revision.

## Reviewer C (foundation models and embodied AI)

**Summary.** The paper reformulates manipulation perception as estimating a minimal sufficient statistic for the action-conditioned outcome family, with calibration, active perception, and test-time adaptation derived from the same object. It positions itself against task-driven information bottlenecks and affordance methods.

**Strengths.** The conceptual contribution is strong and timely: it gives an explicit, calibrated perceptual bottleneck with abstention, which the current vision-language-action wave lacks. Deriving active perception and adaptation from one value-of-information functional is clean. The identifiability limit is the kind of result that reframes how a subfield measures progress.

**Weaknesses.** (1) The delta over task-driven information bottlenecks is argued in prose but not quantified; a reviewer needs an ablation that instantiates the single-task IB and shows the action-family sufficiency and identifiability actually buy something. (2) A vision-language-action baseline is absent; the paper claims relevance to that line but does not compare or integrate. (3) The conformal guarantee assumes exchangeability, which manipulation violates through distribution drift; the guarantee as stated may be vacuous in deployment. (4) The language-defined sufficiency budget (philosophy P5) is asserted but not developed or tested. (5) The encoder is a foundation-model backbone whose latency on a mobile base is not reported, undercutting the efficiency claim.

**Questions.** How much does the action-family estimand improve over a single-task IB quantitatively? Does a VLA benefit from consuming the MSP belief? What coverage does the conformal predictor actually achieve under measured shift? What is end-to-end latency on the target platform?

**Required experiments.** A single-task IB ablation; a VLA comparison and, ideally, an integration where the VLA consumes the calibrated belief; conformal coverage measured under real distribution shift; a latency and compute table; a first experiment on the language-budget mechanism or its removal from the paper's claims.

**Missing baselines.** A task-driven IB instantiation (Pacelli-Majumdar style), an OpenVLA-style policy, and an affordance-transfer method (for example a retrieval-based affordance baseline).

**Novelty assessment.** High on framing, contingent on the quantified delta over prior task-driven representations.

**Technical correctness.** Sound, with the exchangeability assumption as the notable soft spot.

**Confidence.** 4.

**Recommendation.** Major Revision.

## Associate Editor meta-review (Iteration 1)

Three independent Major Revisions with high confidence. The reviewers agree the reformulation is significant and the identifiability framing is a real contribution, and they agree the manuscript is not yet acceptable for three concrete reasons that recur across all three reports.

Decisive issues. First, the physical grounding of the outcome operator is unvalidated (B, and implicitly A and C): the estimand is defined by M, so M's fidelity to real contact is the paper. Second, the theory is only local and ignores symmetry (A), which is the dominant ambiguity for graspable objects. Third, novelty is contingent on a quantified delta over task-driven information bottlenecks and a fair comparison to end-to-end grasping and VLA (C and B).

Decision: Major Revision. The scientific hypothesis is sound and worth pursuing; the paper must be rebuilt around evidence, not argument.

Redesign directives to the authors. (i) Ground M with a validated multi-source outcome operator and a real calibration study, and report the sim-to-real sufficiency gap as a headline result. (ii) Upgrade the identifiability theorem to handle symmetry and discrete ambiguity via a group-theoretic treatment and a certified finite-perturbation bound, and make the encoder symmetry-aware. (iii) Adopt community benchmarks and a fair shared-controller protocol, and add the full baseline set. (iv) Quantify the delta over single-task IB with a direct ablation, add a VLA comparison, and replace vanilla conformal with a shift-robust variant. The hypothesis is preserved; only the evidence and the theory's scope change.

## Redesign: Framework V3

V3 keeps the two-module core and the hypothesis, and hardens the three fault lines.

Grounding (answers B, A4, C3). The outcome operator becomes three-tier: an analytic wrench-space quality (Ferrari-Canny epsilon-metric and force-closure margin) for a differentiable prior, a validated rigid-body contact simulator with pinned versions for dynamic slip and perturbation, and a real residual model fit by a Gaussian process on autonomously collected real grasp outcomes. The robot labels its own grasps by executing and observing lift and post-lift deviation, so the real set grows to low thousands without human teleoperation, and boundary-focused sampling concentrates labels near the outcome decision surface. The reported headline quantity is the sim-to-real sufficiency gap, the drop in outcome-prediction fidelity from simulated to real grasps.

Theory upgrade (answers A1, A2). The equivalence relation is defined as invariance of the outcome map under the object symmetry group G_sym and the observation-induced ambiguity. The identifiability statement becomes two-part: locally, the tangent space of the equivalence class is ker(dM/dx); globally, discrete ambiguities are exactly the orbits of G_sym acting on states that leave M invariant, and a certified finite-perturbation bound is obtained from the Lipschitz constant of the outcome head via randomized smoothing. The encoder is made symmetry-equivariant so a symmetric object yields a belief supported on a group orbit rather than a collapsed mean.

Evaluation and baselines (answers A3, B3, B-missing, C-missing). GraspNet-1Billion under its published protocol, EGAD for shape and difficulty diversity, and real-robot trials with preregistered object sets, multiple seeds, and confidence intervals. All baselines share the controller and action sampler. Baseline set: AnyGrasp, Contact-GraspNet, Dex-Net 4.0, a modular FoundationPose-plus-antipodal pipeline, NOCS-plus-sampling, a single-task task-driven IB, and an OpenVLA-style policy.

Novelty and calibration (answers C1, C4). A single-task IB is implemented as a first-class ablation to quantify the action-family and identifiability gains. Vanilla conformal is replaced by weighted, Mondrian-style conformal to handle covariate shift, with empirical coverage reported. The language-budget claim is either developed with a concrete per-outcome-dimension beta schedule and one experiment, or explicitly scoped out; V3 scopes it to a clearly labeled preliminary study to avoid overclaiming.

Hypothesis check. Unchanged. V3 alters evidence and the theorem's scope, not the claim.

---

# ITERATION 2

Reviewers received V3 with the revision and rebuttal. Concerns are reduced but not eliminated.

## Reviewer A (object pose estimation)

**Summary.** The revision adds a group-theoretic identifiability treatment, a symmetry-equivariant encoder, a certified finite-perturbation bound via randomized smoothing, and a pose-benchmark diagnostic.

**Strengths.** The two-part identifiability statement, local tangent space equal to the outcome Jacobian null space and global discrete ambiguity as symmetry orbits, is now correct and genuinely novel; I have not seen manipulation-relevant identifiability characterized this way. The symmetry-equivariant encoder is the right construction and resolves my dominant concern. The commitment to a BOP-style diagnostic is welcome.

**Weaknesses.** (1) The certified bound rests on a Lipschitz constant of the outcome head that is typically loose; a certificate that is technically valid but numerically vacuous does not support the strong unrecoverability claim. (2) The claim that readout error concentrates on the null space is stated as an expected result but the verification is indirect; I want a direct measurement of the principal angles between the empirically grasp-invariant perturbation subspace and the predicted null space. (3) Non-rigid objects are still out of scope, which is acceptable for this paper but should be stated as a boundary of the identifiability result, since deformation changes dM/dx.

**Questions.** What is the measured tightness of the randomized-smoothing certificate? Do the empirical grasp-invariant directions align with the predicted null space, quantitatively?

**Required experiments.** A subspace-alignment experiment (principal angles) between measured invariant perturbations and the predicted null space; a report of certificate tightness against an empirical lower bound.

**Missing baselines.** None additional; the pose-modular baselines requested earlier now suffice.

**Novelty assessment.** High. The identifiability contribution is now both correct and distinctive.

**Technical correctness.** The theory is now correct. The remaining risk is that the certificate is too loose to be meaningful, which is an empirical, not a logical, gap.

**Confidence.** 4.

**Recommendation.** Minor Revision.

## Reviewer B (robot grasping)

**Summary.** The revision grounds the outcome operator in analytic quality, a validated simulator, and an autonomously collected real residual, adopts GraspNet-1Billion and EGAD with a shared controller, and adds the full baseline set and an active-touch ablation.

**Strengths.** The three-tier outcome operator and the headline sim-to-real sufficiency gap are exactly what was missing; grading the estimand by real outcome fidelity is now credible. Adopting the community protocol and holding the controller fixed makes the comparison fair. The autonomous self-labeling avoids the massive-teleoperation objection.

**Weaknesses.** (1) Autonomous self-labeling introduces selection bias: the robot collects outcomes where its current policy already ventures, so the residual model is well-fit where the method already works and blind where it fails; this can inflate the reported sufficiency. (2) Real-trial counts, while improved, are still modest for the strong generalization claims; long-tail materials (transparent, deformable-surfaced) are under-sampled. (3) Embodiment generalization is argued through a gripper descriptor but demonstrated only on parallel jaws; a multi-finger demonstration would substantiate the cross-embodiment claim. (4) Failure attribution is reported qualitatively; a counterfactual analysis separating perception failure from control failure would strengthen it.

**Questions.** How is the self-labeling selection bias corrected? Does the gripper descriptor transfer to a multi-finger hand without retraining M from scratch? What fraction of failures are perceptual versus control?

**Required experiments.** Importance-weighted or information-gain-driven data collection with a bias-correction analysis; a multi-finger embodiment study; counterfactual failure attribution; expanded long-tail material trials.

**Missing baselines.** None additional; add only a multi-finger grasping reference for the embodiment study.

**Novelty assessment.** High for grasping now that the outcome grounding is real.

**Technical correctness.** Sound. The selection-bias issue is the one substantive threat to the empirical claims.

**Confidence.** 5.

**Recommendation.** Major Revision.

## Reviewer C (foundation models and embodied AI)

**Summary.** The revision quantifies the delta over single-task IB with a direct ablation, adds a VLA comparison, replaces vanilla conformal with weighted Mondrian conformal, and scopes the language-budget claim to a preliminary study.

**Strengths.** The single-task IB ablation is the right control and, if the predicted gains hold, settles the novelty question against the closest prior work. Shift-robust conformal is the correct fix. Scoping the language claim honestly rather than overclaiming is appreciated and raises my confidence in the rest.

**Weaknesses.** (1) The VLA appears only as a competing baseline; the more compelling and more publishable claim is that MSP improves a VLA by supplying it a calibrated bottleneck with abstention, which the revision gestures at but does not demonstrate. (2) Weighted Mondrian conformal handles covariate shift only under known or estimable weights; under the label shift that manipulation also exhibits, marginal coverage can still fail, so an adaptive online scheme with a coverage guarantee under drift would be stronger. (3) Compute: the symmetry-equivariant foundation encoder is heavier than V2; the efficiency claim now needs a distilled variant with a latency table on the mobile platform to remain credible.

**Questions.** Does a VLA that consumes the MSP belief outperform the same VLA without it? What coverage does the conformal scheme guarantee under measured drift, not just covariate shift? What is deployed latency after distillation?

**Required experiments.** A VLA-plus-MSP integration showing improvement over the VLA alone; adaptive conformal inference with a drift guarantee and measured coverage; a distilled-encoder latency and compute table on the target platform.

**Missing baselines.** The same VLA with and without the MSP belief, as an integration study rather than a bake-off.

**Novelty assessment.** High, now nearly settled pending the quantified IB delta.

**Technical correctness.** Sound. Coverage under label shift is the remaining soft spot.

**Confidence.** 4.

**Recommendation.** Minor Revision.

## Associate Editor meta-review (Iteration 2)

Movement is clear: one Major and two Minor, versus three Majors previously. Reviewer A's central theoretical objection is resolved and he now judges the identifiability contribution correct and distinctive. Reviewer C is close to satisfied, contingent on the IB delta and a VLA integration. Reviewer B, the most demanding on evidence, raises one genuinely important new issue that must not be waved off: autonomous self-labeling can bias the real outcome model toward regions where the method already succeeds, which would inflate the headline sufficiency result.

Decisive remaining issues. First, self-labeling selection bias (B), which threatens the empirical core. Second, empirical verification that the predicted null space matches measured grasp-invariant directions (A), which is what turns the identifiability theorem from correct to demonstrated. Third, the VLA integration and drift-robust calibration (C), which convert the paper from competitive to influential.

Decision: Major Revision, trending positive. One reviewer's evidence concern is serious enough to withhold Minor.

Redesign directives. (i) Replace passive self-labeling with information-gain-driven autonomous collection plus importance weighting, and report a bias-correction analysis; add multi-finger embodiment and counterfactual failure attribution. (ii) Directly verify identifiability by measuring principal angles between empirically grasp-invariant perturbations and the predicted null space, and report certificate tightness. (iii) Demonstrate MSP improving a VLA, and upgrade calibration to adaptive conformal inference with a coverage guarantee under drift; deliver a distilled-encoder latency table. Hypothesis preserved.

## Redesign: Framework V4

Debiased grounding (answers B1, B2, B4). Real outcome collection is driven by expected information gain about the outcome-class belief, deliberately sampling actions and objects near and beyond the current decision boundary rather than where the policy already succeeds. Collected outcomes are importance-weighted by the ratio of the collection distribution to a target uniform-over-difficulty distribution, and the residual model is refit with these weights, removing the optimism of passive self-labeling. A bias-correction analysis reports the sufficiency gap before and after weighting. A counterfactual failure attribution executes the same perceptual belief through an oracle controller and the same controller through an oracle belief, separating perception from control error. Long-tail material trials are expanded.

Empirical identifiability (answers A1, A2). A perturbation-invariance probe measures the subspace of state perturbations that leave real grasp outcomes unchanged, and reports the principal angles between that measured subspace and the predicted ker(dM/dx). Tight alignment is the direct evidence the theorem predicts. Certificate tightness is reported against an empirical lower bound obtained by adversarial perturbation search, replacing reliance on a possibly loose Lipschitz constant with a measured gap.

Embodiment (answers B3). The gripper descriptor is exercised on a multi-finger hand by extending the action space and the outcome operator to k-contact closure, and cross-embodiment transfer is measured without retraining the encoder.

VLA integration and calibration (answers C1, C2, C3). MSP is integrated into a VLA by supplying the calibrated sufficiency belief and the abstention signal as additional policy inputs, and the integrated policy is compared to the same VLA without them. Calibration is upgraded to adaptive conformal inference, which maintains long-run coverage under arbitrary drift by adjusting the threshold online, with measured coverage reported. A distilled encoder is trained and a full latency and compute table on the mobile platform is provided.

Hypothesis check. Unchanged. V4 adds bias correction, direct identifiability evidence, embodiment breadth, a VLA integration, and drift-robust calibration, all in service of the same claim.

---

# ITERATION 3

Reviewers received V4. The decisive concerns from prior rounds are addressed.

## Reviewer A (object pose estimation)

**Summary.** V4 adds the subspace-alignment experiment between measured grasp-invariant perturbations and the predicted outcome Jacobian null space, and reports certificate tightness against an adversarial lower bound.

**Strengths.** The principal-angle alignment is the experiment I asked for and it turns the identifiability theorem from a correct statement into a demonstrated property, which is the strongest form of the contribution. Reporting the certified-versus-empirical gap is honest and unusual in this literature. The symmetry-equivariant treatment is now complete for rigid objects.

**Weaknesses.** One residual limitation, acknowledged rather than fixed: the identifiability result is a rigid-object result, and the deformation regime, where dM/dx itself changes with state, is left to future work. This is a legitimate scope boundary, not a flaw, provided it is stated in the abstract rather than buried in the limitations.

**Questions.** None outstanding.

**Required experiments.** None further.

**Missing baselines.** None.

**Novelty assessment.** High and now demonstrated.

**Technical correctness.** Correct, with an appropriately stated rigid-object scope.

**Confidence.** 5.

**Recommendation.** Accept (with the abstract-level scope statement as a minor edit).

## Reviewer B (robot grasping)

**Summary.** V4 replaces passive self-labeling with information-gain-driven collection and importance weighting, reports the sufficiency gap before and after bias correction, adds a multi-finger embodiment study and counterfactual failure attribution.

**Strengths.** The bias-correction analysis directly addresses my main threat, and reporting the pre- and post-weighting sufficiency gap is exactly the transparency I wanted; the residual optimism is now measurable rather than hidden. The counterfactual attribution cleanly separates perception from control, which most grasping papers never do. Multi-finger transfer via the descriptor substantiates the cross-embodiment claim. The shared-controller protocol and confidence intervals make the results trustworthy.

**Weaknesses.** One residual: even after bias correction, the long tail of adversarial materials (fully transparent and specular objects whose real outcomes are hardest to collect) remains thinly sampled, so the generalization claim on that stratum is weaker than on the rest. This should be reported as a stratified result with an explicit caveat, not smoothed into an average.

**Questions.** Can the stratified success on the hardest material class be reported separately with its own interval?

**Required experiments.** Only the stratified reporting above; no new data campaign required.

**Missing baselines.** None.

**Novelty assessment.** High for grasping; the outcome-sufficiency estimand with demonstrated real grounding is a contribution I would cite.

**Technical correctness.** Sound.

**Confidence.** 5.

**Recommendation.** Minor Revision.

## Reviewer C (foundation models and embodied AI)

**Summary.** V4 integrates the MSP belief and abstention into a VLA and compares against the same VLA without them, upgrades to adaptive conformal inference with measured coverage under drift, and provides a distilled-encoder latency table.

**Strengths.** The VLA integration is the result that elevates the paper: showing a calibrated perceptual bottleneck with abstention improves an existing policy is more influential than winning a bake-off, and it gives the community a reusable component. Adaptive conformal inference is the correct tool and the measured long-run coverage under drift closes my calibration concern. The distilled encoder makes the efficiency claim credible with numbers.

**Weaknesses.** One residual: even distilled, the encoder plus outcome scoring carries overhead relative to a single end-to-end forward pass, so the efficiency argument is strongest against modular reconstruction and weaker against raw end-to-end regressors; the paper should frame the efficiency claim precisely against each baseline class rather than in general.

**Questions.** None outstanding.

**Required experiments.** None; a framing correction on the efficiency claim suffices.

**Novelty assessment.** High and, with the IB ablation delta reported, settled against the closest prior work.

**Technical correctness.** Sound.

**Confidence.** 5.

**Recommendation.** Accept.

## Associate Editor final decision (Iteration 3)

Two Accepts and one Minor Revision, all at confidence 5. Every major criticism raised across the three rounds has been addressed: the outcome operator is grounded and its real fidelity reported with bias correction; the identifiability theorem is correct, symmetry-aware, and now empirically demonstrated by subspace alignment; novelty over task-driven information bottlenecks is quantified; the framework improves a VLA rather than merely competing; and calibration holds under drift. The remaining items are genuine but minor: a rigid-object scope statement in the abstract (A), stratified reporting of the hardest material class (B), and a precise per-baseline framing of the efficiency claim (C). None requires new theory or a new data campaign.

Decision: Accept subject to Minor Revision. The scientific hypothesis is preserved unchanged from submission through V4, and it is now supported by theory that is demonstrated and evidence that is fairly obtained.

## Final framework: MSP V5

V5 is V4 plus the three minor edits, and it is the version I would defend as publication-ready.

Content changes. The abstract states the rigid-object scope of the identifiability result. Results are reported stratified by material difficulty, with the transparent and specular class carrying its own confidence interval and an explicit caveat. The efficiency claim is stated per baseline class: a clear latency and dimension win over modular reconstruction pipelines, and a modest overhead over raw end-to-end regressors that is bought back by calibration, abstention, and active perception those regressors cannot provide.

What V5 is. Two learned modules, a symmetry-equivariant probabilistic sufficiency encoder and a physics-grounded outcome head, trained by a policy-free action-conditioned information bottleneck whose distortion lives in outcome space. Grasp selection, distribution-free abstention under drift, active view-or-touch, and test-time adaptation are inference procedures over those two modules. The outcome operator is grounded in analytic quality, a validated simulator, and a debiased real residual, with the sim-to-real sufficiency gap reported as a headline. Pose and shape are an optional readout, identifiable only up to the manipulation-indistinguishability class, whose local tangent space is the outcome Jacobian null space and whose global discrete ambiguity is the object symmetry orbit, a claim now demonstrated by subspace alignment. The belief can be handed to a VLA to give it a calibrated bottleneck with abstention.

Honest residual limitations carried into the published version. Non-rigid and articulated objects are out of scope because deformation changes the outcome Jacobian. Physics that remains hidden after touch (a sealed container's contents) cannot be resolved and is correctly returned as irreducible uncertainty rather than a confident wrong answer. The hardest material stratum is thinly sampled and its generalization claim is weaker. Compute exceeds a raw end-to-end regressor, justified only by the capabilities that regressor lacks.

---

# Arc of the review process

| Iteration | Reviewer A | Reviewer B | Reviewer C | AE decision | Framework |
|-----------|-----------|-----------|-----------|-------------|-----------|
| 1 | Major | Major | Major | Major Revision | submitted -> V3 |
| 2 | Minor | Major | Minor | Major Revision (positive) | V3 -> V4 |
| 3 | Accept | Minor | Accept | Accept, Minor Revision | V4 -> V5 |

What actually moved the paper. Reviewer A was won by making the identifiability theorem first correct (V3, symmetry and global orbits) then demonstrated (V4, subspace alignment). Reviewer B, the hardest, was won by grounding the outcome operator in real, bias-corrected data (V3 then V4) and by counterfactual attribution, and holds one honest minor caveat on the material long tail. Reviewer C was won by quantifying the delta over task-driven information bottlenecks and by integrating MSP into a VLA rather than only competing with one.

Calibrated final assessment [speculation]. Conditional on the V4/V5 experiments returning as designed, this is a credible T-RO accept after minor revision, with the identifiability demonstration and the VLA integration as the two results most likely to drive citations. The single largest real-world risk that no amount of redesign fully removes is the sim-to-real fidelity of the outcome operator on adversarial materials; the paper's honesty is to report that stratum separately rather than average it away. The scientific hypothesis stated at submission is unchanged in V5, which is the mark of a reformulation that survived review rather than a claim that was diluted to pass it.
