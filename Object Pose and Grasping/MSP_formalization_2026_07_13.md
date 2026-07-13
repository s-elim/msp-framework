# Manipulation-Sufficient Perception (MSP): Complete Formalization

Self-contained. Every object is defined before use. Notation is fixed in Section 0 and used consistently. The document ends with training and inference pseudocode with tensor shapes, sufficient to implement without further reference. Proof sketches give the argument structure and the invoked theorems; they are not full proofs.

Convention: `succ` denotes the success coordinate of the outcome. Expectations are over the distributions written in the subscript. All logs are natural. `KL` is relative entropy. `I(.;.)` is Shannon mutual information, `H` is differential or discrete entropy as appropriate.

---

## 0. Spaces and variables

State space. `X`, a measurable space of physical world states, assumed a smooth d_X-dimensional manifold on the rigid-object domain (Assumption A6). A state is
```
x = (S, T, phi) in X,
  S   object shape, an element of a shape space (e.g. an SDF in a function space or a fixed-topology mesh in R^{3V})
  T   object pose in SE(3)
  phi = (mu, m, c, ...) physical parameters: friction mu>0, mass m>0, center of mass c in R^3, optional compliance
```
Prior (scene distribution) `p(x)`, a probability measure on `X`.

Observation space. `O = R^{H x W x 4} x C`, an RGB-D image plus an optional finite language token sequence `c in C`. Observation kernel `p(o | x)` (sensor model, includes camera intrinsics and noise).

Action space. `A = SE(3) x R_{>=0} x G`, a grasp pose, a gripper opening width, and a gripper (embodiment) descriptor `g in G`. Reference action measure `rho` on `A` with full support on admissible actions (Assumption A3). Sensing-action space `B` (viewpoint changes, light probes), disjoint from `A` in effect.

Outcome space. `Y = {0,1} x R x R`, `y = (succ, margin, slip)`: binary lift success, a stability margin in R, and a post-lift slip magnitude in R_{>=0}. Physics kernel (the ground-truth outcome operator)
```
M : X x A -> P(Y),   M(dy | x, a),
```
a Markov kernel giving the true action-conditioned outcome distribution. In simulation `M` is queryable (Assumption A2); in reality it is accessible only through executed grasps.

Latent statistic. `Z = R^d`, the manipulation-sufficient statistic. Encoder (belief) is a stochastic kernel `q_theta(z | o)`. Reference prior `r(z)` on `Z`.

Learned outcome head. `p_psi(y | z, a)`, a Markov kernel approximating the posterior-predictive outcome given the statistic.

Parameters. `theta` (encoder), `psi` (outcome head), optional `omega` (amortized acquisition, Section 5), `tau, q_hat` (calibration, Section 7).

---

## 1. Generative model and graphical structure

Sampling process for one datum:
```
x ~ p(x)
o ~ p(o | x)
a ~ rho              (actions drawn independently of x)
y ~ M(. | x, a)
z ~ q_theta(. | o)   (belief, used at train and test)
```
Conditional independencies, given the action `A = a`:
```
A ⟂ (X, O, Z);   Z ⟂ (X, Y) | O;   Y ⟂ (O, Z) | X.
```
Hence, conditioned on `A`, the variables form the Markov chain
```
Z -- O -- X -- Y.                              (1)
```
By the data-processing inequality applied to (1),
```
I(Z; Y | A)  <=  I(O; Y | A)  <=  I(X; Y | A).  (2)
```
Interpretation: no statistic of the observation can carry more action-relevant outcome information than the observation itself, which in turn is bounded by the physical state. MSP seeks the smallest `Z` that makes the left inequality in (2) tight.

---

## 2. The estimand: sufficiency, minimality, indistinguishability

Posterior-predictive outcome field. For an observation `o`, define
```
eta(o) := ( a  |->  p(y | o, a) ),   where   p(y | o, a) = ∫_X M(y | x, a) p(x | o) dx.   (3)
```
`eta(o)` is an element of `P(Y)^A`, the map from every action to its Bayes-optimal outcome distribution given `o`. This is the operational target: `o` does not determine `x`, so the most that any perception system can extract for manipulation is `eta(o)`.

Definition 2.1 (manipulation sufficiency). A statistic `z = t(o)` (deterministic) or a kernel `q(z|o)` is manipulation-sufficient iff
```
p(y | o, a) = p(y | t(o), a)   for rho-a.e. a,   equivalently   I(Z; Y | A) = I(O; Y | A).   (4)
```

Definition 2.2 (minimal sufficiency). A sufficient `t*` is minimal iff for every sufficient `t`, `t*` is a measurable function of `t`.

Definition 2.3 (manipulation indistinguishability on states). For `x, x' in X`,
```
x ~ x'   iff   M(. | x, a) = M(. | x', a)  for rho-a.e. a.                                   (5)
```
`~` is an equivalence relation; `X / ~` is the quotient, `pi : X -> X/~` the projection. The outcome map is
```
Phi : X -> P(Y)^A,   Phi(x) = ( a |-> M(. | x, a) ),   so   x ~ x'  iff  Phi(x) = Phi(x').    (6)
```

These three definitions pin the estimand: perception should estimate `eta(o)` (equivalently a coordinate on `X/~` weighted by the belief `p(x|o)`), not a metric pose.

---

## 3. Training objective (variational information bottleneck)

Target functional. Minimize over the encoder the rate subject to sufficiency, in Lagrangian form:
```
min_{q_theta, p_psi}   I(Z; O)  -  beta * I(Z; Y | A),   beta > 0.                            (7)
```

Tractable bounds. For any reference `r(z)`,
```
I(Z; O)  <=  E_{o}[ KL( q_theta(z|o) || r(z) ) ]  =: R(theta),                                 (8)
```
with gap `KL(q_theta(z) || r(z)) >= 0` where `q_theta(z) = E_o[q_theta(z|o)]`. For the relevance term, since `I(Z;Y|A) = H(Y|A) - H(Y|Z,A)` and `H(Y|Z,A) <= -E[ log p_psi(y|z,a) ]` for any `p_psi`,
```
I(Z; Y | A)  >=  H(Y | A)  +  E_{x,o,a,y,z}[ log p_psi(y | z, a) ]  =: H(Y|A) - D(theta,psi).  (9)
```
`H(Y|A)` is constant in `(theta, psi)`. Substituting (8) and (9) into (7) gives the training loss (drop constants, rescale by `beta`):
```
L(theta, psi)
  =  E_{x~p, o~p(.|x), a~rho, y~M(.|x,a), z~q_theta(.|o)} [ - log p_psi(y | z, a) ]        (10)
     +  (1/beta) * E_{o} [ KL( q_theta(z|o) || r(z) ) ].
```
First term is the outcome distortion (sufficiency); second is the rate (minimality). `beta` sets the sufficiency budget and may be made per-outcome-dimension, `beta = (beta_succ, beta_margin, beta_slip)`, so language `c` can reweight which outcomes must be resolved (this is the formal handle for task-conditioned sufficiency).

Distortion is measured in outcome space, not geometry space. There is no reconstruction or pose loss anywhere in (10); the only supervision is `y ~ M(.|x,a)`.

Estimators (Gaussian instantiation). Encoder `q_theta(z|o) = N(mu_theta(o), diag sigma_theta(o)^2)`, reparameterize `z = mu_theta(o) + sigma_theta(o) ⊙ epsilon`, `epsilon ~ N(0, I_d)`. With `r(z) = N(0, I_d)`,
```
KL( q_theta(z|o) || r ) = 0.5 * sum_{j=1..d} ( sigma_j^2 + mu_j^2 - 1 - log sigma_j^2 ).       (11)
```
Outcome head factorizes `p_psi(y|z,a) = Bern(succ; sigmoid(f_psi(z,a))) * N(margin; m_psi(z,a), s_psi^2) * N(slip; ...)`, so `-log p_psi` is a binary cross-entropy plus Gaussian NLL terms.

Energy form. Define per-sample energy
```
E(z; o, a, y) = - log p_psi(y|z,a) + (1/beta) * log[ q_theta(z|o) / r(z) ].                     (12)
```
The belief concentrates on low-energy codes that predict outcomes well and stay short in description length; (10) is `E_{o,a,y} E_{z~q}[E]`.

---

## 4. Inference decision rule

Given `o`, draw `K` posterior samples `z_1..z_K ~ q_theta(.|o)`. Define, for action `a`,
```
sigma_psi(z, a) = P_psi(succ = 1 | z, a) = sigmoid(f_psi(z, a)),
s(o, a) = E_{z~q}[ sigma_psi(z, a) ]        ~=  (1/K) sum_k sigma_psi(z_k, a),                  (13)
v(o, a) = Var_{z~q}[ sigma_psi(z, a) ]      ~=  (1/K) sum_k (sigma_psi(z_k, a) - s(o,a))^2.      (14)
```
`s` is the predicted success probability marginalizing epistemic uncertainty; `v` is the epistemic variance induced by the belief spread. Risk-averse selection:
```
a*(o) = argmax_{a in A_cand}  [ s(o, a) - lambda * v(o, a) ],   lambda >= 0.                     (15)
```
`A_cand` is a finite candidate set sampled from `rho` (coarse-to-fine refinement in Section 10). `lambda` trades expected success against epistemic risk.

---

## 5. Active perception (value of information)

Ambiguity functional:
```
U(o) = E_{a~rho}[ v(o, a) ] ~= (1/|A_cand|) sum_{a in A_cand} v(o, a).                          (16)
```
Exact one-step value of information for a sensing action `b in B` with next-observation predictive `p~(o_b | o, b)`:
```
IG(b) = U(o) - E_{o_b ~ p~(.|o,b)}[ U(o ∪ o_b) ],   b* = argmax_{b in B} IG(b),                 (17)
```
where `o ∪ o_b` denotes the belief re-encoded from the fused observation (Section 6). `p~(o_b|o,b)` is the posterior predictive of the future observation; computing it exactly requires a generative observation model.

Amortized estimator (implementable without a world model). Train an acquisition network `alpha_omega(o, b) ~= IG(b)` by simulated look-ahead: in simulation the true `x` is known, so `o_b` can be rendered and the true `IG(b)` computed. Regression objective
```
L_acq(omega) = E_{x, o, b} [ ( alpha_omega(o, b) - IG_true(x, o, b) )^2 ],                       (18)
```
with `IG_true` computed from (16)-(17) using rendered `o_b`. At test time select `b* = argmax_b alpha_omega(o, b)`. Trigger sensing only when `U(o) > tau_U`.

---

## 6. Test-time adaptation

Static scene (probe does not move the object). Executing action `a_p` returns outcome `y_p`. The exact Bayesian belief update is
```
q'(z) ∝ q_theta(z | o) * p_psi(y_p | z, a_p).                                                   (19)
```
Implement by a short optimization of a variational `q'(z) = N(mu', diag sigma'^2)` initialized at `q_theta(.|o)`:
```
min_{mu', sigma'}  E_{z~q'}[ - log p_psi(y_p | z, a_p) ]  +  KL( q'(z) || q_theta(z|o) ),        (20)
```
k gradient steps with a trust region (small step count, bounded `||mu' - mu||`).

Scene-perturbing probe. If `a_p` changes the state through a known transition `Tk(x' | x, a_p)`, re-encode from a fresh observation `o'` after the probe and fuse with the outcome constraint:
```
q'(z) ∝ q_theta(z | o') * p_psi(y_p | z, a_p),                                                   (21)
```
same optimization as (20) with `o` replaced by `o'`. Active perception (Section 5) and TTA optimize the same quantity, the reduction of outcome-class ambiguity `U`, one by choosing the measurement `b` and one by assimilating the returned `y_p`.

---

## 7. Calibrated abstention (conformal)

Goal: a distribution-free certificate on executed grasps. Use split conformal on the success predictor.

Calibration. Hold out `n` labeled pairs `{(o_i, a_i, succ_i)}`. Nonconformity score for a candidate `(o, a)`:
```
E(o, a, succ) = 1 - hat_p_succ   if succ = 1,     hat_p_succ   if succ = 0,                      (22)
```
where `hat_p_succ = s(o, a)` from (13). Let `q_hat` be the `ceil((n+1)(1 - alpha)) / n` empirical quantile of `{E_i}`. The prediction set for the success label at `(o, a)` is
```
C(o, a) = { l in {0,1} : score_l(o, a) <= q_hat },  score_1 = 1 - s,  score_0 = s.               (23)
```
Certified-successful action set and abstention:
```
A_cert(o) = { a in A_cand : C(o, a) = {1} },   act if nonempty, else ABSTAIN.                    (24)
```

Theorem 7 (marginal coverage). Under exchangeability of calibration and test points (Assumption A5), for a fresh `(o, a)`,
```
P( succ in C(o, a) ) >= 1 - alpha.
```
Proof sketch: split-conformal coverage; the score is computed with a model fit on data independent of the calibration fold, so the calibration scores and the test score are exchangeable, and the rank of the test score is uniform, giving the quantile bound. Under distribution drift, replace the fixed `q_hat` by adaptive conformal inference (ACI): update `alpha_t <- alpha_t + gamma (alpha - err_t)` online, which guarantees long-run coverage `|(1/T) sum_t err_t - alpha| -> 0` without exchangeability. QED sketch.

---

## 8. Assumptions

```
A1 (measurability/regularity). All kernels are measurable; densities exist w.r.t. reference measures where written.
A2 (simulation oracle + realizability). M is queryable at train time, and the family {p_psi} can represent the true predictive p(y|z,a) at the optimum.
A3 (action support). rho has full support on the admissible action set for every scene.
A4 (constant rank). The outcome Jacobian J(x) (Def. 9.1) has locally constant rank on a neighborhood of x.
A5 (exchangeability). Calibration and deployment points are exchangeable; relaxed to arbitrary drift under ACI.
A6 (rigid objects). On the identifiability domain, X is a fixed-dimensional smooth manifold and Phi is C^1 in x for each a.
A7 (predictive/transition access). For active perception, either a generative observation model p~(o_b|o,b) or an amortizable IG estimator exists; for perturbing probes, the transition Tk is known.
A8 (informative outcomes). The success functional is nonconstant in a on a set of positive rho-measure for almost every scene (otherwise the task is trivial).
```

---

## 9. Theorems and proof sketches

Definition 9.1 (outcome Jacobian). Fix a countable rho-dense action set `{a_j}`. Represent each `M(.|x,a_j)` by a finite sufficient parameter vector `theta_j(x) in R^{p}` (e.g. success logit and margin/slip moments). The outcome map in coordinates is `Phi(x) = (theta_j(x))_j`. Its differential at `x` is
```
J(x) : T_x X -> ⊕_j R^p,   J(x) delta = ( D_x theta_j(x)[delta] )_j.                             (25)
```

Theorem 1 (sufficiency preserves grasp value). If `z` is manipulation-sufficient (Def. 2.1), then for every `o`,
```
max_{a} E[succ | z, a]  =  max_{a} E[succ | o, a],
```
i.e., acting on `z` loses no attainable success relative to acting on the full observation.
Proof sketch: sufficiency gives `p(y|z,a) = p(y|o,a)` for rho-a.e. `a`, hence the success functionals `s(z,a) = s(o,a)` agree rho-a.e.; taking suprema over a full-support action set preserves equality. QED.

Theorem 2 (eta is minimal sufficient). The posterior-predictive field `eta(o)` in (3) is a minimal sufficient statistic of `o` for the action-conditioned outcome family.
Proof sketch: `eta(o)` determines `p(y|o,a)` by construction, so it is sufficient. For any sufficient `t`, Def. 2.1 gives `p(y|o,a) = p(y|t(o),a)`, so `a |-> p(y|o,a)` is a measurable function of `t(o)`, i.e. `eta = F ∘ t`. Thus `eta` is a measurable function of every sufficient statistic, which is minimality (Lehmann-Scheffe partition argument). QED.

Theorem 3 (IB optimum recovers minimal sufficiency). Let `(theta,psi)` minimize (10). As `beta -> infinity`, any minimizer satisfies `I(Z;Y|A) -> I(O;Y|A)` (sufficiency), and among sufficient encoders the rate term `I(Z;O)` selects one with minimal `I(Z;O)`; its induced partition equals that of `eta`.
Proof sketch: the relevance bound (9) is tight when `p_psi = p(y|z,a)` (cross-entropy meets entropy), and the rate bound (8) is tight when `r = q_theta(z)`. As `beta -> infinity` the objective forces the relevance term to its DPI ceiling `I(O;Y|A)` from (2), which is exactly sufficiency; the residual `(1/beta)` rate term is then minimized among sufficient solutions, whose coarsest representative is the `eta`-partition by Theorem 2. QED.

Theorem 4 (local identifiability). Under A4, A6, the equivalence class `[x]` is, in a neighborhood of `x`, a smooth submanifold of `X` with
```
T_x [x] = ker J(x),   dim [x] = d_X - rank J(x).
```
Consequently any readout functional `g(x)` (pose, shape) is determined by the outcome family only up to first order along `ker J(x)`; the identifiable component is the projection of `dg` onto `row J(x) = (ker J(x))^perp`.
Proof sketch: `[x]` locally equals the level set `Phi^{-1}(Phi(x))`. Under constant rank (A4), the constant-rank theorem makes this level set a submanifold whose tangent space is `ker D_x Phi(x) = ker J(x)`. Directional derivatives of `g` along `ker J(x)` are unconstrained by `Phi`, giving the identifiable/unidentifiable split. QED.

Theorem 5 (global discrete ambiguity via symmetry). Let `Gx = { g acting on X : Phi(g . x) = Phi(x) }` be the outcome-invariance group at `x` (contains object symmetry orbits, e.g. rotations fixing the shape and leaving contact response invariant). Then
```
[x]  ⊇  Gx . x,   and globally   [x] = ∪_{g in Gx} ( local leaf of Theorem 4 through g . x ).
```
Thus the equivalence class is a union of symmetry-related smooth leaves, and the belief `q_theta(z|o)` on a symmetric object should be supported on the corresponding orbit rather than collapsed to a mean.
Proof sketch: invariance `Phi(g.x)=Phi(x)` gives `g.x ~ x`, so `Gx.x ⊆ [x]`. Any `x' ~ x` shares `Phi`, and by Theorem 4 lies on a local leaf; connecting `x'` to some `g.x` by an outcome-preserving path shows `[x]` is exactly the union of leaves over the orbit. QED sketch.

Theorem 6 (compression has no value cost, quantitative). Let `z` be an epsilon-sufficient encoder, `I(O;Y|A) - I(Z;Y|A) <= epsilon`. Then the expected optimal-success gap is bounded:
```
E_o[ max_a s(o,a) - max_a s(z,a) ]  <=  C * sqrt(epsilon),
```
for a constant `C` depending on the outcome parameterization (Pinsker-type).
Proof sketch: the information deficit bounds `E_o KL( p(y|o,a) || p(y|z,a) )` averaged over `a` via the chain-rule identity `I(O;Y|A) - I(Z;Y|A) = E[ KL( p(y|o,a) || p(y|z,a) ) ]` under the Markov chain (1); Pinsker converts KL to total variation, and TV bounds the difference of the bounded success functionals; the sup over `a` is controlled by a uniform (full-support) argument. QED sketch.

Theorem 7 (coverage) is stated in Section 7.

Corollary (evaluation implication). By Theorem 4, pose error metrics (ADD-S, Chamfer) that penalize deviation along `ker J(x)` measure unidentifiable, task-irrelevant quantities; the manipulation-relevant error is the `row J(x)` component. This formalizes why accuracy-driven metrics can misrank methods.

---

## 10. Algorithms

Shapes are annotated as `(dim, ...)`. Batch size `Bsz`, latent dim `d`, samples `K`, candidate actions `Na`.

Algorithm 1 (training).
```
input: scene sampler p(x); sensor p(o|x); action measure rho; physics oracle M; beta, K
init: theta (encoder), psi (outcome head), optional omega (acquisition); r(z) = N(0, I_d)

repeat:
  sample minibatch { x^(b) } ~ p(x)                         # b = 1..Bsz
  render o^(b) ~ p(o | x^(b))                               # o: (Bsz, H, W, 4) [+ language]
  sample actions a^(b,i) ~ rho, importance-focused near the
      outcome decision boundary of M(.|x^(b),.)             # a: (Bsz, Na, 7) with gripper id
  query outcomes y^(b,i) ~ M(. | x^(b), a^(b,i))            # y: (Bsz, Na, 3)  [succ, margin, slip]

  # encoder forward (reparameterized)
  (mu, logsig2) = Encoder_theta(o)                          # mu, logsig2: (Bsz, d)
  eps ~ N(0, I_d)                                           # (Bsz, d)  (single sample for training)
  z = mu + exp(0.5*logsig2) * eps                           # (Bsz, d)

  # outcome head forward for all candidate actions
  yhat = Head_psi(z, a)                                     # per (b,i): (succ_logit, margin, logvar_margin, slip params)
  L_dist = mean over (b,i) of [ BCE(succ_logit, succ)
                                + GaussNLL(margin | yhat)
                                + GaussNLL(slip   | yhat) ] # scalar   (Eq. 10, term 1)
  L_rate = mean over b of 0.5*sum_j (exp(logsig2)+mu^2 -1 -logsig2)   # scalar  (Eq. 11)
  L = L_dist + (1/beta) * L_rate

  step (theta, psi) with grad L

  # optional: acquisition net (Eq. 18), using rendered look-ahead o_b in sim
  if training_active_perception:
     compute IG_true(x, o, b) via (16)-(17) with rendered o_b
     L_acq = mean ( alpha_omega(o, b) - IG_true )^2
     step omega with grad L_acq
until converged

# post-training grounding + calibration
fit low-parameter real residual of Head_psi on autonomously collected, importance-weighted real (o,a,succ)
compute conformal q_hat on held-out calibration fold          # Eq. 22-23
return theta, psi, omega, q_hat
```

Algorithm 2 (inference: perceive, decide, sense, adapt, or abstain).
```
input: observation o; params theta, psi, omega, q_hat; K, Na, lambda, tau_U, alpha
init: belief b0 = q_theta(. | o) = (mu, logsig2)

loop:
  z_1..z_K ~ N(mu, diag exp(logsig2))                        # z: (K, d)
  sample candidate actions a_1..a_Na ~ rho (coarse-to-fine)  # a: (Na, 7)
  for each a:
     p_k = sigmoid(f_psi(z_k, a))  for k=1..K                # (K,)
     s[a] = mean_k p_k                                       # Eq. 13
     v[a] = var_k  p_k                                       # Eq. 14
  U = mean_a v[a]                                            # Eq. 16

  if U > tau_U and sensing budget remains:                   # active perception
     b* = argmax_b alpha_omega(o, b)                         # Eq. 18
     execute b*; obtain fused/fresh observation o'
     update belief: o <- o'; (mu, logsig2) = Encoder_theta(o')
     continue loop

  # calibrated feasibility (Eq. 23-24)
  A_cert = { a : s[a] >= 1 - q_hat }                          # certified-success set
  if A_cert is empty: return ABSTAIN

  a* = argmax_{a in A_cert} ( s[a] - lambda * v[a] )          # Eq. 15
  execute a*; observe outcome y_p
  if outcome uncertain or task continues:                     # test-time adaptation
     (mu, logsig2) = argmin (Eq. 20 or 21) starting from current belief
  return a*, y_p
```

Optional readout (honest pose/shape). Train a decoder `g_xi(z)` to the maximum-entropy representative of `[x]`:
```
g_xi(z) = argmax_{x in support} H(x)  s.t.  Phi(x) matches p_psi(. | z, .);   report only row J(x) component as identified.
```

---

## 11. Instantiation checklist (to implement directly)

```
Encoder_theta:   RGB-D (+ language) backbone -> heads (mu, logsig2) in R^d.  Symmetry-equivariant backbone
                 so symmetric objects yield orbit-supported beliefs (Theorem 5).
Head_psi:        MLP on (z, a-embedding, gripper-embedding) -> (succ_logit, margin mean/logvar, slip params).
M (train):       analytic Ferrari-Canny epsilon-quality + force-closure margin (differentiable prior)
                 composed with a version-pinned rigid-body contact simulator for slip/perturbation.
rho:             mixture of uniform-on-admissible and boundary-focused proposals; importance weights logged.
r(z):            N(0, I_d).  beta: swept; per-outcome-dimension for language budgeting.
K:               8-32 posterior samples (or a small deep ensemble for calibrated epistemic variance).
Calibration:     split conformal (Eq. 22-24); ACI online update in deployment.
Active/TTA:      Eq. 18 (acquisition), Eq. 20/21 (belief update); trust-region, 1-5 steps.
Diagnostics:     estimate J(x) by autodiff of Phi in sim; report principal angles between ker J(x)
                 and the empirically grasp-invariant perturbation subspace (validates Theorem 4).
```

Everything needed to code MSP is fixed above: the loss (10)-(11), the decision rule (13)-(15), the VoI (16)-(18), the TTA update (19)-(21), the conformal sets (22)-(24), the assumptions A1-A8, and the two algorithm blocks with tensor shapes.
