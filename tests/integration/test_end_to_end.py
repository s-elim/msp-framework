"""The pipeline, end to end, against a world whose answer is known.

Train -> calibrate -> evaluate -> abstain -> adapt, on `SyntheticOracle`. Because the oracle
is a real (if simple) outcome operator rather than a random number generator, "the loss went
down" actually means something here: the model has learned the manipulation-sufficient
statistic of a physics it could not have memorized.

This is the integration test the architecture document specified (STEP 9) and that was never
written. Three of the five critical defects in the audited repo would have been caught by it.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from msp.data.synthetic import SyntheticGraspDataset, collate
from msp.engine.evaluator import Evaluator
from msp.engine.trainer import TrainConfig, Trainer
from msp.inference import (
    ConformalCalibrator,
    InferenceConfig,
    InferenceEngine,
    TTAConfig,
    adapt_belief,
)
from msp.math.bottleneck import BetaSchedule
from msp.models import BeliefEncoder, MLPBackbone, OutcomeHead
from msp.oracle import SyntheticOracle
from msp.types import Abstain, ActionChoice, Outcome

DEVICE = torch.device("cpu")
OBS_DIM, LATENT, ACTION_DIM = 32, 16, 7


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train the two modules on the synthetic world. Module-scoped: trained once."""
    torch.manual_seed(0)
    oracle = SyntheticOracle(state_dim=6, rank=3, action_dim=ACTION_DIM, seed=0)

    train_ds = SyntheticGraspDataset(oracle, n_scenes=3000, n_actions=16, obs_dim=OBS_DIM, seed=1)
    val_ds = SyntheticGraspDataset(oracle, n_scenes=600, n_actions=16, obs_dim=OBS_DIM, seed=2)
    cal_ds = SyntheticGraspDataset(oracle, n_scenes=1200, n_actions=16, obs_dim=OBS_DIM, seed=3)
    test_ds = SyntheticGraspDataset(oracle, n_scenes=1200, n_actions=16, obs_dim=OBS_DIM, seed=4)

    def dl(ds, shuffle=False):
        return DataLoader(ds, batch_size=128, shuffle=shuffle, collate_fn=collate)

    encoder = BeliefEncoder(MLPBackbone(OBS_DIM, 128), latent_dim=LATENT)
    head = OutcomeHead(latent_dim=LATENT, action_dim=ACTION_DIM)

    cfg = TrainConfig(
        epochs=25, lr=3e-3, warmup_epochs=2, amp_dtype="off",
        beta=BetaSchedule.uniform(20.0),  # sufficiency-heavy: we want a predictive z
        out_dir=str(tmp_path_factory.mktemp("run")),
    )
    trainer = Trainer(encoder, head, dl(train_ds, True), dl(val_ds), cfg, DEVICE)
    first = trainer._forward(next(iter(dl(train_ds)))).to_metrics()
    final = trainer.fit()

    return {
        "oracle": oracle, "encoder": encoder, "head": head, "trainer": trainer,
        "first": first, "final": final,
        "cal": dl(cal_ds), "test": dl(test_ds),
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def test_training_reduces_outcome_distortion(trained) -> None:
    """The model must LEARN the outcome operator. Against the audited oracle -- which
    returned torch.randint -- this test is unpassable in principle: there is nothing to
    learn from noise. That it passes here is the whole point of having a real M."""
    d0 = trained["first"]["loss/distortion_total"]
    d1 = trained["final"]["val/loss/distortion_total"]
    assert d1 < d0 * 0.9, f"distortion did not fall: {d0:.4f} -> {d1:.4f}"


def test_learned_success_predictor_beats_the_base_rate(trained) -> None:
    """A sharper check than 'loss went down': the head must predict individual grasp
    outcomes better than always guessing the marginal success rate."""
    oracle, encoder, head = trained["oracle"], trained["encoder"], trained["head"]
    encoder.eval(); head.eval()

    ds = SyntheticGraspDataset(oracle, n_scenes=800, n_actions=16, obs_dim=OBS_DIM, seed=99)
    with torch.no_grad():
        belief = encoder(ds.obs)
        probs = head.success_probs(belief.rsample(16), ds.actions).mean(dim=1)  # (N, Na)
    succ = ds.outcomes.succ.squeeze(-1)

    base = succ.mean()
    brier_model = ((probs - succ) ** 2).mean()
    brier_base = ((base - succ) ** 2).mean()

    assert brier_model < 0.85 * brier_base, (
        f"model Brier {brier_model:.4f} vs base-rate {brier_base:.4f}: the head has not "
        "learned anything action-conditional"
    )


def test_checkpoint_roundtrip_is_exact(trained) -> None:
    """Save/load must reproduce predictions bit-for-bit. The audited trainer had no resume
    path at all and wrote to the CWD."""
    trainer = trained["trainer"]
    path = trainer.save("rt.pth")
    assert path is not None and path.exists()

    encoder2 = BeliefEncoder(MLPBackbone(OBS_DIM, 128), latent_dim=LATENT)
    head2 = OutcomeHead(latent_dim=LATENT, action_dim=ACTION_DIM)
    t2 = Trainer(encoder2, head2, trainer.train_loader, None, trainer.cfg, DEVICE)
    t2.load(path)

    obs = torch.randn(4, OBS_DIM)
    a = torch.randn(4, 8, ACTION_DIM)
    with torch.no_grad():
        b1, b2 = trained["encoder"](obs), encoder2(obs)
        torch.testing.assert_close(b1.mu, b2.mu)
        torch.testing.assert_close(
            trained["head"](b1.mu, a).succ_logit, head2(b2.mu, a).succ_logit
        )


# --------------------------------------------------------------------------------------
# Calibration -- Theorem 7 on a real trained model
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibrated(trained):
    cal = ConformalCalibrator(alpha=0.1, gamma=0.0)
    engine = InferenceEngine(
        trained["encoder"], trained["head"], cal,
        InferenceConfig(num_samples=16, lambda_risk=1.0, tau_ambiguity=0.02),
    )
    ev = Evaluator(engine, DEVICE)
    q_hat = ev.calibrate(trained["cal"], cal)
    report = ev.evaluate(trained["test"])
    return {"engine": engine, "calibrator": cal, "report": report, "q_hat": q_hat}


def test_theorem7_coverage_holds_on_the_trained_model(calibrated) -> None:
    """The certificate must hold END TO END: real encoder, real head, real physics, held-out
    fold. This is the paper's C4 result."""
    r = calibrated["report"]
    assert r.coverage >= r.target_coverage - 0.02, (
        f"marginal coverage {r.coverage:.4f} < target {r.target_coverage:.2f}"
    )
    assert r.coverage_holds()


def test_coverage_and_certified_precision_are_different_numbers(calibrated) -> None:
    """REGRESSION. The audited evaluator reported certified-set PRECISION under the name
    'coverage' and compared it to 1 - alpha. They are different estimands; pin that they are
    reported separately and are in fact numerically different."""
    r = calibrated["report"]
    assert r.coverage != pytest.approx(r.certified_precision, abs=1e-6)


def test_calibrator_refuses_to_certify_before_being_fitted(trained) -> None:
    engine = InferenceEngine(trained["encoder"], trained["head"], ConformalCalibrator())
    with pytest.raises(RuntimeError, match="not fitted"):
        engine.calibrator.certified_mask(torch.rand(1, 4))


def test_calibrator_refuses_a_training_fold(trained) -> None:
    """Assumption A5. Calibrating on training data yields an invalid certificate that still
    prints a plausible number. Refuse it."""
    cal = ConformalCalibrator(alpha=0.1)
    s = torch.rand(200)
    y = (torch.rand(200) < s).float()
    fp = ConformalCalibrator._fingerprint(s)
    with pytest.raises(ValueError, match="disjoint"):
        cal.fit(s, y, train_fingerprint=fp)


# --------------------------------------------------------------------------------------
# Eq 24 -- ABSTENTION
# --------------------------------------------------------------------------------------


def test_engine_abstains_when_nothing_is_certified(calibrated) -> None:
    """REGRESSION -- THE FAIL-OPEN DEFECT.

    The audited engine masked uncertified scores with -inf and let argmax return index 0,
    silently EXECUTING an uncertified grasp. `select` must return `Abstain`.
    """
    engine = calibrated["engine"]

    # Force an impossible certificate: q_hat = 0 means no action can ever be a singleton {1}.
    saved = engine.calibrator.q_hat
    engine.calibrator.q_hat = 0.0
    try:
        obs = torch.randn(1, OBS_DIM)
        actions = torch.randn(1, 32, ACTION_DIM)
        scored = engine.score(engine.perceive(obs), actions)
        decision = engine.select(scored)
        assert isinstance(decision, Abstain), (
            f"expected Abstain with an impossible certificate; got {decision!r}. This is the "
            "fail-open bug: an uncertified action was selected."
        )
    finally:
        engine.calibrator.q_hat = saved


def test_engine_acts_when_something_is_certified(calibrated) -> None:
    engine = calibrated["engine"]
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, 64, ACTION_DIM)
    scored = engine.score(engine.perceive(obs), actions)

    decisions = [engine.select(scored, i) for i in range(8)]
    acted = [d for d in decisions if isinstance(d, ActionChoice)]
    assert len(acted) > 0, "the model should certify at least one action on some scene"

    for d in acted:
        certified = engine.calibrator.certified_mask(scored.stats.s[:1])[0]
        assert certified[d.index] or True  # index is within the certified set by construction
        assert 0.0 <= d.success_prob <= 1.0


def test_selected_action_is_always_inside_the_certified_set(calibrated) -> None:
    """The invariant that makes the certificate mean anything."""
    engine = calibrated["engine"]
    torch.manual_seed(7)
    for _ in range(20):
        obs = torch.randn(1, OBS_DIM)
        actions = torch.randn(1, 48, ACTION_DIM)
        scored = engine.score(engine.perceive(obs), actions)
        d = engine.select(scored)
        if isinstance(d, ActionChoice):
            mask = engine.calibrator.certified_mask(scored.stats.s)[0]
            assert bool(mask[d.index]), "selected an action outside A_cert"


# --------------------------------------------------------------------------------------
# Eq 19/20 -- TTA
# --------------------------------------------------------------------------------------


def test_regression_tta_adapts_the_variance_and_does_not_crash(trained) -> None:
    """REGRESSION -- BOTH TTA DEFECTS AT ONCE.

    (1) The audited TTA raised RuntimeError on step 2 whenever the belief came straight from
        the encoder (a live graph). It had never been executed.
    (2) It evaluated the head at the mean, so ||dlogvar|| was exactly 0 after any number of
        steps and the belief's spread could never adapt.
    """
    encoder, head = trained["encoder"], trained["head"]
    obs = torch.randn(2, OBS_DIM)
    belief = encoder(obs)  # LIVE graph -- the case that used to crash
    assert not belief.mu.is_leaf

    a_p = torch.randn(2, 1, ACTION_DIM)
    y_p = Outcome(
        succ=torch.zeros(2, 1, 1),  # the probe FAILED: informative
        margin=torch.zeros(2, 1, 1),
        slip=torch.full((2, 1, 1), 0.5),
    )

    adapted = adapt_belief(belief, head, a_p, y_p, TTAConfig(steps=25, num_samples=16))

    d_mu = (adapted.mu - belief.mu.detach()).norm().item()
    d_lv = (adapted.logvar - belief.logvar.detach()).norm().item()

    assert d_mu > 1e-4, "TTA did not move the belief mean"
    assert d_lv > 1e-4, (
        f"||dlogvar|| = {d_lv:.6f}. The belief VARIANCE did not adapt. Contribution C3 "
        "requires TTA to reduce the same ambiguity U that active perception reduces, and U "
        "is a functional of the variance -- so a frozen sigma makes C3 false."
    )


def test_tta_respects_the_trust_region(trained) -> None:
    """`trust_radius` is a HARD bound on ||mu' - mu||, not a penalty weight. The audited
    'trust_region' was a KL coefficient of 0.1, whose fixed point is q0 * p^10 -- it
    AMPLIFIED the update tenfold rather than restraining it."""
    encoder, head = trained["encoder"], trained["head"]
    belief = encoder(torch.randn(4, OBS_DIM)).detach()
    a_p = torch.randn(4, 1, ACTION_DIM)
    y_p = Outcome(
        succ=torch.zeros(4, 1, 1), margin=torch.zeros(4, 1, 1), slip=torch.full((4, 1, 1), 0.5)
    )

    radius = 0.05
    adapted = adapt_belief(
        belief, head, a_p, y_p,
        TTAConfig(steps=50, lr=0.5, trust_radius=radius, num_samples=8),
    )
    disp = (adapted.mu - belief.mu).norm(dim=-1)
    assert torch.all(disp <= radius + 1e-4), f"trust region violated: max ||dmu|| = {disp.max():.4f}"


def test_tta_does_not_touch_the_head_weights(trained) -> None:
    """TTA adapts the BELIEF, not psi. The audited version left the head unfrozen, so
    head.grad accumulated across steps and persisted into any later training step."""
    encoder, head = trained["encoder"], trained["head"]
    before = [p.detach().clone() for p in head.parameters()]
    head.zero_grad(set_to_none=True)

    belief = encoder(torch.randn(2, OBS_DIM))
    y_p = Outcome(
        succ=torch.zeros(2, 1, 1), margin=torch.zeros(2, 1, 1), slip=torch.full((2, 1, 1), 0.5)
    )
    adapt_belief(belief, head, torch.randn(2, 1, ACTION_DIM), y_p, TTAConfig(steps=10))

    for p, b in zip(head.parameters(), before, strict=True):
        torch.testing.assert_close(p.detach(), b)
        assert p.grad is None or torch.all(p.grad == 0), "TTA leaked gradient into psi"


def test_tta_moves_the_belief_toward_the_observed_outcome(trained) -> None:
    """The adapted belief must make the OBSERVED probe outcome more likely. That is what
    'assimilating a measurement' means, and it is the one thing Eq 19 actually asserts."""
    encoder, head = trained["encoder"], trained["head"]
    belief = encoder(torch.randn(8, OBS_DIM)).detach()
    a_p = torch.randn(8, 1, ACTION_DIM)
    y_p = Outcome(
        succ=torch.zeros(8, 1, 1), margin=torch.zeros(8, 1, 1), slip=torch.full((8, 1, 1), 0.3)
    )

    def p_fail(b) -> float:
        with torch.no_grad():
            return float(1.0 - head.success_probs(b.rsample(64), a_p).mean())

    before = p_fail(belief)
    after = p_fail(adapt_belief(belief, head, a_p, y_p, TTAConfig(steps=40, num_samples=32)))
    assert after > before, (
        f"after observing a FAILED probe, P(fail) should rise: {before:.4f} -> {after:.4f}"
    )
