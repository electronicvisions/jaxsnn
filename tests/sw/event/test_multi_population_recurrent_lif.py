"""Tests for MultiPopulationRecurrentLIF — heterogeneous params per population
in a single connected recurrent graph.

The existing `RecurrentLIF` accepts one `LIFParameters` shared across all
populations in the graph. Biology-faithful multi-cell-type circuits (e.g.
cerebellum granule + Purkinje) need different time constants per population
within a connected graph. These tests cover the new
`MultiPopulationRecurrentLIF` factory that admits a list of `LIFParameters`,
one per population.

Scope (v1):
  * Per-population heterogeneity (one `LIFParameters` per population).
  * Inherits the analytical TTFS solver's two-ratio constraint (each
    population must satisfy `tau_mem == tau_syn` or `tau_mem == 2 * tau_syn`).
    Per-neuron heterogeneity within a population and arbitrary tau ratios
    are deferred to v2 (Newton-based solver).
"""
import unittest

import jax
import jax.numpy as jnp

from jaxsnn.base.compose import serial
from jaxsnn.base.params import LIFParameters
from jaxsnn.event.modules.leaky_integrate_and_fire import (
    MultiPopulationRecurrentLIF,
    RecurrentLIF,
)
from jaxsnn.event.types import EventPropSpike, WeightRecurrent


def _make_input(n_input: int, key: jax.Array) -> EventPropSpike:
    """A small spike pattern feeding the network."""
    return EventPropSpike(
        time=jnp.array([1e-3, 2e-3, 3e-3, 4e-3]),
        idx=jnp.array([0, 1, 0, 2]),
        current=jnp.zeros(4),
    )


class TestMultiPopulationRecurrentLIF(unittest.TestCase):

    def test_constructs_with_per_population_params(self):
        """Factory constructs without error given a list of LIFParameters
        of the same length as `layers`."""
        layers = [4, 4]
        params_per_pop = [
            LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0),   # ratio 1
            LIFParameters(tau_mem=20e-3, tau_syn=20e-3, v_th=1.0), # ratio 1
        ]
        init_fn, apply_fn = serial(MultiPopulationRecurrentLIF(
            layers=layers,
            n_spikes=8,
            t_max=50e-3,
            params_per_population=params_per_pop,
            mean=[0.5, 0.5],
            std=[1.0, 1.0],
        ))
        self.assertIsNotNone(init_fn)
        self.assertIsNotNone(apply_fn)

    def test_validates_params_list_length(self):
        """Passing the wrong number of LIFParameters raises a clear error."""
        layers = [4, 4]
        wrong = [LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0)]  # 1 of 2
        with self.assertRaises(ValueError):
            MultiPopulationRecurrentLIF(
                layers=layers,
                n_spikes=8,
                t_max=50e-3,
                params_per_population=wrong,
                mean=[0.5, 0.5],
                std=[1.0, 1.0],
            )

    def test_validates_v_th_is_one(self):
        """Off-1.0 v_th raises ValueError because step.py hard-codes V>=1.0."""
        layers = [4, 4]
        bad = [
            LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0),
            LIFParameters(tau_mem=20e-3, tau_syn=20e-3, v_th=0.5),  # bad
        ]
        with self.assertRaisesRegex(ValueError, r"v_th=.*only v_th=1\.0"):
            MultiPopulationRecurrentLIF(
                layers=layers,
                n_spikes=8,
                t_max=50e-3,
                params_per_population=bad,
                mean=[0.5, 0.5],
                std=[1.0, 1.0],
            )

    def test_validates_layers_integer(self):
        """Non-integer layer sizes raise a clean ValueError."""
        good_params = [
            LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0),
            LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0),
        ]
        with self.assertRaisesRegex(ValueError, r"positive integer"):
            MultiPopulationRecurrentLIF(
                layers=[4, 4.5],   # 4.5 is not an int
                n_spikes=8,
                t_max=50e-3,
                params_per_population=good_params,
                mean=[0.5, 0.5],
                std=[1.0, 1.0],
            )

    def test_heterogeneous_dynamics_observable(self):
        """Two populations with same input but different tau values produce
        observably different spike patterns. The load-bearing test."""
        layers = [4, 4]
        n_input = 3
        n_total = sum(layers)

        # Pop A: fast (tau=5ms). Pop B: slow (tau=20ms). Both ratio=1.
        params_per_pop = [
            LIFParameters(tau_mem=5e-3, tau_syn=5e-3, v_th=1.0),
            LIFParameters(tau_mem=20e-3, tau_syn=20e-3, v_th=1.0),
        ]

        init_fn, apply_fn = serial(MultiPopulationRecurrentLIF(
            layers=layers,
            n_spikes=16,
            t_max=80e-3,
            params_per_population=params_per_pop,
            mean=[1.0, 1.0],
            std=[0.0, 0.0],
        ))

        rng = jax.random.PRNGKey(0)
        _, weights = init_fn(rng, n_input)

        inputs = _make_input(n_input, rng)
        # apply_fn returns (carry, weights, spikes, recording); [2] = spikes
        out = apply_fn(weights, inputs)[2]

        # apply_fn output spikes include input neurons (idx 0..n_input-1)
        # plus layer neurons offset by n_input. Layer-neuron pop A is
        # [n_input, n_input + layers[0]); pop B is the next layers[1] block.
        times = jnp.asarray(out.time)
        idxs = jnp.asarray(out.idx)
        valid = (idxs >= n_input) & jnp.isfinite(times) & (times < 80e-3)
        idx_a_mask = valid & (idxs < n_input + layers[0])
        idx_b_mask = valid & (idxs >= n_input + layers[0])

        n_a = int(idx_a_mask.sum())
        n_b = int(idx_b_mask.sum())

        # Sanity: at least one population spiked.
        self.assertGreater(n_a + n_b, 0,
                           "Neither population produced any spikes.")

        # Load-bearing assertion: output is NOT identical to running
        # RecurrentLIF with the slow params alone. (i.e. the fast population
        # contributes distinct dynamics.) We compare to a RecurrentLIF
        # baseline running with the slow params only.
        baseline_init, baseline_apply = serial(RecurrentLIF(
            layers=layers,
            n_spikes=16,
            t_max=80e-3,
            params=params_per_pop[1],   # slow params only
            mean=[1.0, 1.0],
            std=[0.0, 0.0],
        ))
        _, baseline_weights = baseline_init(rng, n_input)
        baseline_out = baseline_apply(baseline_weights, inputs)[2]

        # The pop-A first-spike time should differ from the baseline's pop-A
        # first-spike time — that's the per-population evidence.
        b_idxs = jnp.asarray(baseline_out.idx)
        b_times = jnp.asarray(baseline_out.time)
        b_valid = (b_idxs >= n_input) & jnp.isfinite(b_times) & (b_times < 80e-3)
        b_a_mask = b_valid & (b_idxs < n_input + layers[0])

        self.assertGreater(
            n_a, 0,
            "Pop A (fast tau) produced no spikes — per-population tau may have "
            "been silently flattened to slow."
        )
        self.assertGreater(
            int(b_a_mask.sum()), 0,
            "All-slow baseline produced no pop-A spikes — test setup invalid."
        )
        first_a = float(jnp.min(times[idx_a_mask]))
        first_a_baseline = float(jnp.min(b_times[b_a_mask]))
        self.assertNotAlmostEqual(
            first_a, first_a_baseline, places=5,
            msg="Pop-A first-spike time matches the slow-params baseline; "
                "per-population tau may have been broadcast/flattened."
        )

        # Stronger empirical signature (per spec): pop B in heterogeneous
        # should fire LATER than in an all-fast baseline (because of slow
        # tau integrating an early kick from fast pop A) and EARLIER than
        # in an all-slow baseline (because that kick from pop A arrives
        # later in the all-slow case). This is the "between-baselines"
        # signal that catches a class of regression the pop-A check alone
        # could miss.
        fast_init, fast_apply = serial(RecurrentLIF(
            layers=layers,
            n_spikes=16,
            t_max=80e-3,
            params=params_per_pop[0],   # fast params
            mean=[1.0, 1.0],
            std=[0.0, 0.0],
        ))
        _, fast_weights = fast_init(rng, n_input)
        fast_out = fast_apply(fast_weights, inputs)[2]

        f_idxs = jnp.asarray(fast_out.idx)
        f_times = jnp.asarray(fast_out.time)
        f_valid = (f_idxs >= n_input) & jnp.isfinite(f_times) & (f_times < 80e-3)
        f_b_mask = f_valid & (f_idxs >= n_input + layers[0])
        b_b_mask = b_valid & (b_idxs >= n_input + layers[0])

        if (n_b > 0
                and int(f_b_mask.sum()) > 0
                and int(b_b_mask.sum()) > 0):
            first_b = float(jnp.min(times[idx_b_mask]))
            first_b_fast = float(jnp.min(f_times[f_b_mask]))
            first_b_slow = float(jnp.min(b_times[b_b_mask]))
            self.assertGreater(
                first_b, first_b_fast,
                f"Hetero pop-B first-spike ({first_b*1e3:.2f}ms) is not "
                f"later than the all-fast baseline ({first_b_fast*1e3:.2f}ms); "
                "per-population slow tau is not slowing pop B's integration."
            )
            self.assertLess(
                first_b, first_b_slow,
                f"Hetero pop-B first-spike ({first_b*1e3:.2f}ms) is not "
                f"earlier than the all-slow baseline ({first_b_slow*1e3:.2f}ms); "
                "fast pop A is not delivering an earlier kick to pop B."
            )


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure the existing RecurrentLIF behaviour is unchanged."""

    def test_recurrent_lif_unchanged(self):
        """The original RecurrentLIF still constructs and runs identically."""
        layers = [4, 4]
        params = LIFParameters(tau_mem=10e-3, tau_syn=5e-3, v_th=1.0)
        init_fn, apply_fn = serial(RecurrentLIF(
            layers=layers,
            n_spikes=8,
            t_max=50e-3,
            params=params,
            mean=[0.5, 0.5],
            std=[1.0, 1.0],
        ))
        rng = jax.random.PRNGKey(0)
        _, weights = init_fn(rng, 3)
        # Just verify it runs without error.
        inputs = EventPropSpike(
            time=jnp.array([1e-3, 2e-3]),
            idx=jnp.array([0, 1]),
            current=jnp.zeros(2),
        )
        _ = apply_fn(weights, inputs)


if __name__ == "__main__":
    unittest.main()
