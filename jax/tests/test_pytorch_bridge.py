"""
Tests for PyTorch bridge and Viterbi integration.

Created: 2025-11-17
"""

import pytest
import torch
import jax.numpy as jnp

from chmm_jax import init_chmm
from chmm_jax.pytorch_bridge import TorchCHMM, TorchCHMMSensory, TorchCHMMFromPretrained
from chmm_jax.message_passing import viterbi


def test_torch_chmm_viterbi_basic():
    """Test basic Viterbi functionality in TorchCHMM."""
    # Create TorchCHMM
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    # Create test sequence
    observations = torch.tensor([0, 1, 2, 5, 4, 3, 0, 1, 2], dtype=torch.long)
    actions = torch.tensor([2, 2, 1, 3, 3, 2, 2, 2], dtype=torch.long)

    # Run Viterbi
    states, log_prob = chmm.viterbi(observations, actions)

    # Check output shapes and types
    assert states.shape == (len(observations),)
    assert states.dtype == torch.long
    assert log_prob.shape == ()
    assert log_prob.dtype == torch.float32

    # Check log probability is finite
    assert torch.isfinite(log_prob)

    # Check states are valid
    assert torch.all(states >= 0)
    assert torch.all(states < 27)


def test_torch_chmm_viterbi_no_gradients():
    """Test that Viterbi does not create gradients."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    # Create test sequence with requires_grad=True
    observations = torch.tensor([0, 1, 2, 5, 4], dtype=torch.long)
    actions = torch.tensor([2, 2, 1, 3], dtype=torch.long)

    # Ensure CHMM parameters require gradients
    assert chmm.log_T_logits.requires_grad
    assert chmm.Pi_x.requires_grad

    # Run Viterbi
    states, log_prob = chmm.viterbi(observations, actions)

    # Check that outputs do not require gradients
    assert not states.requires_grad
    assert not log_prob.requires_grad

    # Verify we can't backprop through Viterbi (expected behavior)
    # This should not raise an error, but gradients should be None
    log_prob_copy = log_prob.clone().requires_grad_(True)
    try:
        log_prob_copy.backward()
        # If we get here, the gradient should be None for parameters
        assert chmm.log_T_logits.grad is None
        assert chmm.Pi_x.grad is None
    except RuntimeError:
        # This is also acceptable - Viterbi shouldn't support backprop
        pass


def test_torch_chmm_viterbi_consistency():
    """Test that PyTorch Viterbi matches JAX implementation."""
    # Create identical CHMMs in JAX and PyTorch
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm_jax = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    chmm_torch = TorchCHMMFromPretrained(chmm_jax)

    # Test sequence
    observations = torch.tensor([0, 1, 2, 5, 4, 3], dtype=torch.long)
    actions = torch.tensor([2, 2, 1, 3, 3], dtype=torch.long)

    # Run PyTorch Viterbi
    states_torch, log_prob_torch = chmm_torch.viterbi(observations, actions)

    # Run JAX Viterbi
    obs_jax = jnp.array(observations.numpy(), dtype=jnp.int32)
    actions_jax = jnp.array(actions.numpy(), dtype=jnp.int32)
    states_jax, log_prob_jax = viterbi(
        chmm_jax.T,
        chmm_jax.Pi_x,
        n_clones,
        obs_jax,
        actions_jax
    )

    # Check numerical consistency
    assert torch.allclose(
        states_torch,
        torch.from_numpy(states_jax.__array__()).long(),
        atol=0
    )
    assert torch.allclose(
        log_prob_torch,
        torch.tensor(float(log_prob_jax)),
        atol=1e-5
    )


def test_torch_chmm_sensory_viterbi():
    """Test Viterbi for sensory-only CHMM."""
    # Create sensory CHMM
    chmm = TorchCHMMSensory(n_states=27, seed=42)

    # Test sequence (no actions needed)
    observations = torch.tensor([0, 1, 2, 5, 4, 3, 0, 1, 2], dtype=torch.long)

    # Run Viterbi
    states, log_prob = chmm.viterbi(observations)

    # Check output shapes and types
    assert states.shape == (len(observations),)
    assert states.dtype == torch.long
    assert log_prob.shape == ()
    assert log_prob.dtype == torch.float32

    # Check log probability is finite
    assert torch.isfinite(log_prob)

    # Check states are valid
    assert torch.all(states >= 0)
    assert torch.all(states < 27)


def test_viterbi_single_observation():
    """Test Viterbi with single observation."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    # Single observation
    observations = torch.tensor([0], dtype=torch.long)
    actions = torch.tensor([], dtype=torch.long)

    # Run Viterbi
    states, log_prob = chmm.viterbi(observations, actions)

    # Check output
    assert states.shape == (1,)
    assert torch.isfinite(log_prob)
    assert 0 <= states[0] < 27


def test_viterbi_state_observation_consistency():
    """Test that Viterbi states match their observations."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    observations = torch.tensor([0, 1, 2, 5, 4, 3], dtype=torch.long)
    actions = torch.tensor([2, 2, 1, 3, 3], dtype=torch.long)

    states, log_prob = chmm.viterbi(observations, actions)

    # Check that each state belongs to the correct observation
    # For uniform clones (27 states, 9 observations = 3 clones each)
    for t, (obs, state) in enumerate(zip(observations, states)):
        obs_start = obs.item() * 3
        obs_stop = (obs.item() + 1) * 3
        assert obs_start <= state < obs_stop, (
            f"At t={t}: state {state} should be in range [{obs_start}, {obs_stop}) "
            f"for observation {obs}"
        )


def test_torch_chmm_forward_batch():
    """Test batched forward pass with vmap."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    # Batch of 3 sequences, all length 5
    observations = torch.tensor([
        [0, 1, 2, 5, 4],
        [4, 3, 6, 7, 8],
        [1, 2, 3, 4, 5],
    ], dtype=torch.long)
    actions = torch.tensor([
        [2, 2, 1, 3],
        [3, 1, 2, 1],
        [0, 1, 3, 2],
    ], dtype=torch.long)

    # Run batched forward
    log_liks, posteriors = chmm.forward_batch(observations, actions)

    # Check shapes
    assert log_liks.shape == (3,), f"Expected shape (3,), got {log_liks.shape}"
    assert posteriors.shape[0] == 3, f"Expected batch size 3"
    assert posteriors.shape[1] == 5, f"Expected T=5"

    # Check all finite
    assert torch.all(torch.isfinite(log_liks))
    assert torch.all(torch.isfinite(posteriors))


def test_torch_chmm_forward_batch_vs_single():
    """Test that batched forward matches individual forwards."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    # Test sequences
    observations = torch.tensor([
        [0, 1, 2, 5, 4],
        [4, 3, 6, 7, 8],
    ], dtype=torch.long)
    actions = torch.tensor([
        [2, 2, 1, 3],
        [3, 1, 2, 1],
    ], dtype=torch.long)

    # Run batched
    log_liks_batch, _ = chmm.forward_batch(observations, actions)

    # Run individually
    log_liks_individual = []
    for i in range(2):
        log_lik, _ = chmm(observations[i], actions[i])
        log_liks_individual.append(log_lik)
    log_liks_individual = torch.stack(log_liks_individual)

    # Should match
    assert torch.allclose(log_liks_batch, log_liks_individual, atol=1e-5)


def test_torch_chmm_forward_batch_gradients():
    """Test that gradients flow through batched forward."""
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    observations = torch.tensor([
        [0, 1, 2, 5, 4],
        [4, 3, 6, 7, 8],
    ], dtype=torch.long)
    actions = torch.tensor([
        [2, 2, 1, 3],
        [3, 1, 2, 1],
    ], dtype=torch.long)

    # Run forward_batch
    log_liks, posteriors = chmm.forward_batch(observations, actions)

    # Compute loss and backprop
    loss = -log_liks.mean()
    loss.backward()

    # Check gradients exist
    assert chmm.log_T_logits.grad is not None
    assert chmm.Pi_x.grad is not None
    assert torch.all(torch.isfinite(chmm.log_T_logits.grad))
    assert torch.all(torch.isfinite(chmm.Pi_x.grad))


def test_torch_chmm_sensory_forward_batch():
    """Test batched forward for sensory-only CHMM."""
    chmm = TorchCHMMSensory(n_states=27, seed=42)

    # Batch of observations (no actions)
    observations = torch.tensor([
        [0, 1, 2, 5, 4],
        [4, 3, 6, 7, 8],
        [1, 2, 3, 4, 5],
    ], dtype=torch.long)

    # Run batched forward
    log_liks, posteriors = chmm.forward_batch(observations)

    # Check shapes
    assert log_liks.shape == (3,)
    assert posteriors.shape[0] == 3
    assert posteriors.shape[1] == 5

    # Check all finite
    assert torch.all(torch.isfinite(log_liks))
    assert torch.all(torch.isfinite(posteriors))


def test_torch_chmm_from_pretrained_forward_batch():
    """Test batched forward for pretrained CHMM."""
    # Create JAX CHMM
    n_clones = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    chmm_jax = init_chmm(
        n_clones=n_clones,
        n_observations=9,
        n_actions=4,
        seed=42
    )

    # Wrap in PyTorch
    chmm = TorchCHMMFromPretrained(chmm_jax)

    # Batch of sequences
    observations = torch.tensor([
        [0, 1, 2, 5, 4],
        [4, 3, 6, 7, 8],
    ], dtype=torch.long)
    actions = torch.tensor([
        [2, 2, 1, 3],
        [3, 1, 2, 1],
    ], dtype=torch.long)

    # Run batched forward
    log_liks, posteriors = chmm.forward_batch(observations, actions)

    # Check shapes and values
    assert log_liks.shape == (2,)
    assert torch.all(torch.isfinite(log_liks))


def test_forward_soft_encoder_gradient_flow():
    """End-to-end gradient flow: encoder -> forward_soft -> loss -> encoder.grad.

    This is the main claim of the soft-observation interface: CHMM likelihood
    gradients reach an upstream PyTorch encoder through log_obs_weights.
    """
    import torch.nn as nn

    encoder = nn.Linear(16, 9)  # 9 observations
    chmm = TorchCHMM(n_states=27, n_actions=4, seed=42)

    x = torch.randn(5, 16, requires_grad=True)  # 5 timesteps
    actions = torch.randint(0, 4, (4,))

    # Encoder produces soft observation weights (differentiable)
    log_obs_weights = encoder(x)  # [5, 9]
    log_lik, posteriors = chmm.forward_soft(log_obs_weights, actions)

    loss = -log_lik
    loss.backward()

    # Gradients must reach the encoder parameters
    assert encoder.weight.grad is not None, "Encoder weight should have gradients"
    assert torch.any(encoder.weight.grad != 0), "Encoder gradients should be non-zero"
    assert torch.all(torch.isfinite(encoder.weight.grad)), "Encoder gradients should be finite"

    # Gradients must reach the input
    assert x.grad is not None, "Input x should have gradients"
    assert torch.any(x.grad != 0), "Input gradients should be non-zero"

    # CHMM parameters should also have gradients
    assert chmm.log_T_logits.grad is not None, "T should have gradients"
    assert chmm.Pi_x.grad is not None, "Pi_x should have gradients"

    # Posteriors should be valid
    assert posteriors.shape == (5, 27)
    assert torch.all(torch.isfinite(posteriors))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
