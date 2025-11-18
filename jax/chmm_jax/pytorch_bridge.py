"""
PyTorch integration via custom autograd function.

Enables using JAX CHMM modules within PyTorch neural networks with full gradient flow.
Created: 2025-11-03
Modified: 2025-11-17
"""

import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np

from .core import CHMM, init_chmm, forward_backward
from .message_passing import viterbi
from .batching import forward_batch, forward_backward_batch


class JAXFunction(torch.autograd.Function):
    """Custom PyTorch autograd function that wraps JAX computation.

    Enables gradient flow from PyTorch through JAX functions.
    """

    @staticmethod
    def forward(ctx, T_torch, Pi_x_torch, chmm, observations_jax, actions_jax):
        """Forward pass: PyTorch -> JAX -> PyTorch.

        Args:
            ctx: PyTorch autograd context
            T_torch: Transition matrix (PyTorch tensor)
            Pi_x_torch: Initial state distribution (PyTorch tensor)
            chmm: Base CHMM (for structure info)
            observations_jax: Observation sequence (JAX array)
            actions_jax: Action sequence (JAX array)

        Returns:
            log_likelihood, posteriors (as PyTorch tensors)
        """
        # Convert PyTorch tensors to JAX arrays
        T_jax = jnp.array(T_torch.detach().cpu().numpy())
        Pi_x_jax = jnp.array(Pi_x_torch.detach().cpu().numpy())

        # Create CHMM with current parameters
        chmm_current = chmm._replace(T=T_jax, Pi_x=Pi_x_jax)

        # Define function for JAX to differentiate
        def jax_fn(T, Pi_x):
            chmm_temp = chmm._replace(T=T, Pi_x=Pi_x)
            log_lik, posteriors = forward_backward(
                chmm_temp,
                observations_jax,
                actions_jax
            )
            return log_lik, posteriors

        # Compute forward pass and gradients
        (log_lik, posteriors), vjp_fn = jax.vjp(jax_fn, T_jax, Pi_x_jax)

        # Save for backward
        ctx.vjp_fn = vjp_fn
        ctx.save_for_backward(T_torch, Pi_x_torch)

        # Convert results to PyTorch
        log_lik_torch = torch.from_numpy(np.array(log_lik)).float()
        posteriors_torch = torch.from_numpy(np.array(posteriors)).float()

        # Enable gradient tracking
        log_lik_torch.requires_grad = T_torch.requires_grad or Pi_x_torch.requires_grad

        return log_lik_torch, posteriors_torch

    @staticmethod
    def backward(ctx, grad_log_lik, grad_posteriors):
        """Backward pass: compute gradients via JAX vjp.

        Args:
            ctx: PyTorch autograd context
            grad_log_lik: Gradient w.r.t. log_likelihood
            grad_posteriors: Gradient w.r.t. posteriors

        Returns:
            Gradients for (T, Pi_x, chmm, observations, actions)
        """
        T_torch, Pi_x_torch = ctx.saved_tensors

        # Convert PyTorch gradients to JAX
        grad_log_lik_jax = jnp.array(grad_log_lik.cpu().numpy())
        grad_posteriors_jax = jnp.array(grad_posteriors.cpu().numpy())

        # Combine gradients (both outputs contribute)
        grad_output = (grad_log_lik_jax, grad_posteriors_jax)

        # Compute gradients via VJP
        grad_T_jax, grad_Pi_x_jax = ctx.vjp_fn(grad_output)

        # Convert back to PyTorch
        grad_T = torch.from_numpy(np.array(grad_T_jax)).float() if T_torch.requires_grad else None
        grad_Pi_x = torch.from_numpy(np.array(grad_Pi_x_jax)).float() if Pi_x_torch.requires_grad else None

        # Return gradients (None for non-differentiable inputs)
        return grad_T, grad_Pi_x, None, None, None


class JAXFunctionBatch(torch.autograd.Function):
    """Custom PyTorch autograd function for batched JAX computation.

    Uses vmap for 10-50x speedup on same-length sequences.
    """

    @staticmethod
    def forward(ctx, T_torch, Pi_x_torch, chmm, observations_jax, actions_jax):
        """Forward pass: batched PyTorch -> JAX -> PyTorch.

        Args:
            ctx: PyTorch autograd context
            T_torch: Transition matrix (PyTorch tensor)
            Pi_x_torch: Initial state distribution (PyTorch tensor)
            chmm: Base CHMM (for structure info)
            observations_jax: Batched observations [B, T] (JAX array)
            actions_jax: Batched actions [B, T-1] (JAX array)

        Returns:
            log_likelihoods [B], posteriors [B, T, max_block_size] (as PyTorch tensors)
        """
        # Convert PyTorch tensors to JAX arrays
        T_jax = jnp.array(T_torch.detach().cpu().numpy())
        Pi_x_jax = jnp.array(Pi_x_torch.detach().cpu().numpy())

        # Create CHMM with current parameters
        chmm_current = chmm._replace(T=T_jax, Pi_x=Pi_x_jax)

        # Define function for JAX to differentiate
        def jax_fn(T, Pi_x):
            chmm_temp = chmm._replace(T=T, Pi_x=Pi_x)
            log_liks, posteriors = forward_backward_batch(
                chmm_temp,
                observations_jax,
                actions_jax
            )
            return log_liks, posteriors

        # Compute forward pass and gradients
        (log_liks, posteriors), vjp_fn = jax.vjp(jax_fn, T_jax, Pi_x_jax)

        # Save for backward
        ctx.vjp_fn = vjp_fn
        ctx.save_for_backward(T_torch, Pi_x_torch)

        # Convert results to PyTorch
        log_liks_torch = torch.from_numpy(np.array(log_liks)).float()
        posteriors_torch = torch.from_numpy(np.array(posteriors)).float()

        # Enable gradient tracking
        log_liks_torch.requires_grad = T_torch.requires_grad or Pi_x_torch.requires_grad

        return log_liks_torch, posteriors_torch

    @staticmethod
    def backward(ctx, grad_log_liks, grad_posteriors):
        """Backward pass: compute gradients via JAX vjp.

        Args:
            ctx: PyTorch autograd context
            grad_log_liks: Gradient w.r.t. log_likelihoods [B]
            grad_posteriors: Gradient w.r.t. posteriors [B, T, max_block_size]

        Returns:
            Gradients for (T, Pi_x, chmm, observations, actions)
        """
        T_torch, Pi_x_torch = ctx.saved_tensors

        # Convert PyTorch gradients to JAX
        grad_log_liks_jax = jnp.array(grad_log_liks.cpu().numpy())
        grad_posteriors_jax = jnp.array(grad_posteriors.cpu().numpy())

        # Combine gradients (both outputs contribute)
        grad_output = (grad_log_liks_jax, grad_posteriors_jax)

        # Compute gradients via VJP
        grad_T_jax, grad_Pi_x_jax = ctx.vjp_fn(grad_output)

        # Convert back to PyTorch
        grad_T = torch.from_numpy(np.array(grad_T_jax)).float() if T_torch.requires_grad else None
        grad_Pi_x = torch.from_numpy(np.array(grad_Pi_x_jax)).float() if Pi_x_torch.requires_grad else None

        # Return gradients (None for non-differentiable inputs)
        return grad_T, grad_Pi_x, None, None, None


class TorchCHMM(nn.Module):
    """PyTorch wrapper for JAX CHMM.

    Enables using CHMM within PyTorch models with automatic gradient flow.

    Example:
        ```python
        import torch
        import torch.nn as nn
        from chmm_jax.pytorch_bridge import TorchCHMM

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(64, 9)
                self.chmm = TorchCHMM(n_states=27, n_actions=4)
                self.decoder = nn.Linear(27, 10)

            def forward(self, x, actions):
                obs = self.encoder(x)  # (batch, T, 9)
                log_lik, posteriors = self.chmm(obs, actions)
                output = self.decoder(posteriors.mean(dim=1))
                return output, log_lik
        ```

    Args:
        n_states: Total number of hidden states
        n_actions: Number of actions
        n_observations: Number of observations (default: n_states // 3)
        pseudocount: Smoothing parameter
        seed: Random seed
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_observations: int = None,
        pseudocount: float = 1e-10,
        seed: int = 42
    ):
        super().__init__()

        if n_observations is None:
            # Default: 3 clones per observation
            assert n_states % 3 == 0, "n_states must be divisible by 3 for default n_observations"
            n_observations = n_states // 3

        # Initialize CHMM with uniform clones
        n_clones = jnp.full(n_observations, n_states // n_observations, dtype=jnp.int32)
        self.chmm = init_chmm(
            n_clones=n_clones,
            n_observations=n_observations,
            n_actions=n_actions,
            pseudocount=pseudocount,
            seed=seed
        )

        # Register JAX parameters as PyTorch parameters
        # Make writable copies to avoid NumPy warnings
        self.T = nn.Parameter(torch.from_numpy(np.array(self.chmm.T)).float())
        self.Pi_x = nn.Parameter(torch.from_numpy(np.array(self.chmm.Pi_x)).float())

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CHMM.

        Args:
            observations: Observation sequence [T] (int64)
            actions: Action sequence [T-1] (int64)

        Returns:
            log_likelihood: Log P(observations, actions)
            posteriors: Posterior state probabilities [varies] (compressed)
        """
        # Convert observations and actions to JAX arrays
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.detach().cpu().numpy(), dtype=jnp.int32)

        # Call custom autograd function
        log_lik, posteriors = JAXFunction.apply(
            self.T,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_lik, posteriors

    @torch.no_grad()
    def viterbi(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode most likely state sequence (inference only, no gradients).

        Uses Viterbi algorithm (max-product message passing) to find the single
        most likely path through the hidden states. This is useful for:
        - Post-training analysis and interpretability
        - Debugging what the model learned
        - Clone pruning (identifying unused states)
        - Planning and vicarious evaluation

        Note: This method does NOT support gradient flow. Use forward() for
        training with backpropagation through forward-backward algorithm.

        Args:
            observations: Observation sequence [T] (int64)
            actions: Action sequence [T-1] (int64)

        Returns:
            states: Most likely state sequence [T] (int64, global state indices)
            log_prob: Log probability of the most likely path (float32)

        Example:
            ```python
            # After training
            model.eval()
            with torch.no_grad():
                states, log_prob = model.chmm.viterbi(test_obs, test_actions)
                print(f"Most likely path: {states}")
                print(f"Path log-probability: {log_prob}")
            ```
        """
        # Convert to JAX arrays
        obs_jax = jnp.array(observations.cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.cpu().numpy(), dtype=jnp.int32)

        # Get current parameters from PyTorch
        T_jax = jnp.array(self.T.detach().cpu().numpy())
        Pi_x_jax = jnp.array(self.Pi_x.detach().cpu().numpy())

        # Call JAX Viterbi
        states_jax, log_prob_jax = viterbi(
            T_jax,
            Pi_x_jax,
            self.chmm.n_clones,
            obs_jax,
            actions_jax
        )

        # Convert back to PyTorch
        states_torch = torch.from_numpy(np.array(states_jax)).long()
        log_prob_torch = torch.from_numpy(np.array(log_prob_jax)).float()

        return states_torch, log_prob_torch

    def forward_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass through CHMM with vmap parallelization.

        Processes multiple sequences in parallel for 10-50x speedup.
        All sequences must be the same length.

        Args:
            observations: Batched observations [B, T] (int64)
            actions: Batched actions [B, T-1] (int64)

        Returns:
            log_likelihoods: Log P(observations, actions) per sequence [B]
            posteriors: Posterior state probabilities [B, T, max_block_size] (padded)

        Example:
            ```python
            # Batch of 3 sequences, all length 5
            obs = torch.tensor([[0, 1, 2, 5, 4], [4, 3, 6, 7, 8], [1, 2, 3, 4, 5]])
            acts = torch.tensor([[2, 2, 1, 3], [3, 1, 2, 1], [0, 1, 3, 2]])
            log_liks, posteriors = model.chmm.forward_batch(obs, acts)
            # log_liks.shape: (3,)
            # posteriors.shape: (3, 5, max_block_size)
            ```
        """
        # Convert observations and actions to JAX arrays
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.detach().cpu().numpy(), dtype=jnp.int32)

        # Call custom autograd function for batched processing
        log_liks, posteriors = JAXFunctionBatch.apply(
            self.T,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_liks, posteriors

    def update_from_chmm(self, chmm: CHMM):
        """Update PyTorch parameters from a JAX CHMM.

        Useful after EM training in JAX.

        Args:
            chmm: Trained JAX CHMM
        """
        self.chmm = chmm
        self.T.data = torch.from_numpy(np.array(chmm.T)).float()
        self.Pi_x.data = torch.from_numpy(np.array(chmm.Pi_x)).float()


class TorchCHMMFromPretrained(nn.Module):
    """Load a pretrained JAX CHMM into PyTorch.

    Args:
        chmm: Pretrained JAX CHMM
    """

    def __init__(self, chmm: CHMM):
        super().__init__()

        self.chmm = chmm

        # Register parameters
        self.T = nn.Parameter(torch.from_numpy(np.array(chmm.T)).float())
        self.Pi_x = nn.Parameter(torch.from_numpy(np.array(chmm.Pi_x)).float())

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.detach().cpu().numpy(), dtype=jnp.int32)

        log_lik, posteriors = JAXFunction.apply(
            self.T,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_lik, posteriors

    @torch.no_grad()
    def viterbi(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode most likely state sequence (inference only, no gradients).

        See TorchCHMM.viterbi() for full documentation.
        """
        obs_jax = jnp.array(observations.cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.cpu().numpy(), dtype=jnp.int32)

        T_jax = jnp.array(self.T.detach().cpu().numpy())
        Pi_x_jax = jnp.array(self.Pi_x.detach().cpu().numpy())

        states_jax, log_prob_jax = viterbi(
            T_jax,
            Pi_x_jax,
            self.chmm.n_clones,
            obs_jax,
            actions_jax
        )

        states_torch = torch.from_numpy(np.array(states_jax)).long()
        log_prob_torch = torch.from_numpy(np.array(log_prob_jax)).float()

        return states_torch, log_prob_torch

    def forward_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass with vmap parallelization.

        See TorchCHMM.forward_batch() for full documentation.
        """
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)
        actions_jax = jnp.array(actions.detach().cpu().numpy(), dtype=jnp.int32)

        log_liks, posteriors = JAXFunctionBatch.apply(
            self.T,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_liks, posteriors


class TorchCHMMSensory(nn.Module):
    """PyTorch wrapper for sensory-only CHMM (no action conditioning).

    For passive perception tasks where observations form a sequence but there's
    no action control - e.g., language modeling, time series, video understanding.

    This marginalizes over actions, learning P(z_{n+1}|z_n) directly instead of
    P(z_{n+1}|z_n, a). Equivalent to a standard CHMM (Dedieu et al. 2019).

    Example:
        ```python
        import torch
        from chmm_jax.pytorch_bridge import TorchCHMMSensory

        # Sensory-only CHMM
        chmm = TorchCHMMSensory(n_states=81)

        # Process observation sequence (no actions needed)
        observations = torch.randint(0, 9, (100,))
        log_lik, posteriors = chmm(observations)
        ```

    Args:
        n_states: Total number of hidden states
        n_observations: Number of observations (default: n_states // 3)
        pseudocount: Smoothing parameter
        seed: Random seed
    """

    def __init__(
        self,
        n_states: int,
        n_observations: int = None,
        pseudocount: float = 1e-10,
        seed: int = 42
    ):
        super().__init__()

        if n_observations is None:
            # Default: 3 clones per observation
            assert n_states % 3 == 0, "n_states must be divisible by 3 for default n_observations"
            n_observations = n_states // 3

        # Initialize CHMM with single dummy action dimension
        n_clones = jnp.full(n_observations, n_states // n_observations, dtype=jnp.int32)
        self.chmm = init_chmm(
            n_clones=n_clones,
            n_observations=n_observations,
            n_actions=1,  # Single dummy action
            pseudocount=pseudocount,
            seed=seed
        )

        # Extract transition matrix (marginalized over single action)
        # T shape: (n_actions=1, n_states, n_states)
        # We take T[0] to get (n_states, n_states) sensory transition matrix
        T_sensory = self.chmm.T[0]

        # Register as PyTorch parameters
        self.T = nn.Parameter(torch.from_numpy(np.array(T_sensory)).float())
        self.Pi_x = nn.Parameter(torch.from_numpy(np.array(self.chmm.Pi_x)).float())

    def forward(
        self,
        observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through sensory CHMM.

        Args:
            observations: Observation sequence [T] (int64)

        Returns:
            log_likelihood: Log P(observations)
            posteriors: Posterior state probabilities [varies] (compressed)
        """
        # Convert observations to JAX
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)

        # Create dummy action sequence (all zeros, since we have single action)
        actions_jax = jnp.zeros(len(obs_jax) - 1, dtype=jnp.int32)

        # Add action dimension back for JAXFunction
        T_with_action = self.T.unsqueeze(0)  # (1, n_states, n_states)

        # Call standard forward_backward with dummy actions
        log_lik, posteriors = JAXFunction.apply(
            T_with_action,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_lik, posteriors

    @torch.no_grad()
    def viterbi(
        self,
        observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode most likely state sequence for sensory-only CHMM.

        Uses Viterbi algorithm to find the most likely path through states
        given only observations (no action conditioning).

        Args:
            observations: Observation sequence [T] (int64)

        Returns:
            states: Most likely state sequence [T] (int64, global state indices)
            log_prob: Log probability of the most likely path (float32)

        Example:
            ```python
            # After training sensory CHMM
            model.eval()
            with torch.no_grad():
                states, log_prob = model.viterbi(test_obs)
                print(f"Most likely path: {states}")
            ```
        """
        obs_jax = jnp.array(observations.cpu().numpy(), dtype=jnp.int32)

        # Create dummy actions (all zeros, since we have single action dimension)
        actions_jax = jnp.zeros(len(obs_jax) - 1, dtype=jnp.int32)

        # Get current parameters (add action dimension back)
        T_jax = jnp.array(self.T.detach().cpu().numpy())[None, :, :]  # (1, n_states, n_states)
        Pi_x_jax = jnp.array(self.Pi_x.detach().cpu().numpy())

        # Call JAX Viterbi with dummy actions
        states_jax, log_prob_jax = viterbi(
            T_jax,
            Pi_x_jax,
            self.chmm.n_clones,
            obs_jax,
            actions_jax
        )

        # Convert back to PyTorch
        states_torch = torch.from_numpy(np.array(states_jax)).long()
        log_prob_torch = torch.from_numpy(np.array(log_prob_jax)).float()

        return states_torch, log_prob_torch

    def forward_batch(
        self,
        observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass for sensory-only CHMM with vmap parallelization.

        Processes multiple observation sequences in parallel for 10-50x speedup.
        All sequences must be the same length.

        Args:
            observations: Batched observations [B, T] (int64)

        Returns:
            log_likelihoods: Log P(observations) per sequence [B]
            posteriors: Posterior state probabilities [B, T, max_block_size] (padded)

        Example:
            ```python
            # Batch of 3 sequences, all length 5
            obs = torch.tensor([[0, 1, 2, 5, 4], [4, 3, 6, 7, 8], [1, 2, 3, 4, 5]])
            log_liks, posteriors = model.forward_batch(obs)
            # log_liks.shape: (3,)
            # posteriors.shape: (3, 5, max_block_size)
            ```
        """
        # Convert observations to JAX
        obs_jax = jnp.array(observations.detach().cpu().numpy(), dtype=jnp.int32)

        # Create dummy action sequences (all zeros, since we have single action)
        batch_size, seq_len = observations.shape
        actions_jax = jnp.zeros((batch_size, seq_len - 1), dtype=jnp.int32)

        # Add action dimension back for JAXFunctionBatch
        T_with_action = self.T.unsqueeze(0)  # (1, n_states, n_states)

        # Call batched forward_backward with dummy actions
        log_liks, posteriors = JAXFunctionBatch.apply(
            T_with_action,
            self.Pi_x,
            self.chmm,
            obs_jax,
            actions_jax
        )

        return log_liks, posteriors

    def update_from_chmm(self, chmm: CHMM):
        """Update PyTorch parameters from a JAX CHMM.

        Useful after EM training in JAX.

        Args:
            chmm: Trained JAX CHMM (must have single action dimension)
        """
        assert chmm.T.shape[0] == 1, "Sensory CHMM must have single action dimension"

        self.chmm = chmm
        self.T.data = torch.from_numpy(np.array(chmm.T[0])).float()
        self.Pi_x.data = torch.from_numpy(np.array(chmm.Pi_x)).float()
