# new file: src/multi_opt_trainer.py
import torch
from transformers import get_scheduler
from torch import Tensor
from src.custom_trainer import DebugSeq2SeqTrainer
import torch.distributed as dist


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
             # You might want to raise a warning or error if a closure is passed,
             # as Muon's logic doesn't typically involve re-evaluation within the step.
             # For now, we'll just ignore it.
             pass
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"] * getattr(p_world, "wd_mul", 1.0))
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

class _ComboOptim(torch.optim.Optimizer):
    """Forwards step()/zero_grad() to each inner optimizer
       and exposes a merged param_groups view."""
    def __init__(self, optim_list):
        self.optim_list = optim_list
        if not optim_list:
            raise ValueError("optim_list cannot be empty for _ComboOptim")

        # Use defaults from the first optimizer for the base class initialization.
        # This is primarily needed for compatibility with schedulers like
        # get_scheduler which might expect optimizer.defaults['lr'].
        # The actual LR used for each group comes from the group itself.
        first_optimizer_defaults = optim_list[0].defaults
        if 'lr' not in first_optimizer_defaults:
            # Fallback if the first optimizer doesn't have 'lr' in defaults.
            # This might happen with custom optimizers. Provide a dummy default.
            # The scheduler should use the 'lr' from each param_group anyway.
            print("Warning: First optimizer in _ComboOptim lacks 'lr' in defaults. Using a placeholder default for scheduler compatibility.")
            first_optimizer_defaults = {'lr': 1e-3} # Placeholder default LR

        # Initialize self.param_groups as a standard list attribute BEFORE calling super().__init__
        self.param_groups = []

        # We still need to call super().__init__ to initialize other necessary attributes like self.state
        # Pass dummy parameters but use the defaults from the first optimizer.
        super().__init__([torch.nn.Parameter(torch.empty(0))], first_optimizer_defaults) # Pass defaults here

        # Now, construct the actual combined param_groups list
        combined_groups = []
        for opt in self.optim_list:
            combined_groups.extend(opt.param_groups)
        # Assign the combined list to the instance attribute, overwriting the dummy group added by super().__init__
        self.param_groups = combined_groups

    # Remove the @property getter as self.param_groups is now a regular list attribute

    def step(self, closure=None):
        for opt in self.optim_list:
            opt.step(closure)

    def zero_grad(self, set_to_none=False):
        for opt in self.optim_list:
            opt.zero_grad(set_to_none=set_to_none)

 

class MultiOptSeq2SeqTrainer(DebugSeq2SeqTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            m = self.model
            # Ensure parameters are converted to lists
            adamw_params = list(m.decoder.lm_head.parameters())
            muon_params = list(m.encoder.parameters()) + list(m.decoder.model.parameters())

            # Filter out any potential empty parameter lists
            if not adamw_params:
                 raise ValueError("AdamW optimizer received no parameters for lm_head.")
            if not muon_params:
                 raise ValueError("Muon optimizer received no parameters for encoder/decoder.")

            o1 = torch.optim.AdamW(
                adamw_params,
                lr=self.args.learning_rate, # Use learning_rate from args for AdamW
                betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay,
                fused=True,
            )
            # You might want a different LR for Muon, adjust as needed
            muon_lr = 0.0001 # Example: Keep Muon LR fixed or derive from args
            o2 = Muon(
                muon_params,
                lr=muon_lr,
                momentum=0.95,
                # Pass rank and world_size if using distributed training
                # rank=self.args.local_rank if dist.is_initialized() else 0,
                # world_size=self.args.world_size if dist.is_initialized() else 1,
            )
            # Ensure o1 is first if its defaults are used by _ComboOptim
            self.optimizer = _ComboOptim([o1, o2])
        return self.optimizer
