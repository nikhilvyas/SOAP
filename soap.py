import gc
import string
from typing import List

import torch
import torch.optim as optim


_mode = None

if _mode is None:
    def decorator(func):
        return func
else:
    decorator = torch.compile(fullgraph=False, dynamic=True, mode=_mode)

_einsum_base = string.ascii_lowercase + string.ascii_uppercase


def warmup(lr: float, step: int, warmup_steps: int):
    return lr * min(step / warmup_steps, 1)


def dim_merger(grad, max_precond_dim):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape >= max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)
    return new_grad


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta ** step)


def exp_avg_sq_(state, grad, beta2, eps):
    torch._foreach_mul_(state, beta2)
    torch._foreach_addcmul_(state, grad, grad, value=1 - beta2)
    denom = torch._foreach_sqrt(state)
    torch._foreach_maximum_(denom, eps)
    return denom


def set_(dst: torch.Tensor, src: torch.Tensor):
    if src.is_contiguous() and dst.is_contiguous() and src.dtype == dst.dtype:
        dst.set_(src)
    else:
        dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


@decorator
def get_orthogonal_matrix_QR(GG, Q, exp_avg_sq, max_precond_dim=10000, merge_dims=False):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    indices = []

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        del m
        est_eig = torch.einsum('ij,ij->j', o, tmp)
        del o
        sort_idx = torch.argsort(est_eig, descending=True)
        del est_eig
        indices.append(sort_idx)
        power_iter = tmp[:, sort_idx]
        set_(q, torch.linalg.qr(power_iter)[0])
        del power_iter

    exp_avg_sq_new = exp_avg_sq
    if merge_dims:
        exp_avg_sq_new = dim_merger(exp_avg_sq, max_precond_dim)

    indices = tuple(slice(None) if ind is None else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq_new.dim() - i - 1))  #
                    for i, ind in enumerate(indices))
    exp_avg_sq_new = exp_avg_sq_new[indices]

    if merge_dims:
        exp_avg_sq_new = exp_avg_sq_new.reshape(exp_avg_sq.shape)
    set_(exp_avg_sq, exp_avg_sq_new)


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, 'cpu'):
            if modifier is not None:
                m = m.to(modifier)
            try:
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device,
                                                                                                dtype=dtype)
                break
            except torch.OutOfMemoryError:
                pass
            except RuntimeError:  # failed to compute eigenvalues
                continue
            clean()
        else:
            raise RuntimeError("Failed to compute eigenvalues.")

        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    return final


@decorator
def compute_ggt(grad, GG, max_precond_dim, merge_dims, precondition_1d, beta):
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return

    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    for idx, sh in enumerate(grad.shape):
        if sh > max_precond_dim:
            continue
        b = _einsum_base[idx]
        g0 = _einsum_base[:grad.dim()]
        g1 = g0.replace(b, b.upper())
        outer_product = torch.einsum(f'{g0},{g1}->{b + b.upper()}', grad, grad)
        GG[idx].lerp_(promote(outer_product), 1 - beta)


def promote(x):
    if x is (torch.bfloat16, torch.float16):
        return torch.float32
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


def update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    compute_ggt(grad, state['GG'], max_precond_dim, merge_dims, precondition_1d, beta)
    if state['Q'] is None:
        state['Q'] = get_orthogonal_matrix(state['GG'])
    if update_precond:
        get_orthogonal_matrix_QR(state['GG'], state['Q'], state['exp_avg_sq'], max_precond_dim, merge_dims)


def init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['Q'] = None  # Will hold all the eigenbases of the preconditioner.
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append([])
            return
        state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype))
        return

    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    for sh in grad.shape:
        if sh > max_precond_dim:
            state['GG'].append([])
        else:
            state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))


@decorator
def project(grad, Q, merge_dims, max_precond_dim, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    original_shape = grad.shape
    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    param = _einsum_base[:grad.dim()]
    preconditioners = ",".join((g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if len(m) > 0)
    if preconditioners:
        out = ''.join(c.upper() if c.upper() in preconditioners else c for c in param)
        grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if len(q) > 0])
    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


def copy_stochastic_list_(target: List[torch.Tensor], source: List[torch.Tensor]):
    for t, s in zip(target, source):
        if t.dtype == torch.bfloat16:
            copy_stochastic_(t, s)
        elif t.data_ptr() != s.data_ptr():
            t.copy_(s)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    if target.data_ptr() == source.data_ptr():
        return

    """Taken as-is from https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905"""
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def update_param_(param: List[torch.Tensor], update: List[torch.Tensor], lr: float, decay: float,
                  add_fn: callable = None):
    param32 = [promote(p) for p in param]
    update32 = [promote(u) for u in update]
    if decay > 0:
        torch._foreach_mul_(param32, 1 - decay * lr)
    if add_fn is None:
        torch._foreach_add_(param32, update32, alpha=lr)
    else:
        add_fn(param32, update32, lr)
    copy_stochastic_list_(param, param32)


class SOAP(optim.Optimizer):
    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 10000,  #
                 merge_dims: bool = False, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1):
        defaults = {"lr": lr, "betas": betas, "shampoo_beta": shampoo_beta, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps}
        super().__init__(params, defaults)
        if data_format not in ('channels_first',):
            raise ValueError("'channels_last' is currently not supported.")
        self._data_format = data_format

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            vals = []
            step = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                merge_dims = group['merge_dims']
                max_precond_dim = group['max_precond_dim']
                precondition_1d = group['precondition_1d']

                grad = p.grad.float()
                p.grad = None

                state = self.state[p]
                step = state['step'] = state.get("step", -1) + 1

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
                    init_preconditioner(grad, state, max_precond_dim, precondition_1d, merge_dims)
                    update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = project(grad, state['Q'], merge_dims, max_precond_dim, False)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                vals.append((p, grad, grad_projected, exp_avg, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, exp_avg, exp_avg_sq = zip(*vals)
            beta1, beta2 = group["betas"]

            old_debiased1 = beta_debias(beta1, step)
            old_debiased2 = beta_debias(beta2, step)

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            torch._foreach_mul_(exp_avg, old_debiased1)
            torch._foreach_add_(exp_avg, grad, alpha=1 - old_debiased1)
            denom = exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, group['eps'])

            for p, g, ea, d in zip(p_list, grad, exp_avg, denom):
                state = self.state[p]
                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = project(ea, state['Q'], merge_dims, max_precond_dim, False)

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                set_(d, project(exp_avg_projected / d, state['Q'], merge_dims, max_precond_dim, True))

                update_preconditioner(g, state, max_precond_dim, merge_dims, precondition_1d, old_debiased2,
                                      step > 0 and step % group['precondition_frequency'] == 0)

            # Why does this have to be rebiased here?
            step_size = -group["lr"] * min(step / group['warmup_steps'], 1)
            if group["normalize_grads"]:
                norm = torch._foreach_norm(denom)
                torch._foreach_mul_(norm, [d.size ** -0.5 for d in denom])
                torch._foreach_add_(norm, 1e-30)  # 1e-30 has no effect, what about eps?
                torch._foreach_div_(denom, norm)
            update_param_(p_list, denom, step_size, group["weight_decay"])
        return loss
