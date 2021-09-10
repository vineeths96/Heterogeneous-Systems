import torch
import torch.distributed

from seed import set_seed


class TensorBuffer:
    """
    Class to flatten and deflatten the gradient vector.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._len_tensors = len(tensors)
        self._tensor_shapes = [tensor.size() for tensor in tensors]

        self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors


class NoneAllReducer:
    """
    All reduce reducer without any compressing.
    """

    def __init__(self, device, timer):
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0

        self._device = device
        self._timer = timer

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.allreduce", verbosity=2):
            if self.n_workers > 1:
                tensor_reduce_op = torch.distributed.all_reduce(tensor=flat_grad.buffer, async_op=True)
                tensor_reduce_op.wait()
            else:
                flat_grad = flat_grad

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1 / self.n_workers)

            bits_communicated += self.n_bits(flat_grad.buffer)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()
