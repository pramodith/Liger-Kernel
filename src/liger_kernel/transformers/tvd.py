from typing import Optional

import torch

from liger_kernel.ops.tvd import LigerTVDFunction


class LigerTVD(torch.nn.Module):
    r"""The Total Variation Distance Module
    .. math::

    TVD = 1/2 * L1_NORM(Student_Logits - Teacher_Logits)

    Args:
        ignore_index (int): The index to ignore in the target. Default: `-100`

    Shape:
        - Input: :math:`(BT, V)`, where B is batch size, T is sequence length, V is vocab size.
        - Target: :math:`(BT, V)`, same shape as the input.
        - shift_labels (Optional): :math:`(BT,)`
        - Output: a scalar.

    Examples:
    ```python
    >>> (B, T, V) = (2, 2, 5)
    >>> jsd = LigerJSD(beta=0.1)
    >>> # input should be a distribution in the log space
    >>> input = torch.randn(B * T, V, requires_grad=True).log_softmax(dim=-1)
    >>> target = torch.randn(B * T, V).log_softmax(dim=-1)
    >>> output = jsd(input, target)
    >>>
    >>> # Example with labels for supervised fine-tuning (SFT) context
    >>> # Assume logits and corresponding labels are given
    >>> student_logits = torch.randn(B * T, V, requires_grad=True).log_softmax(dim=-1)
    >>> teacher_logits = torch.randn(B * T, V).log_softmax(dim=-1)
    >>> labels = torch.randint(0, V, (B * T,), torch.long)
    >>> # Shift so that tokens < n predict n
    >>> shift_student_logits = student_logits[..., :-1, :].contiguous()
    >>> shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    >>> shift_labels = labels[..., 1:].contiguous()
    >>> # Flatten tokens
    >>> shift_student_logits = shift_student_logits.view(-1, V)
    >>> shift_teacher_logits = shift_teacher_logits.view(-1, V)
    >>> shift_labels = shift_labels.view(-1)
    >>> # Calculate loss
    >>> loss_fct = LigerJSD(beta=0.1)
    >>> loss = loss_fct(shift_studetn_logits, shift_teacher_logits, shift_labels)

    ```
    """

    def __init__(self, ignore_index: int = -100, reduction_mode: str = "batchmean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction_mode = reduction_mode

    def forward(
        self,
        log_student: torch.Tensor,
        log_teacher: torch.Tensor,
        shift_labels: Optional[torch.LongTensor] = None,
    ):
        return LigerTVDFunction.apply(
            log_student, log_teacher, shift_labels, self.ignore_index, self.reduction_mode
        )
