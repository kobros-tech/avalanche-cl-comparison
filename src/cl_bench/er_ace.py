"""
ER-ACE (Experience Replay with Asymmetric Cross-Entropy).

Ported and lightly adapted from AlbinSou/ocl_survey
(https://github.com/AlbinSou/ocl_survey, `src/strategies/erace.py`), the
code release accompanying:

    Soutif-Cormerais, Carta, Cossu, Hurtado, Lomonaco, Van de Weijer, Hemati.
    "A Comprehensive Empirical Evaluation on Online Continual Learning."
    ICCV Workshop 2023. https://arxiv.org/abs/2308.10328

Original method: Caccia et al., "New Insights on Reducing Abrupt
Representation Change in Online Continual Learning," ICLR 2022
(https://openreview.net/forum?id=N8MaByOzUfb).

Not part of stock Avalanche (`avalanche.training`) as of avalanche-lib
0.6.0 -- included here because it's a genuinely stronger, still-simple
replay baseline than plain `Replay`/`GEM`/`A-GEM`, and it's a good
apples-to-apples comparison point for the Skill Memory demo since both
are trying to solve the same "avoid abrupt representation change" problem
with very different mechanisms (asymmetric loss weighting vs. an explicit
per-skill parameter registry).

Changes from the original: only import-path fixes for avalanche-lib
0.6.0 and docstring/comment cleanup; the training logic is unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.regularization import RegularizationMethod
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


def cross_entropy_with_oh_targets(outputs, targets, eps=1e-5):
    """Cross-entropy that accepts soft/one-hot targets (which must sum to 1)."""
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    ce = -(targets * outputs.log()).sum(1)
    return ce.mean()


class ACECriterion(RegularizationMethod):
    """Asymmetric cross-entropy (ACE): the *current* minibatch's loss is
    restricted to only the classes present in that minibatch (so old-class
    logits aren't pushed down just because they're absent from this batch),
    while the *replay buffer* minibatch uses ordinary cross-entropy over
    all classes seen so far. This asymmetry is the core trick that reduces
    "abrupt representation change" relative to plain Replay."""

    def __call__(
        self,
        out_in,
        target_in,
        out_buffer,
        target_buffer,
        weight_current=0.5,
        weight_buffer=0.5,
    ):
        current_classes = torch.unique(target_in)
        loss_buffer = F.cross_entropy(out_buffer, target_buffer)
        oh_target_in = F.one_hot(target_in, num_classes=out_in.shape[1])
        oh_target_in = oh_target_in[:, current_classes]
        loss_current = cross_entropy_with_oh_targets(
            out_in[:, current_classes], oh_target_in
        )
        return weight_buffer * loss_buffer + weight_current * loss_current


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


class ER_ACE(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        batch_size_mem: int,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        alpha: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=None,
        eval_every=-1,
        peval_mode="experience",
    ):
        """
        :param mem_size: replay buffer capacity (class-balanced).
        :param alpha: weight on the *current*-batch ACE loss; (1 - alpha)
            weights the ordinary cross-entropy loss on the replayed batch.
        :param batch_size_mem: minibatch size sampled from the buffer.
        """
        if evaluator is None:
            evaluator = default_evaluator()
        super().__init__(
            model, optimizer, criterion, train_mb_size, train_epochs,
            eval_mb_size, device, plugins, evaluator, eval_every, peval_mode,
        )
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(max_size=self.mem_size, adaptive_size=True)
        self.replay_loader = None
        self.ace_criterion = ACECriterion()
        self.alpha = alpha

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(self.replay_loader)
                self.mb_buffer_x = self.mb_buffer_x.to(self.device)
                self.mb_buffer_y = self.mb_buffer_y.to(self.device)
                self.mb_buffer_tid = self.mb_buffer_tid.to(self.device)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(self.model, self.mb_buffer_x, self.mb_buffer_tid)
            self._after_forward(**kwargs)

            if self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.ace_criterion(
                    self.mb_output, self.mb_y, self.mb_buffer_out, self.mb_buffer_y,
                    weight_current=self.alpha, weight_buffer=(1 - self.alpha),
                )

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = _cycle(
                torch.utils.data.DataLoader(
                    buffer, batch_size=self.batch_size_mem, shuffle=True, drop_last=True,
                )
            )
        else:
            self.replay_loader = None
        super()._before_training_exp(**kwargs)

    def _train_cleanup(self):
        super()._train_cleanup()
        self.replay_loader = None
