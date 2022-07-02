"""LightiningModules with Connectionist Temporal Classification Loss."""
import argparse
import itertools

import torch

from .base import BaseImageToTextLitModel
from .util import first_appearance


def compute_input_lengths(padded_sequences: torch.Tensor) -> torch.Tensor:
    """Treating trailing zeros as padding, compute non-padded lengths.

    Parameters
    ----------
    padded_sequences
        (N, S) tensor where elements that equal 0 correspond to padding

    Returns
    -------
    torch.Tensor
        (N,) tensor where each element corresponds to the non-padded length of each sequence

    Examples
    --------
    >>> X = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0], [1, 2, 3, 0, 5]])
    >>> compute_input_lengths(X)
    tensor([2, 3, 5])
    """
    lengths = torch.arange(padded_sequences.shape[1]).type_as(padded_sequences)
    return ((padded_sequences > 0) * lengths).argmax(1) + 1


class CTCLitModel(BaseImageToTextLitModel):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        self.blank_index = self.inverse_mapping["<B>"]

        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)
        # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

    def forward(self, x):
        return self.model(x)

    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)

        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = first_appearance(y, self.padding_index).type_as(y)
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)

        if with_preds:
            decoded = self.decode(logprobs, max_length=y.shape[1])
            return x, y, logits, loss, decoded
        else:
            return x, y, logits, loss

    def predict(self, x, max_length):
        log_probs = torch.log_softmax(self(x), dim=1)
        decoded = self.decode(log_probs, max_length)
        return decoded

    def decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Greedily decode sequences, collapsing repeated tokens, and removing the CTC blank token.

        See the "Inference" sections of https://distill.pub/2017/ctc/

        Using groupby inspired by https://github.com/nanoporetech/fast-ctc-decode/blob/master/tests/benchmark.py#L8

        Parameters
        ----------
        logprobs
            (B, C, S) log probabilities
        max_length
            max length of a sequence

        Returns
        -------
        torch.Tensor
            (B, S) class indices
        """
        with torch.no_grad():
            B = logprobs.shape[0]
            argmax = logprobs.argmax(1)
            decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.padding_index
            for i in range(B):
                seq = [b for b, _g in itertools.groupby(argmax[i].tolist()) if b != self.blank_index][:max_length]
                for ii, char in enumerate(seq):
                    decoded[i, ii] = char
        return decoded
