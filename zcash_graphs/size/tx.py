from math import ceil
from dataclasses import dataclass, field

from . import compact_size_size, script_size
from .proto3 import p_bytes, p_uint64

ZIP_317_MARGINAL_FEE = 5000
ZIP_317_GRACE_ACTIONS = 2

@dataclass
class TxIn:
    script_sig: bytes

    def size(self):
        return (
            32 + # prevout.hash
            4  + # prevout.n
            script_size(self.script_sig) +
            4    # nSequence
        )

@dataclass
class TxOut:
    value: int
    script_pubkey: bytes

    def size(self):
        return (
            8 + # nValue
            script_size(self.script_pubkey)
        )

    def compact_size(self):
        return (
            p_uint64(1, self.value) +
            p_bytes(2, len(self.script_pubkey))
        )

@dataclass
class Tx:
    version: int
    vin: list[TxIn] = field(default_factory=list)
    vout: list[TxOut] = field(default_factory=list)
    sprout_joinsplits: int = 0
    sapling_spends: int = 0
    sapling_outputs: int = 0
    orchard_actions: int = 0

    def has_sprout(self):
        return self.sprout_joinsplits > 0

    def has_sapling(self):
        return (self.sapling_spends + self.sapling_outputs) > 0

    def has_orchard(self):
        return self.orchard_actions > 0

    def zip317_fee(self):
        """
        Returns the ZIP 317 fee that a transaction of this shape would need to pay.
        """
        logical_actions = (
            max(
                ceil(sum([txin.size() for txin in self.vin]) / 150),
                ceil(sum([txout.size() for txout in self.vout]) / 34),
            ) +
            2 * self.sprout_joinsplits +
            max(self.sapling_spends, self.sapling_outputs) +
            self.orchard_actions
        )
        return ZIP_317_MARGINAL_FEE * max(ZIP_317_GRACE_ACTIONS, logical_actions)

    def size(self):
        """
        Returns the size of this transaction's network encoding.
        """
        if self.version == 5:
            assert self.sprout_joinsplits == 0
            return self.v5_size()
        else:
            assert self.version == 4
            assert self.orchard_actions == 0
            return self.v4_size()

    def v4_size(self):
        return (
            4 + # header
            4 + # nVersionGroupId
            compact_size_size(len(self.vin)) +
            sum([tx_in.size() for tx_in in self.vin]) +
            compact_size_size(len(self.vout)) +
            sum([tx_out.size() for tx_out in self.vout]) +
            4 + # lock_time
            4 + # nExpiryHeight
            8 + # valueBalance
            compact_size_size(self.sapling_spends) +
            (384 * self.sapling_spends) +
            compact_size_size(self.sapling_outputs) +
            (948 * self.sapling_outputs) +
            compact_size_size(self.sprout_joinsplits) +
            (1698 * self.sprout_joinsplits) +
            ((32 + 64) if self.has_sprout() else 0) + # joinSplitPubKey + joinSplitSig
            (64 if self.has_sapling() else 0) # bindingSig
        )

    def v5_size(self):
        has_sapling_spends = self.sapling_spends > 0
        proofs_orchard = 2720 + 2272 * self.orchard_actions
        return (
            # Common Transaction Fields
            4 + # header
            4 + # nVersionGroupId
            4 + # nConsensusBranchId
            4 + # lock_time
            4 + # nExpiryHeight

            # Transparent Transaction Fields
            compact_size_size(len(self.vin)) +
            sum([tx_in.size() for tx_in in self.vin]) +
            compact_size_size(len(self.vout)) +
            sum([tx_out.size() for tx_out in self.vout]) +

            # Sapling Transaction Fields
            compact_size_size(self.sapling_spends)  + # nSpendsSapling
            (96 * self.sapling_spends) +              # vSpendsSapling
            compact_size_size(self.sapling_outputs) + # nOutputsSapling
            (756 * self.sapling_outputs) +            # vOutputsSapling
            (8 if self.has_sapling() else 0) +   # valueBalanceSapling
            (32 if has_sapling_spends else 0) +  # anchorSapling
            ((192 + 64) * self.sapling_spends) + # vSpendProofsSapling + vSpendAuthSigsSapling
            (192 * self.sapling_outputs) +       # vOutputProofsSapling
            (64 if self.has_sapling() else 0) +  # bindingSigSapling

            # Orchard Transaction Fields
            compact_size_size(self.orchard_actions) +     # nActionsOrchard
            (820 * self.orchard_actions) +                # vActionsOrchard
            ((1 + 8 + 32) if self.has_orchard() else 0) + # flagsOrchard + valueBalanceOrchard + anchorOrchard
            (compact_size_size(proofs_orchard) + proofs_orchard if self.has_orchard() else 0) + # sizeProofsOrchard + proofsOrchard
            (64 * self.orchard_actions) +     # vSpendAuthSigsOrchard
            (64 if self.has_orchard() else 0) # bindingSigOrchard
        )
