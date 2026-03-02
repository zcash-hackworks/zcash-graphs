# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.2",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Zcash Performance Simulation

    This notebook simulates the performance of the Zcash chain.

    There are currently four protocols in Zcash consensus:
    - Transparent
    - Sprout
    - Sapling
    - Orchard
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Constants
    """)
    return


@app.cell
def _():
    # This notebook is in a subdirectory. Ensure we can import `zcash_graphs`.
    import sys
    sys.path.append('..')
    return


@app.cell
def _():
    MAX_BLOCK_SIZE = 2000000
    TARGET_BLOCK_SPACING = 75

    CURRENT_HEIGHT = 3256000
    CURRENT_TIME = 1772264000
    CURRENT_SAPLING_TREE_SIZE = 73900000
    CURRENT_ORCHARD_TREE_SIZE = 49800000

    COMPACT_NOTE_CIPHERTEXT_SIZE = 1 + 11 + 32 + 8
    # COMPACT_NOTE_CIPHERTEXT_SIZE = COMPACT_NOTE_CIPHERTEXT_SIZE + 32 # with assetbase
    # COMPACT_NOTE_CIPHERTEXT_SIZE = COMPACT_NOTE_CIPHERTEXT_SIZE + 32 # with memo key
    # COMPACT_NOTE_CIPHERTEXT_SIZE = COMPACT_NOTE_CIPHERTEXT_SIZE + 16 # with auth tag

    COINBASE_SCRIPT_SIG = b'\x00' * 5
    P2PKH_SCRIPT_PUBKEY = b'\x00' * 34
    P2SH_SCRIPT_PUBKEY = b'\x00' * 26
    return (
        COINBASE_SCRIPT_SIG,
        COMPACT_NOTE_CIPHERTEXT_SIZE,
        CURRENT_HEIGHT,
        CURRENT_ORCHARD_TREE_SIZE,
        CURRENT_SAPLING_TREE_SIZE,
        CURRENT_TIME,
        MAX_BLOCK_SIZE,
        P2PKH_SCRIPT_PUBKEY,
        P2SH_SCRIPT_PUBKEY,
        TARGET_BLOCK_SPACING,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Verification times
    """)
    return


@app.cell
def _():
    import os
    import subprocess

    def collect_times():
        # Find bench_bitcoin binary
        base_dir = os.getcwd()
        bench_bitcoin = os.path.join(base_dir, 'bench_bitcoin')

        # Run bench_bitcoin binary
        try:
            result = subprocess.run([bench_bitcoin], stdout=subprocess.PIPE, universal_newlines=True)
            result.check_returncode()
            result = result.stdout
        except AttributeError:
            # Use the older API
            result = subprocess.check_output([bench_bitcoin], universal_newlines=True)

        # Collect benchmarks
        benchmarks = {}
        for row in result.strip().split('\n')[1:]: # Skip the headings
            parts = row.split(',')
            benchmarks[parts[0]] = int(parts[2])

        return {
            'groth16-proof': benchmarks['SaplingOutput'],
            'ecdsa': benchmarks['ECDSA'],
            'redjubjub': benchmarks['SaplingSpend'] - benchmarks['SaplingOutput'],
            'ed25519': benchmarks['JoinSplitSig'],
        }

    print('Collecting benchmarks...')
    times = collect_times()
    print('Times (ns):', times)
    return (times,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Size calculations

    Transactions on chain use an encoding scheme evolved from Bitcoin Core.

    The light client protocol by contrast uses gRPC with proto3 encodings, and is based around `CompactBlock`s. All of the authenticating data is stripped out, as well as any of the effecting data that is not strictly necessary for detecting notes and making them spendable.
    """)
    return


@app.cell
def _(
    COMPACT_NOTE_CIPHERTEXT_SIZE,
    CURRENT_HEIGHT,
    CURRENT_ORCHARD_TREE_SIZE,
    CURRENT_SAPLING_TREE_SIZE,
    CURRENT_TIME,
):
    from zcash_graphs.size import BLOCK_HEADER_SIZE
    from zcash_graphs.size.proto3 import (
        p_bytes,
        p_nested,
        p_repeated,
        p_repeated_of,
        p_uint32,
        p_uint64,
    )
    from zcash_graphs.size.tx import Tx

    compact_sapling_spend_size = p_bytes(1, 32) # nf
    compact_sapling_output_size = (
        p_bytes(1, 32) + # cmu
        p_bytes(2, 32) + # ephemeralKey
        p_bytes(3, COMPACT_NOTE_CIPHERTEXT_SIZE) # ciphertext
    )

    compact_orchard_action_size = (
        p_bytes(1, 32) + # nullifier
        p_bytes(2, 32) + # cmx
        p_bytes(3, 32) + # ephemeralKey
        p_bytes(4, COMPACT_NOTE_CIPHERTEXT_SIZE) # ciphertext
    )

    def compact_tx_in_size(max_transparent_outputs):
        return (
            p_bytes(1, 32) + # prevoutTxid
            p_uint32(2, max_transparent_outputs - 1) # prevoutIndex
        )

    def compact_tx_size(max_transparent_outputs):
        def helper(i, tx: Tx):
            return (
                p_uint64(1, i) + # index
                p_bytes(2, 32) + # txid
                p_uint32(3, tx.zip317_fee()) + # fee
                p_repeated(4, compact_sapling_spend_size, tx.sapling_spends) +
                p_repeated(5, compact_sapling_output_size, tx.sapling_outputs) +
                p_repeated(6, compact_orchard_action_size, tx.orchard_actions) +
                p_repeated(7, compact_tx_in_size(max_transparent_outputs), len(tx.vin)) +
                p_repeated_of(8, lambda _, tx_out: tx_out.size(), tx.vout)
            )
        return helper

    chain_metadata_size = (
        p_uint32(1, CURRENT_SAPLING_TREE_SIZE) + # saplingCommitmentTreeSize
        p_uint32(2, CURRENT_ORCHARD_TREE_SIZE) # orchardCommitmentTreeSize
    )

    def compact_block_size(vtx, max_transparent_outputs, header=False):
        return (
            p_uint32(1, 1) + # protoVersion
            p_uint64(2, CURRENT_HEIGHT) + # height
            p_bytes(3, 32) + # hash
            (p_bytes(6, BLOCK_HEADER_SIZE) if header else (
                p_bytes(4, 32) + # prevHash
                p_uint32(5, CURRENT_TIME) # time
            )) +
            p_repeated_of(7, compact_tx_size(max_transparent_outputs), vtx) +
            p_nested(8, chain_metadata_size) # chainMetadata
        )


    return Tx, compact_block_size, compact_tx_size


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulated transactions

    The things we want to vary within the simulation are the parts of the transaction that can be adjusted by adversaries.
    """)
    return


@app.cell
def _(Tx, compact_tx_size):
    sapling_spam_tx_v4 = Tx(version=4, sapling_spends=1, sapling_outputs=32)
    print(sapling_spam_tx_v4.size())
    print(compact_tx_size(16)(1, sapling_spam_tx_v4))
    return (sapling_spam_tx_v4,)


@app.cell
def _(Tx, compact_tx_size):
    orchard_spam_tx = Tx(version=5, orchard_actions=32)
    print(orchard_spam_tx.size())
    print(compact_tx_size(16)(1, orchard_spam_tx))
    return (orchard_spam_tx,)


@app.cell
def _(
    COINBASE_SCRIPT_SIG,
    MAX_BLOCK_SIZE,
    P2PKH_SCRIPT_PUBKEY,
    P2SH_SCRIPT_PUBKEY,
    Tx,
):
    from zcash_graphs.size import block_size
    from zcash_graphs.size.tx import TxIn, TxOut

    typical_coinbase_tx = Tx(
        version=4,
        vin=[TxIn(script_sig=COINBASE_SCRIPT_SIG)],
        vout=[
            TxOut(125105000, P2PKH_SCRIPT_PUBKEY),
            TxOut(12500000, P2SH_SCRIPT_PUBKEY),
        ],
    )

    def with_typical_coinbase(vtx):
        return [typical_coinbase_tx] + vtx

    def worst_case_many_identical_txs(tx):
        vtx = []
        while True:
            vtx.append(tx)
            if block_size(with_typical_coinbase(vtx)) > MAX_BLOCK_SIZE:
                # Keep under the size limit
                vtx.pop()
                break
        return vtx

    def worst_case_one_tx_containing(field: str, version=4):
        if field == 'orchard_actions':
            version = 5
        vtx = [Tx(version)]
        while True:
            setattr(vtx[0], field, getattr(vtx[0], field) + 1)
            if block_size(with_typical_coinbase(vtx)) > MAX_BLOCK_SIZE:
                # Keep under the size limit
                setattr(vtx[0], field, getattr(vtx[0], field) - 1)
                break
        return vtx


    return (
        block_size,
        worst_case_many_identical_txs,
        worst_case_one_tx_containing,
    )


@app.cell
def _(TARGET_BLOCK_SPACING, Tx, block_size, compact_block_size):
    from dataclasses import dataclass

    @dataclass
    class Costs:
        tx_count: int
        sapling_tx_count: int
        orchard_tx_count: int
        block_size_bytes: int
        compact_block_size_bytes: int
        groth16_proofs: int
        ecdsa_sigs: int
        ed25519_sigs: int
        redjubjub_sigs: int
        redpallas_sigs: int

        def __init__(self, vtx: list[Tx]):
            """
            Determines the costs to the network if every block contained the given pattern of transactions.
            """
            self.tx_count = len(vtx)
            self.sapling_tx_count = sum([1 if tx.has_sapling() else 0 for tx in vtx])
            self.orchard_tx_count = sum([1 if tx.has_orchard() else 0 for tx in vtx])

            self.block_size_bytes = block_size(vtx)

            max_transparent_outputs = max([len(tx.vout) for tx in vtx])
            self.compact_block_size_bytes = compact_block_size(vtx, max_transparent_outputs)

            # One Groth16 proof per Sapling spend, Sapling output, and JoinSplit
            self.groth16_proofs = sum([tx.sapling_spends + tx.sapling_outputs + tx.sprout_joinsplits for tx in vtx])

            # One ECDSA signature per transparent input
            self.ecdsa_sigs = sum([len(tx.vin) for tx in vtx])

            # One Ed25519 signature per transaction that contains JoinSplits
            self.ed25519_sigs = sum([1 if tx.sprout_joinsplits > 0 else 0 for tx in vtx])

            # One RedJubjub signature per Sapling spend (spendAuthSig) and per transaction (bindingSig)
            self.redjubjub_sigs = sum([tx.sapling_spends + (
                1 if tx.sapling_spends + tx.sapling_outputs > 0 else 0) for tx in vtx])

            # One RedPallas signature per Orchard action (spendAuthSig) and per transaction (bindingSig)
            self.redpallas_sigs = sum([tx.orchard_actions + (
                1 if tx.orchard_actions > 0 else 0) for tx in vtx])

        def chain_usage(self):
            blocks_per_day = (24 * 60 * 60) // TARGET_BLOCK_SPACING

        def tps(self):
            return self.tx_count / TARGET_BLOCK_SPACING

        def sapling_tps(self):
            return self.sapling_tx_count / TARGET_BLOCK_SPACING

        def orchard_tps(self):
            return self.orchard_tx_count / TARGET_BLOCK_SPACING

        def print_makeup(self, times):
            print('- Block size:             ', self.block_size_bytes, 'bytes')
            print('- CompactBlock size:       {} bytes ({:.2%})'.format(self.compact_block_size_bytes, self.compact_block_size_bytes / self.block_size_bytes))
            print('- Transactions:           ', self.tx_count)
            print('- TPS:                     %0.2f' % self.tps())
            print('  - Sapling TPS:           %0.2f' % self.sapling_tps())
            print('  - Orchard TPS:           %0.2f' % self.orchard_tps())
            print('- Groth16 proofs:         ', self.groth16_proofs)
            print('- ECDSA signatures:       ', self.ecdsa_sigs)
            print('- Ed25519 signatures:     ', self.ed25519_sigs)
            print('- RedJubjub signatures:   ', self.redjubjub_sigs)
            print('- RedPallas signatures:   ', self.redpallas_sigs)
            print('- Unbatched verification:  %0.2f seconds' % (float(
                (times['groth16-proof'] * self.groth16_proofs) +
                (times['ecdsa'] * self.ecdsa_sigs) +
                (times['ed25519'] * self.ed25519_sigs) +
                (times['redjubjub'] * self.redjubjub_sigs) +
                (times['redjubjub'] * self.redpallas_sigs)
            ) / 10**9))


    return (Costs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Typical-case scenarios
    """)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    typical_v5_sapling_txs = worst_case_many_identical_txs(Tx(version=5, sapling_spends=1, sapling_outputs=2))
    print('Blocks full of typical v5 Sapling transactions:')
    Costs(typical_v5_sapling_txs).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    typical_v5_orchard_txs = worst_case_many_identical_txs(Tx(version=5, orchard_actions=2))
    print('Blocks full of typical v5 Orchard transactions:')
    Costs(typical_v5_orchard_txs).print_makeup(times)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Worst-case scenarios
    """)
    return


@app.cell
def _(Costs, sapling_spam_tx_v4, times, worst_case_many_identical_txs):
    sapling_spam_txs = worst_case_many_identical_txs(sapling_spam_tx_v4)
    print('Blocks full of Sapling spam transaction:')
    Costs(sapling_spam_txs).print_makeup(times)
    return


@app.cell
def _(Costs, orchard_spam_tx, times, worst_case_many_identical_txs):
    orchard_spam_txs = worst_case_many_identical_txs(orchard_spam_tx)
    print('Blocks full of Orchard spam transaction:')
    Costs(orchard_spam_txs).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_sapling_spend_per_v4_tx = worst_case_many_identical_txs(Tx(version=4, sapling_spends=1))
    print('One Sapling spend per v4 transaction:')
    Costs(one_sapling_spend_per_v4_tx).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_sapling_spend_per_v5_tx = worst_case_many_identical_txs(Tx(version=5, sapling_spends=1))
    print('One Sapling spend per v5 transaction:')
    Costs(one_sapling_spend_per_v5_tx).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_v4_tx_containing_sapling_spends = worst_case_one_tx_containing('sapling_spends')
    print('One v4 transaction containing Sapling spends:')
    Costs(one_v4_tx_containing_sapling_spends).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_v5_tx_containing_sapling_spends = worst_case_one_tx_containing('sapling_spends', version=5)
    print('One v5 transaction containing Sapling spends:')
    Costs(one_v5_tx_containing_sapling_spends).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_sapling_output_per_v4_tx = worst_case_many_identical_txs(Tx(version=4, sapling_outputs=1))
    print('One Sapling output per v4 transaction:')
    Costs(one_sapling_output_per_v4_tx).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_sapling_output_per_v5_tx = worst_case_many_identical_txs(Tx(version=5, sapling_outputs=1))
    print('One Sapling output per v5 transaction:')
    Costs(one_sapling_output_per_v5_tx).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_v4_tx_containing_sapling_outputs = worst_case_one_tx_containing('sapling_outputs')
    print('One v4 transaction containing Sapling outputs:')
    Costs(one_v4_tx_containing_sapling_outputs).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_v5_tx_containing_sapling_outputs = worst_case_one_tx_containing('sapling_outputs', version=5)
    print('One v5 transaction containing Sapling outputs:')
    Costs(one_v5_tx_containing_sapling_outputs).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_orchard_action_per_v5_tx = worst_case_many_identical_txs(Tx(version=5, orchard_actions=1))
    print('One Orchard action per v5 transaction:')
    Costs(one_orchard_action_per_v5_tx).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_v5_tx_containing_orchard_actions = worst_case_one_tx_containing('orchard_actions')
    print('One v5 transaction containing Orchard actions:')
    Costs(one_v5_tx_containing_orchard_actions).print_makeup(times)
    return


@app.cell
def _(Costs, Tx, times, worst_case_many_identical_txs):
    one_joinsplit_per_tx = worst_case_many_identical_txs(Tx(version=4, sprout_joinsplits=1))
    print('One JoinSplit per transaction:')
    Costs(one_joinsplit_per_tx).print_makeup(times)
    return


@app.cell
def _(Costs, times, worst_case_one_tx_containing):
    one_tx_containing_joinsplits = worst_case_one_tx_containing('sprout_joinsplits')
    print('One transaction containing JoinSplits:')
    Costs(one_tx_containing_joinsplits).print_makeup(times)
    return


if __name__ == "__main__":
    app.run()
