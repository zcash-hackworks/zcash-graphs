#!/usr/bin/env python3
# Copyright (c) 2022 The Zcash developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php .
"""Simple Transaction Analysis

This contains a class, `Analyzer`, for defining analyses of the blocks and
transactions on the blockchain. It also exposes a function
`analyze_blocks`, which handles applying multiple analyses simultaneously over
some common range of blocks.
"""

import datetime
import itertools
import math
import numpy as np
from statistics import mean
import sys

from analyze import Analysis, Analyzer
from helpers import *

### TODO: Get host/port from config
if len(sys.argv) > 1:
    connection_string = sys.argv[1]
else:
    raise Exception(
        "%s needs to be provided a connection string, like \"http://user:pass@localhost:port\"."
        % (sys.argv[0],))

analyzer = Analyzer(connection_string)

### Requested Statistics

def storeAnchor(pool, cache, block):
    """
    Caches the block height as the value for its anchor hash.
    """
    try:
        final_root = block[pool]
        try:
            cache[final_root]
        except KeyError:
            cache[final_root] = block['height']
    except KeyError:
        None

    return cache

# "how old of anchors are people picking"
# --- https://zcash.slack.com/archives/CP6SKNCJK/p1660103126252979
anchor_age_orchard = Analysis(
    "how old of anchors are people picking (for orchard)",
    lambda _c, _b, tx: is_orchard_tx(tx),
    [(orchard_anchorage, sum)],
    lambda *_: 1,
    ({}, lambda c, b: storeAnchor('finalorchardroot', c, b)),
    blocks_per_hour * 24
)

anchor_age_sapling = Analysis(
    "how old of anchors are people picking (for sapling)",
    lambda _c, _b, tx: is_saplingspend_tx(tx),
    [(sapling_anchorage, sum)],
    lambda *_: 1,
    ({}, lambda c, b: storeAnchor('finalsaplingroot', c, b)),
    blocks_per_hour * 24
)

# "what's the distribution of expiry height deltas"
# --- https://zcash.slack.com/archives/CP6SKNCJK/p1660103126252979
expiry_height_deltas = Analysis(
    "distribution of expiry height deltas",
    lambda *_: True,
    [(lambda _, b, t: expiry_height_delta(b, t), sum)],
    lambda *_: 1
)

tx_type_with_long_expiry = Analysis(
    "types of tx with expiries longer than about a month",
    lambda _, b, t: expiry_height_delta(b, t) >= blocks_per_hour * 24 * 30,
    [# (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
     #  identity),
     (lambda _c, _b, tx: tx_type(tx), sum)],
    lambda *_: 1
)

# "does anyone use locktime"
# --- https://zcash.slack.com/archives/CP6SKNCJK/p1660103126252979
locktime_usage = Analysis(
    "proportion of tx using locktime",
    lambda *_: True,
    [(lambda *_: 1,
      lambda d: dict(d)[True] / (dict(d)[False] + dict(d)[True])),
     (lambda _c, _b, tx: tx['locktime'] != 0, sum)],
    lambda *_: 1
)

# "I'm seeing a slightly different pattern to the sandblasting transactions,
#  unless I've just missed this before. The transactions I've looked at recently
#  have had > 400 sapling outputs. Has this been the case before and I just
#  missed it? I thought primarily these transactions had slightly over 100
#  outputs in most cases."
# --- https://zcash.slack.com/archives/CP6SKNCJK/p1660195664187769


# "Calculate the POFM threshold for historical transactions on-chain and
#  calculate what proportion of those transactions would fall below the POFM
#  threshold"
# --- https://docs.google.com/document/d/18wtGFCB2N4FO7SoqDPnEgVudAMlCArHMz0EwhE1HNPY/edit
tx_below_pofm_threshold = Analysis(
    "rate of transactions below POFM threshold",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: dict(d)[False] / (dict(d)[False] + dict(d)[True])),
      (lambda _c, _b, tx: count_ins_and_outs(tx) - 4 > 0, sum)
    ],
    lambda *_: 1
)

tx_below_pofm_threshold_abs = Analysis(
    "transactions below POFM threshold",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: (dict(d)[False], dict(d)[True])),
      (lambda _c, _b, tx: count_ins_and_outs(tx) - 4 > 0, sum)
    ],
    lambda *_: 1
)

outs_below_pofm_threshold_abs = Analysis(
    "outputs below POFM threshold",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: (dict(d)[False], dict(d)[True])),
      (lambda _c, _b, tx: count_ins_and_outs(tx) - 4 > 0, sum)
    ],
    lambda _c, _b, tx: count_outputs(tx)
)

tx_below_pofm_threshold_5 = Analysis(
    "rate of transactions below POFM threshold with a grace window of 5",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: dict(d)[False] / (dict(d)[False] + dict(d)[True])),
      (lambda _c, _b, tx: count_ins_and_outs(tx) - 5 > 0, sum)
    ],
    lambda *_: 1
)


tx_below_pofm_threshold_max = Analysis(
    "rate of transactions below POFM threshold with max",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: dict(d)[False] / (dict(d)[False] + dict(d)[True])),
      (lambda _c, _b, tx: count_actions(tx) - 4 > 0, sum)
    ],
    lambda *_: 1
)

tx_below_pofm_threshold_ins = Analysis(
    "rate of transactions below POFM threshold only on inputs",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
       lambda d: dict(d)[False] / (dict(d)[False] + dict(d)[True])),
      (lambda _c, _b, tx: count_inputs(tx) - 4 > 0, sum)
    ],
    lambda *_: 1
)

### Other Examples

tx_per_day = Analysis(
    "count transactions per day (treating block 0 as midnight ZST)",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), sum)],
    lambda *_: 1
)

mean_tx_per_day = Analysis(
    "mean transactions per day, by block",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] % (blocks_per_hour * 24)), lambda d: mean([x[1] for x in d])),
     (lambda _c, block, _t: int(block['height']/(blocks_per_hour * 24)), sum)
    ],
    lambda *_: 1
)

mean_inout_per_tx_per_day = Analysis(
    "mean inputs, outputs per transaction per day, by block",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] % (blocks_per_hour * 24)), lambda d: mean(itertools.chain(d.values()))),
     (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity)
    ],
    lambda _c, _b, tx: (count_inputs(tx), count_outputs(tx))
)

mean_inout_per_tx = Analysis(
    "mean inputs, outputs per transaction, by week",
    lambda *_: True,
    [ ( lambda _c, block, _t: int(block['height']/(blocks_per_hour * 24 * 7)),
        lambda d: (mean([x[0] for x in d]), mean([x[1] for x in d]))
       )
     ],
    lambda _c, _b, tx: (count_inputs(tx), count_outputs(tx))
)

minimum_pofm_fees_nuttycom = Analysis(
    "distribution of fees in ZAT, by day, using nuttycom's pricing",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: math.ceil(math.log((1000 + 250 * max(0, count_ins_and_outs(tx) - 4)) / 1000, 2)), sum)
    ],
    lambda *_: 1
)

minimum_pofm_fees_nuttycom = Analysis(
    "distribution of fees in ZAT, by day, using nuttycom's pricing",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: math.ceil(math.log((1000 + 250 * max(0, count_ins_and_outs(tx) - 4)) / 1000, 2)), sum)
    ],
    lambda *_: 1
)

minimum_pofm_fees_nuttycom2 = Analysis(
    "distribution of fees in ZAT, by day, using nuttycom's changed pricing",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: math.ceil(math.log((1000 + 200 * max(0, count_ins_and_outs(tx) - 5)) / 1000, 2)), sum)
    ],
    lambda *_: 1
)

def meh_fees(tx):
    fee = tx['feePaid']
    if fee == 0:
        return -1
    else:
        result = math.ceil(math.log(tx['feePaid'], 2))
        return result

actual_fees = Analysis(
    "actual fees",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, _b, tx: meh_fees(tx), sum)
    ],
    lambda *_: 1
)

proposed_fees = Analysis(
    "",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, _b, tx: math.ceil(math.log(5000 * max(2, count_actions(tx)), 2)), sum)
    ],
    lambda *_: 1
)

arity_heat_map = Analysis(
    "inputs vs outputs",
    lambda *_: True,
    [(lambda _c, _b, tx: min(100, count_outputs(tx)), identity),
     (lambda _c, _b, tx: min(100, count_inputs(tx)), sum)],
    lambda *_: 1
)

input_size_dist = Analysis(
    "distribution of input sizes",
    lambda *_: True,
    [(lambda _c, _b, tx: [len(x['scriptSig']['hex']) for x in tx['vin']], identity)],
    lambda *_: 1,
)

# very_high_inout_tx = Analysis(
#     "tx with very high in/out counts",
#     lambda _c, _b, tx: count_ins_and_outs(tx) > 100,
#     [(lambda _c, _b, tx: (count_inputs(tx), count_outputs(tx)), identity)],
#     lambda _c, _b, tx: tx['txid']
# )

very_high_inout_tx = Analysis(
    "tx with very high in/out counts",
    lambda _c, _b, tx: count_ins_and_outs(tx) > 5000,
    [],
    lambda _c, _b, tx: (tx['txid'], count_ins_and_outs(tx))
)

def track_utxos(cache, block):
    for tx in block[tx]:
        for vin in tx['vin']:
            del cache[(vin['txid'], vin['vout'])]
        for vout in tx['vout']:
            cache[(tx['txid'], vout['n'])] = vout['valueZat']
    return cache

utxo_distribution = Analysis(
    "how many UTXOs and how big are they?",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)),
      lambda caches: sum([caches[-1][k] for k in caches[-1]]))],
    lambda cache, _b, _t: cache,
    ({}, track_utxos),
    1_000_000_000 # back to block 0, TODO: should be able to say this explicitly
)

def is_sandblasting(tx):
    return get_shielded_outputs(tx) > 300

sandblasters_per_day = Analysis(
    "how many transactions have >300 Sapling outputs each day?",
    lambda _c, _b, tx: is_sandblasting(tx),
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), sum)],
    lambda *_: 1
)

sandblasters_and_more_per_day = Analysis(
    "how many transactions have >300 outputs each day?",
    lambda _c, _b, tx: count_outputs(tx) > 300,
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), sum)],
    lambda *_: 1
)

sandblaster_average_outputs_per_day = Analysis(
    "how many outputs do sandblasters have?",
    lambda _c, _b, tx: is_sandblasting(tx),
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), (lambda txs: sum(txs) / len(txs)))],
    lambda _c, _b, tx: count_outputs(tx)
)

nuttycom_fees_vs_actual = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using nuttycom's pricing",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(250 * max(4, count_ins_and_outs(tx)), tx), sum)
    ],
    lambda *_: 1
)

action_fees_vs_actual = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(500 * max(3, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

nuttycom_fees_vs_actual_trans = Analysis(
    "transparent transactions that wouldn't pay more under the new model, by day, using nuttycom's pricing",
    lambda _c, _b, tx: tx_type(tx) == 't-t',
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(250 * max(4, count_ins_and_outs(tx)), tx), sum)
    ],
    lambda *_: 1
)

action_fees_vs_actual_trans = Analysis(
    "transparent transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: tx_type(tx) == 't-t',
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(500 * max(3, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

greg_fees_vs_actual = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(250 * max(4, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

greg_fees_vs_actual_trans = Analysis(
    "transparent transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: tx_type(tx) == 't-t',
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(250 * max(4, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

latest_fees_vs_actual = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(1000 * max(2, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

latest_fees_vs_actual_trans = Analysis(
    "transparent transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: tx_type(tx) == 't-t',
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(1000 * max(2, count_actions(tx)), tx), sum)
    ],
    lambda *_: 1
)

flat_fees_vs_actual = Analysis(
    "transactions that would pass the original 10k ZAT fee, by day",
    lambda *_: True,
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(10_000, tx), sum)
    ],
    lambda *_: 1
)

shielding_tx_heat_map = Analysis(
    "shielding tx",
    lambda _c, _b, tx: is_not_coinbase(tx) and (tx_type(tx) == 't-z' or tx_type(tx) == 'm-z'),
    [(lambda _c, _b, tx: min(100, count_outputs(tx)), identity),
     (lambda _c, _b, tx: min(100, count_inputs(tx)), sum)],
    lambda *_: 1
)

shielding_tx_actions = Analysis(
    "shielding tx",
    lambda _c, _b, tx: is_not_coinbase(tx) and (tx_type(tx) == 't-z' or tx_type(tx) == 'm-z'),
    [(lambda _c, _b, tx: min(100, count_actions(tx)), sum)],
    lambda *_: 1
)

fees_from_sandblasting = Analysis(
    "fees collected from sandblasting",
    lambda _c, _b, tx: is_sandblasting(tx),
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), sum)],
    lambda _c, _b, tx: 500 * max(2, count_actions(tx))
)

flat_fees_vs_actual_trans = Analysis(
    "transparent transactions that would pass the original 10k ZAT fee, by day",
    lambda _c, _b, tx: tx_type(tx) == 't-t',
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: getFeeDiff(10_000, tx), sum)
    ],
    lambda *_: 1
)

transparent_tx_that_would_fail_heat_map = Analysis(
    "heat map of transparent tx that would fail under `500 * max(3, |actions|)`",
    lambda _c, _b, tx: tx_type(tx) == 't-t' and getFeeDiff(500 * max(3, count_actions(tx)), tx) == False,
    [(lambda _c, _b, tx: min(100, count_outputs(tx)), identity),
     (lambda _c, _b, tx: min(100, count_inputs(tx)), sum)],
    lambda *_: 1
)

historical_fees = Analysis(
    "histogram of actual fees paid",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [(lambda _c, _b, tx: check_fee_paid(tx), sum)],
    lambda *_: 1
)

arity_heat_map = Analysis(
    "inputs vs outputs",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [(lambda _c, _b, tx: min(100, count_outputs(tx)), identity),
     (lambda _c, _b, tx: min(100, count_inputs(tx)), sum)],
    lambda *_: 1
)

transparent_input_histogram = Analysis(
    "how many transparent inputs do txs have?",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: min(100, len(tx['vin'])), sum)],
    lambda *_: 1
)

nuttycom_fees_vs_10k = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using nuttycom's pricing",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 250 * max(4, count_ins_and_outs(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)

action_fees_vs_10k = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 500 * max(3, count_actions(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)

latest_fees_vs_10k = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: is_not_coinbase(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 1000 * max(2, count_actions(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)


def make_weekly_range(starting_week, number_of_weeks):
    start_of_range = blocks_per_hour * 24 * 7 * starting_week
    end_of_range = start_of_range + (blocks_per_hour * 24 * 7 * number_of_weeks)
    return range(start_of_range, end_of_range)


# start about a month before sandblasting, overlapping with it
pre_sandblasting_range = make_weekly_range(206, 12)

# well into sandblasting
recent_range = make_weekly_range(220, 1)

start = datetime.datetime.now()
for analysis in analyzer.analyze_blocks(pre_sandblasting_range,
                       [ # sandblaster_average_outputs_per_day,
                         # flat_fees_vs_actual,
                         # flat_fees_vs_actual_trans,
                         # transparent_tx_that_would_fail_heat_map
                         nuttycom_fees_vs_actual,
                         action_fees_vs_actual,
                         nuttycom_fees_vs_actual_trans,
                         action_fees_vs_actual_trans,
                         greg_fees_vs_actual,
                         greg_fees_vs_actual_trans,
                         # historical_fees,
                         # transparent_input_histogram,
                       ]):
    print(analysis)
print(datetime.datetime.now() - start)

# rerunning old data …
start = datetime.datetime.now()
for analysis in analyzer.analyze_blocks(make_weekly_range(206, 1),
                       [ actual_fees,
                         proposed_fees,
                       ]):
    print(analysis)
print(datetime.datetime.now() - start)

nuttycom_fees_vs_10k2 = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using nuttycom's pricing",
    lambda _c, _b, tx: is_not_coinbase(tx) and not is_sandblasting(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 250 * max(4, count_ins_and_outs(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)

action_fees_vs_10k2 = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: is_not_coinbase(tx) and not is_sandblasting(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 500 * max(3, count_actions(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)

latest_fees_vs_10k2 = Analysis(
    "transactions that wouldn't pay more under the new model, by day, using actions",
    lambda _c, _b, tx: is_not_coinbase(tx) and not is_sandblasting(tx),
    [ (lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), identity),
      (lambda _c, _b, tx: 1000 * max(2, count_actions(tx)) <= 10_000, sum)
    ],
    lambda *_: 1
)

pool_movement = Analysis(
    "how are funds moving between pools?",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), lambda vals: np.sum(np.array(vals), 0))],
    lambda _c, _b, tx: tx_pool_movement(tx)
)

start = datetime.datetime.now()
for analysis in analyzer.analyze_blocks(recent_range,
                       [ pool_movement
                       ]):
    print(analysis)
print(datetime.datetime.now() - start)

start = datetime.datetime.now()
for analysis in analyzer.analyze_blocks(pre_sandblasting_range,
                       [ tx_below_pofm_threshold,
                         tx_below_pofm_threshold_5,
                         tx_below_pofm_threshold_max,
                         tx_below_pofm_threshold_ins,
                         tx_below_pofm_threshold_abs,
                         outs_below_pofm_threshold_abs,
                         arity_heat_map,
                         minimum_pofm_fees_nuttycom,
                         minimum_pofm_fees_nuttycom2,
                       ]):
    print(analysis)
print(datetime.datetime.now() - start)
