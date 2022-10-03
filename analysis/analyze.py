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
import os.path
from progress.bar import IncrementalBar
from statistics import mean
import sys

sys.path.insert(
    1,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../qa/rpc-tests")
)

from test_framework.authproxy import AuthServiceProxy

### TODO: Get host/port from config
if len(sys.argv) > 1:
    connection_string = sys.argv[1]
else:
    raise Exception(
        "%s needs to be provided a connection string, like \"http://user:pass@localhost:port\"."
        % sys.argv[0])

class Analysis:
    """
    An analysis collects a single aggregated data structure from the blockchain.

    If you had a block and a single tx from that block, you could simply
   `my_analysis.aggregate(my_analysis.extract(block, tx))` to generate the stats
    for that analysis. However, since we generally want to aggregate across many
    transactions in many blocks and also because we usually want to collect
    multiple statistics at once (because re-fetching blocks and tx is slow),
   `extract` and `aggregate are separated out. See `analyze_blocks` for how to
    take advantage of this structure.
    """

    def __init__(self, name, tx_filter, bucketers, extractor, cache = ((), lambda c, _: c), preCache = 0):
        """It takes various functions to apply to the transactions therein. The functions are typed as follows:

    tx_filter :: cache -> Block -> Tx -> Boolean
    bucketers :: [ ...,
                   (cache -> Block -> Tx -> k_n-2, [(k_n-1, a)] -> b),
                   (cache -> Block -> Tx -> k_n-1, [(k_n, a)] -> b),
                   (cache -> Block -> Tx -> k_n,   [v] -> a)
                 ]
    extractor :: cache -> Block -> Tx -> v
    cache :: (cache, cache -> Block -> cache)
    preCache = Natural

    `tx_filter` decides whether the given transaction should be included in the
                result,
    `extractor` reduces each transaction to the parts we care about in the
                results,
    `bucketers` is a list of pairs of functions -- the first of each pair
                produces a key for bucketing the results and the second is how
                to accumulate the values in that bucket. The list allows us to
                create buckets of buckets.
    `cache`, if provided, is a tuple of an initial cache value and a function to
             update it so that later transactions can look at information from
             previous blocks.
    `preCache` is how many blocks before the start of our range to start
               caching. This is generally a _minimum_, don't be suprised if the
               cache is updated from some much earlier point. Also, it may be
               truncated if there aren't enough blocks between the beginning of
               the chain and and the start of the range.

    If no bucketers are provided, this returns a list of all the extracted data
    in a list, one for each transaction. If there are bucketers, it returns a
    map, with the keys from the first bucketer in the list and the values from
    the first accumulator in the list.

        """
        self.name = name
        self.__filter = tx_filter
        self.__bucketers = bucketers
        self.__extractor = extractor
        (self.__cache, self.__cacheUpdater) = cache
        self.preCache = preCache
        self.__lastCachedBlock = 0

    def updateCache(self, block):
        """
        This is exposed in order to handle the "precache", where we need to
        build up the cache for blocks before the blocks we actually care to have
        in our results.
        """
        if block['height'] > self.__lastCachedBlock:
            self.__cache = self.__cacheUpdater(self.__cache, block)
            self.__lastCachedBlock = block['height']

    def extract(self, block, tx):
        """
        Extracts all the data from a given transaction (and its block) needed to
        compute the statistics for this analysis.

        TODO: Allow a bucketer to return multiple keys. This hopefully allows
              things like sub-transaction extraction. E.g., looking at the sizes
              of all vouts by day, without caring which ones are in the same tx
        TODO: Distinguish between streamable and non-streamable analyses. The
              difference is that a streamable analysis has an outermost bucketer
              where duplicate keys are adjacent (much like POSIX `uniq`).
        """
        self.updateCache(block)

        if self.__filter(self.__cache, block, tx):
            value = self.__extractor(self.__cache, block, tx)
            keys = [x[0](self.__cache, block, tx) for x in self.__bucketers]
            return [(keys, value)]
        else:
            return []

    def aggregate(self, kvs):
        """
        Given a `[([k_0, k_1, ..., k_n-1], v)]` (where `n` is the length of the
        bucketer list provided at initialization and `k_*` are the results of
        each bucketer), this groups and accumulates the results, returning their
        final form.
        """
        kvs.sort(key=lambda x: x[0])
        return self.__group(kvs, [x[1] for x in self.__bucketers])

    def __group(self, kvs, accumulators):
        if accumulators:
            buck = []
            accum, *remaining_accum = accumulators
            for k, g in itertools.groupby(kvs, lambda x: x[0].pop(0)):
                buck.append((k, accum(self.__group(list(g), remaining_accum))))
            return buck
        else:
            return [x[1] for x in kvs]


class Analyzer:
    def __init__(self, node_url):
        self.node = AuthServiceProxy(node_url)

    def analyze_blocks(self, block_range, analyses):
        """
        This function executes multiple analyses over a common range of blocks,
        returning results keyed by the name of the analysis.
        """
        current_height = self.node.getblockchaininfo()['estimatedheight']
        bounded_range = range(
            max(0, min(block_range[0], current_height)),
            max(0, min(block_range[1], current_height))
        )
        longest_precache = max([x.preCache for x in analyses])
        data_start = bounded_range[0]
        for i in IncrementalBar('Building Cache   ').iter(range(max(0, data_start - longest_precache), data_start)):
            [x.updateCache(self.node.getblock(str(i), 2)) for x in analyses]

        bucketses = [(x, []) for x in analyses]
        for block_height in IncrementalBar('Processing Blocks').iter(block_range):
            block = self.node.getblock(str(block_height), 2)
            for tx in block['tx']:
                for analysis in analyses:
                    dict(bucketses)[analysis].extend(analysis.extract(block, tx))

        result = []
        for analysis in IncrementalBar('Running Analyses ').iter(analyses):
            result.append((analysis.name, analysis.aggregate(dict(bucketses)[analysis])))

        return result

### Helpers

def identity(x):
    return x

def get_shielded_spends(tx):
    try:
        shielded_spends = len(tx['vShieldedSpend'])
    except KeyError:
        shielded_spends = 0

    return shielded_spends

def get_shielded_outputs(tx):
    try:
        shielded_outputs = len(tx['vShieldedOutput'])
    except KeyError:
        shielded_outputs = 0

    return shielded_outputs

def get_orchard_actions(tx):
    try:
        orchard_actions = len(tx['orchard']['actions'])
    except KeyError:
        orchard_actions = 0

    return orchard_actions

def count_inputs(tx):
    return len(tx['vin']) + 2 * len(tx['vjoinsplit']) + get_shielded_spends(tx) + get_orchard_actions(tx)

def count_outputs(tx):
    return len(tx['vout']) + 2 * len(tx['vjoinsplit']) + get_shielded_outputs(tx) + get_orchard_actions(tx)

def count_ins_and_outs(tx):
    return (len(tx['vin'])
            + len(tx['vout'])
            + get_shielded_spends(tx)
            + get_shielded_outputs(tx)
            + 2 * len(tx['vjoinsplit'])
            + 2 * get_orchard_actions(tx))

def count_actions(tx):
    return (max(len(tx['vin']), len(tx['vout']))
            + max(get_shielded_spends(tx), get_shielded_outputs(tx))
            + 2 * len(tx['vjoinsplit'])
            + get_orchard_actions(tx))

def expiry_height_delta(block, tx):
    """
    Returns -1 if there's no expiry, also returns approximately 35,000 (the
    number of blocks in a month) if the expiry is beyond 1 month.
    """
    month = blocks_per_hour * 24 * 30
    try:
        expiry_height = tx['expiryheight']
        if expiry_height == 0:
            return -1
        elif tx['expiryheight'] - block['height'] > month:
            return month
        else:
            return tx['expiryheight'] - block['height']
    except KeyError:
        # `tx['expiryheight']` is ostensibly an optional field, but it seems
        # like `0` is what tends to be used for "don't expire", so this case
        # generally isn't hit.
        return -1

def tx_type(tx):
    """
    Categorizes all tx into one of nine categories: (t)ransparent, (z)shielded,
    or (m)ixed for both inputs and outputs. So some possible results are "t-t",
    "t-z", "m-z", etc.
    """
    if tx['vjoinsplit'] or get_shielded_spends(tx) != 0 or get_orchard_actions(tx) != 0:
        if tx['vin']:
            ins = "m"
        else:
            ins = "z"
    else:
        ins = "t"

    if tx['vjoinsplit'] or get_shielded_outputs(tx) != 0 or get_orchard_actions(tx) != 0:
        if tx['vout']:
            outs = "m"
        else:
            outs = "z"
    else:
        outs = "t"

    return ins + "-" + outs

def is_orchard_tx(tx):
    try:
        return tx['orchard']['actions']
    except KeyError:
        return False

def is_saplingspend_tx(tx):
    try:
        return tx['vShieldedSpend']
    except KeyError:
        return False

def orchard_anchorage(cache, block, tx):
    """
    Returns -1 if there is no anchor
    """
    try:
        return block['height'] - cache[tx['orchard']['anchor']]
    except KeyError:
        return -1

def sapling_anchorage(cache, block, tx):
    """
    Returns -1 if there is no anchor
    """
    try:
        return block['height'] - cache[tx['vShieldedSpend'][0]['anchor']]
    except KeyError:
        return -1

def is_not_coinbase(tx):
    return 'feePaid' in tx

# NB: This requires zcashd to be running with `experimentalfeatures=1`,
#    `txindex=1` and `insightexplorer=1`.
def getFeeDiff(proposedFee, tx):
    try:
        return proposedFee <= tx['feePaid']
    except KeyError:
        return -1

blocks_per_hour = 48 # half this before NU2?

# start about a month before sandblasting
start_range = blocks_per_hour * 24 * 7 * 206

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
        # if result < 0:
        #     print("negative result: %s, %s" % (fee, tx['txid']))
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


a = Analyzer(connection_string)

def make_weekly_range(starting_week, number_of_weeks):
    start_of_range = blocks_per_hour * 24 * 7 * starting_week
    end_of_range = start_of_range + (blocks_per_hour * 24 * 7 * number_of_weeks)
    return range(start_of_range, end_of_range)


# start about a month before sandblasting, overlapping with it

pre_sandblasting_range = make_weekly_range(206, 12)
recent_range = make_weekly_range(220, 1)

# start = datetime.datetime.now()
# for analysis in a.analyze_blocks(some_range,
#                        [ # sandblaster_average_outputs_per_day,
#                            # flat_fees_vs_actual,
#                            # flat_fees_vs_actual_trans,
#                            # transparent_tx_that_would_fail_heat_map
#                          nuttycom_fees_vs_actual,
#                          action_fees_vs_actual,
#                          nuttycom_fees_vs_actual_trans,
#                          action_fees_vs_actual_trans,
#                          greg_fees_vs_actual,
#                          greg_fees_vs_actual_trans,
#                          # historical_fees,
#                          # transparent_input_histogram,
#                        ]):
#     print(analysis)
# print(datetime.datetime.now() - start)

# rerunning old data …
# start = datetime.datetime.now()
# for analysis in a.analyze_blocks(make_weekly_range(206, 1),
#                        [ actual_fees,
#                          proposed_fees,
#                        ]):
#     print(analysis)
# print(datetime.datetime.now() - start)

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

def vin_value(vin):
    if 'valueSat' in vin:
        return vin['valueSat']
    else:
        return 0

def tx_pool_movement(tx):
    transparent = sum(vout['valueZat'] for vout in tx['vout']) - sum([vin_value(vin) for vin in tx['vin']])
    sprout = sum([vjoinsplit['vpub_newZat'] - vjoinsplit['vpub_oldZat'] for vjoinsplit in tx['vjoinsplit']])
    sapling = - tx['valueBalanceZat']
    if 'orchard' in tx:
        orchard = - tx['orchard']['valueBalanceZat']
    else:
        orchard = 0
    # print("(%d, %d, %d, %d) – %d -> %d" % (transparent, sprout, sapling, orchard, count_inputs(tx), count_outputs(tx)))
    return (transparent, sprout, sapling, orchard)

pool_movement = Analysis(
    "how are funds moving between pools?",
    lambda *_: True,
    [(lambda _c, block, _t: int(block['height'] / (blocks_per_hour * 24)), lambda vals: np.sum(np.array(vals), 0))],
    lambda _c, _b, tx: tx_pool_movement(tx)
)


start = datetime.datetime.now()
for analysis in a.analyze_blocks(recent_range,
                       [ pool_movement
                       ]):
    print(analysis)
print(datetime.datetime.now() - start)

# start = datetime.datetime.now()
# for analysis in a.analyze_blocks(pre_sandblasting_range,
#                        [ tx_below_pofm_threshold,
#                          tx_below_pofm_threshold_5,
#                          tx_below_pofm_threshold_max,
#                          tx_below_pofm_threshold_ins,
#                          tx_below_pofm_threshold_abs,
#                          outs_below_pofm_threshold_abs,
#                          arity_heat_map,
#                          minimum_pofm_fees_nuttycom,
#                          minimum_pofm_fees_nuttycom2,
#                        ]):
#     print(analysis)
# print(datetime.datetime.now() - start)
