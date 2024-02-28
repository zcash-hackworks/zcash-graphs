# Copyright (c) 2022 The Zcash developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php .
"""Simple Transaction Analysis

This contains a class, `Analysis`, for defining analyses of the blocks and
transactions on the blockchain. It also contains a class `Analyzer` with a
method `analyze_blocks`, which handles applying multiple analyses simultaneously
over some common range of blocks.
"""

import datetime
import itertools
import math
from progress.bar import IncrementalBar
from slickrpc.rpc import Proxy

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
        self.node = Proxy(node_url)

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
