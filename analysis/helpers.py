# Copyright (c) 2022 The Zcash developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://www.opensource.org/licenses/mit-license.php .
"""Useful function for transaction analyses

This is a collection of functions that make it easier to write new transaction
analyses.
"""

import datetime
import itertools
import math

blocks_per_hour = 48 # half this before NU2?

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
