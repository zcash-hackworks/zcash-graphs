from slickrpc.rpc import Proxy
import csv
import progressbar

rpc_connection = Proxy("http://RPC_USER:RPC_PASSWORD@127.0.0.1:8232")

cur_height = rpc_connection.getblockcount()
pbar = progressbar.ProgressBar(max_value=cur_height)

# Genesis
cur_block_hash = '00040fe8ec8471911baa1db1266ea15dd06b4a8a5c453883c000b031973dce08'

with open('zcash-sprout-data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    while cur_block_hash:
        data = rpc_connection.getblock(cur_block_hash, 2)

        sprout_txs = []
        num_transparent = 0
        num_with_sapling = 0
        for tx in data['tx']:
            if len(tx['vjoinsplit']) > 0:
                # Transactions that involve Sprout
                value_shielded = [jsdesc['vpub_oldZat'] for jsdesc in tx['vjoinsplit'] if jsdesc['vpub_oldZat'] > 0]
                value_unshielded = [jsdesc['vpub_newZat'] for jsdesc in tx['vjoinsplit'] if jsdesc['vpub_newZat'] > 0]
                sprout_txs.append({
                    'num_jsdesc': len(tx['vjoinsplit']),
                    'value_shielded': value_shielded,
                    'value_unshielded': value_unshielded,
                })
            elif 'vShieldedSpend' in tx and ((len(tx['vShieldedSpend']) + len(tx['vShieldedOutput'])) > 0):
                # Transactions that don't involve Sprout and do involve Sapling
                num_with_sapling = num_with_sapling + 1
            else:
                # Purely transparent transactions
                num_transparent = num_transparent + 1

        # Extract some interesting metrics
        num_sprout = len(sprout_txs)
        num_jsdesc = sum([tx['num_jsdesc'] for tx in sprout_txs])
        num_shielding = len([tx for tx in sprout_txs if len(tx['value_shielded']) > 0])
        num_unshielding = len([tx for tx in sprout_txs if len(tx['value_unshielded']) > 0])
        value_shielded = sum([value for tx in sprout_txs for value in tx['value_shielded']])
        value_unshielded = sum([value for tx in sprout_txs for value in tx['value_unshielded']])
        num_less_private_shielding = len([tx for tx in sprout_txs if len(tx['value_shielded']) > 1])
        num_less_private_unshielding = len([tx for tx in sprout_txs if len(tx['value_unshielded']) > 1])

        writer.writerow([
            data['height'],
            num_transparent,
            num_with_sapling,
            num_sprout,
            num_jsdesc,
            num_shielding,
            num_less_private_shielding,
            value_shielded,
            num_unshielding,
            num_less_private_unshielding,
            value_unshielded,
        ])
        if data['height'] <= cur_height:
            pbar.update(data['height'])
        cur_block_hash = data.get('nextblockhash', None)

