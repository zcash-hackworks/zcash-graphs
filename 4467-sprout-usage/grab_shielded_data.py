import os
import sys
from slickrpc.rpc import Proxy
import csv
import progressbar

usage = f'Usage: {sys.argv[0]} OUTPUTDIR RPCUSER RPCPASSWORD [ HOST [ PORT ] ]\n\nFound: {sys.argv}'
args = sys.argv[1:]

def pop_arg(name, default=None):
    try:
        return args.pop(0)
    except IndexError:
        if default is None:
            raise SystemExit(f'Missing {name} argument.\n\n{usage}')
        else:
            return default

os.chdir(pop_arg('OUTPUTDIR'))
user = pop_arg('RPCUSER')
passwd = pop_arg('RPCPASSWORD')
host = pop_arg('HOST', default='127.0.0.1')
port = pop_arg('PORT', default='8232')

if len(args) > 0:
    raise SystemExit(f'Unexpected args.\n\n{usage}')

rpc_url = f'http://{user}:{passwd}@{host}:{port}/'
print(f'RPC URL: {rpc_url!r}')
rpc_connection = Proxy(rpc_url)

cur_height = rpc_connection.getblockcount()
pbar = progressbar.ProgressBar(max_value=cur_height)

# Genesis
cur_block_hash = '00040fe8ec8471911baa1db1266ea15dd06b4a8a5c453883c000b031973dce08'

with open('zcash-shielded-data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    while cur_block_hash:
        data = rpc_connection.getblock(cur_block_hash, 2)

        txs = []
        num_sprout = 0
        num_transparent = 0
        num_with_sapling = 0
        for tx in data['tx']:
            sprout_shielding = [jsdesc['vpub_oldZat'] for jsdesc in tx['vjoinsplit'] if jsdesc['vpub_oldZat'] > 0]
            sprout_unshielding = [jsdesc['vpub_newZat'] for jsdesc in tx['vjoinsplit'] if jsdesc['vpub_newZat'] > 0]
            sapling_shielding = -tx['valueBalanceZat'] if 'valueBalanceZat' in tx and tx['valueBalanceZat'] < 0 else 0
            sapling_unshielding = tx['valueBalanceZat'] if 'valueBalanceZat' in tx and tx['valueBalanceZat'] > 0 else 0

            is_mixed = len(tx['vjoinsplit']) > 0 and 'vShieldedSpend' in tx and (len(tx['vShieldedSpend']) + len(tx['vShieldedOutput'])) > 0
            is_migration = len(sprout_unshielding) > 0 and 'vShieldedSpend' in tx and len(tx['vShieldedSpend']) == 0 and len(tx['vShieldedOutput']) > 0

            txs.append({
                'num_jsdesc': len(tx['vjoinsplit']),
                'sprout_shielding': sprout_shielding,
                'sprout_unshielding': sprout_unshielding,
                'sapling_shielding': sapling_shielding,
                'sapling_unshielding': sapling_unshielding,
                'is_mixed': is_mixed,
                'is_migration': is_migration,
            })

            if len(tx['vjoinsplit']) > 0:
                # Transactions that involve Sprout
                num_sprout = num_sprout + 1
            elif 'vShieldedSpend' in tx and ((len(tx['vShieldedSpend']) + len(tx['vShieldedOutput'])) > 0):
                # Transactions that don't involve Sprout and do involve Sapling
                num_with_sapling = num_with_sapling + 1
            else:
                # Purely transparent transactions
                num_transparent = num_transparent + 1

        num_jsdesc = sum([tx['num_jsdesc'] for tx in txs])
        num_shielding = len([tx for tx in txs if len(tx['sprout_shielding']) > 0])
        num_unshielding = len([tx for tx in txs if len(tx['sprout_unshielding']) > 0])
        num_less_private_shielding = len([tx for tx in txs if len(tx['sprout_shielding']) > 1])
        num_less_private_unshielding = len([tx for tx in txs if len(tx['sprout_unshielding']) > 1])
        num_mixed = len([tx for tx in txs if tx['is_mixed']])
        num_migrating = len([tx for tx in txs if tx['is_migration']])

        sprout_shielding = sum([value for tx in txs for value in tx['sprout_shielding']])
        sprout_unshielding = sum([value for tx in txs for value in tx['sprout_unshielding']])
        sapling_shielding = sum([tx['sapling_shielding'] for tx in txs])
        sapling_unshielding = sum([tx['sapling_unshielding'] for tx in txs])

        writer.writerow([
            data['height'],
            num_transparent,
            num_with_sapling,
            num_sprout,
            num_jsdesc,
            num_shielding,
            num_unshielding,
            num_less_private_shielding,
            num_less_private_unshielding,
            num_mixed,
            num_migrating,
            sprout_shielding,
            sapling_shielding,
            sprout_unshielding,
            sapling_unshielding,
        ])
        if data['height'] <= cur_height:
            pbar.update(data['height'])
        cur_block_hash = data.get('nextblockhash', None)

