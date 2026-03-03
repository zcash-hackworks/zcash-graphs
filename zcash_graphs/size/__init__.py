def compact_size_size(nSize):
    """
    Returns the encoded size of a `CompactSize` type.

    - `nSize` is the number being encoded.
    """
    if nSize < 253:
        return 1
    elif nSize <= 0xFFFF:
        return 3
    elif nSize <= 0xFFFFFFFF:
        return 5
    else:
        return 9

def script_size(script):
    """
    Returns the encoded size of a transparent script.

    - `script` is the raw byte encoding of the script.
    """
    return (
        compact_size_size(len(script)) +
        len(script)
    )

BLOCK_HEADER_SIZE = (
    4  + # nVersion
    32 + # hashPrevBlock
    32 + # hashMerkleRoot
    32 + # hashFinalSaplingRoot
    4  + # nTime
    4  + # nBits
    32 + # nNonce
    compact_size_size(1344) + # solutionSize
    1344 # solution
)

def block_size(vtx):
    return (
        BLOCK_HEADER_SIZE +
        compact_size_size(len(vtx)) +
        sum([tx.size() for tx in vtx])
    )
