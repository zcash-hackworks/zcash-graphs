def _varint_size(n):
    return (len(bin(n)[2:]) + 6) // 7

def _tag_size(field_number, wire_type):
    return _varint_size((field_number << 3) | wire_type)

def _wire_varint(field_number, n):
    return _tag_size(field_number, 0) + _varint_size(n)

def _wire_len(field_number, len):
    return _tag_size(field_number, 2) + _varint_size(len) + len

def p_uint32(f, n):
    """
    Returns the encoded size of a `uint32` scalar value type.

    - `n` is the number being encoded.
    """
    return _wire_varint(f, n)

def p_uint64(f, n):
    """
    Returns the encoded size of a `uint64` scalar value type.

    - `n` is the number being encoded.
    """
    return _wire_varint(f, n)

def p_bytes(f, len):
    """
    Returns the encoded size of a `bytes` scalar value type.

    - `f` is the field number.
    - `len` is the length in bytes of the field's data.
    """
    return _wire_len(f, len)

def p_repeated_of(f, kind, data):
    """
    Returns the encoded size of a `repeated` field where each element may have a different
    length.

    This function implements the non-packed format (used by non-primitive types, i.e.
    `string`, `bytes`, or a non-scalar type).

    - `f` is the field number.
    - `kind` is a function that will be called with the arguments `(i, x)` where `x` is an
      element of `data`, and `i` is its enumeration. It should return the length of the
      proto3 encoding of the element.
    """
    return sum([_wire_len(f, kind(i, x)) for (i, x) in enumerate(data)])

def p_repeated(f, len, count):
    """
    Returns the encoded size of a `repeated` field where every element has the same length.

    This function implements the non-packed format (used by non-primitive types, i.e.
    `string`, `bytes`, or a non-scalar type).

    - `f` is the field number.
    - `len` is the length of each element.
    - `count` is the number of elements being encoded.
    """
    return _wire_len(f, len) * count

def p_nested(f, inner_len):
    """
    Returns the encoded size of a submessage field (i.e. a non-scalar field).
    """
    # LEN wrapper around submessages so unknown ones can be skipped
    return _wire_len(f, inner_len)
