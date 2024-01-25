"""
Assignment 2 starter code
CSC148, Winter 2023

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}

    for byte in text:
        if byte in d:
            d[byte] += 1
        else:
            d[byte] = 1

    return d


def spec_sort(lst: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Variation of Mergesort to sort key-value tuples by values
    """
    if len(lst) < 2:
        return lst[:]
    else:
        mid = len(lst) // 2
        left_sorted = spec_sort(lst[:mid])
        right_sorted = spec_sort(lst[mid:])

        return spec_merge(left_sorted, right_sorted)


def spec_merge(left: list[tuple[int, int]],
               right: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Variation of _merge to sort key-value tuples by values
    """
    index1 = 0
    index2 = 0
    merged = []

    while index1 < len(left) and index2 < len(right):
        if left[index1][1] <= right[index2][1]:
            merged.append(left[index1])
            index1 += 1
        else:
            merged.append(right[index2])
            index2 += 1

    return merged + left[index1:] + right[index2:]


def sorted_append(ns_trees: list[HuffmanTree],
                  ns_freqs: list[int],
                  ns_tree: HuffmanTree,
                  ns_freq: int) -> None:
    """
    Appends ns_tree in ns_trees and ns_freq in
    ns_freqs based on value of ns_freq
    """
    if not ns_trees:
        ns_trees.append(ns_tree)
        ns_freqs.append(ns_freq)
    else:
        for i in range(len(ns_trees)):
            if ns_freq < ns_freqs[i]:
                ns_trees.insert(i - 1, ns_tree)
                ns_freqs.insert(i - 1, ns_freq)
                break
            elif i == len(ns_trees) - 1:
                ns_trees.append(ns_tree)
                ns_freqs.append(ns_freq)


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        k = next(iter(freq_dict))
        if k == 0:
            return HuffmanTree(None, HuffmanTree(0), HuffmanTree(1))
        else:
            return HuffmanTree(None, HuffmanTree(k - 1), HuffmanTree(k))

    lst = [(key, freq_dict[key]) for key in freq_dict]
    lst = spec_sort(lst)

    ns_trees = [HuffmanTree(None, HuffmanTree(lst[0][0]),
                            HuffmanTree(lst[1][0]))]
    ns_freqs = [lst[0][1] + lst[1][1]]

    new_added = False
    i = 2  # Counter variable

    while i < len(lst):
        leaf = HuffmanTree(lst[i][0])
        freq_l = lst[i][1]

        if i < len(lst) - 1:
            next_l = HuffmanTree(lst[i + 1][0])
            next_fr = lst[i + 1][1]

            if ns_freqs[0] >= next_fr:
                ns_tree = HuffmanTree(None, leaf, next_l)
                ns_freq = next_fr + freq_l
                sorted_append(ns_trees, ns_freqs, ns_tree, ns_freq)
                i += 2
                new_added = True

        if not new_added:
            if len(ns_freqs) > 1 and ns_freqs[1] < freq_l:
                ns_tree = HuffmanTree(None, ns_trees[0], ns_trees[1])
                ns_freq = ns_freqs[0] + ns_freqs[1]
                ns_trees, ns_freqs = ns_trees[2:], ns_freqs[2:]
                sorted_append(ns_trees, ns_freqs, ns_tree, ns_freq)
            elif freq_l <= ns_freqs[0]:
                n_tree = HuffmanTree(None, leaf, ns_trees[0])
                n_freq = freq_l + ns_freqs[0]
                ns_trees, ns_freqs = ns_trees[1:], ns_freqs[1:]
                sorted_append(ns_trees, ns_freqs, n_tree, n_freq)
                i += 1
            else:
                n_tree = HuffmanTree(None, ns_trees[0], leaf)
                n_freq = freq_l + ns_freqs[0]
                ns_trees, ns_freqs = ns_trees[1:], ns_freqs[1:]
                sorted_append(ns_trees, ns_freqs, n_tree, n_freq)
                i += 1

        new_added = False

    while len(ns_trees) > 1:
        ns_tree = HuffmanTree(None, ns_trees[0], ns_trees[1])
        ns_freq = ns_freqs[0] + ns_freqs[1]
        ns_trees, ns_freqs = ns_trees[2:], ns_freqs[2:]
        sorted_append(ns_trees, ns_freqs, ns_tree, ns_freq)

    return ns_trees[0]


def find_codes(tree: HuffmanTree, code: str) -> dict[int, str]:
    """
    Recursively loops through tree while keeping track of codes
    with code parameter
    """
    if tree.is_leaf():
        return {tree.symbol: code}
    else:
        d = {}
        d.update(find_codes(tree.left, code + "0"))
        d.update(find_codes(tree.right, code + "1"))

        return d


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return find_codes(tree, "")


def all_internal_nodes(tree: HuffmanTree) -> list[HuffmanTree]:
    """
    Returns a list of internal nodes in tree
    """
    if tree.is_leaf():
        return []
    elif tree.left.is_leaf() and tree.right.is_leaf():
        return [tree]
    else:
        ret_lst = []
        ret_lst.extend(all_internal_nodes(tree.left))
        ret_lst.extend(all_internal_nodes(tree.right))
        ret_lst.append(tree)

        return ret_lst


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    int_nod_lst = all_internal_nodes(tree)

    for i in range(len(int_nod_lst)):
        int_nod_lst[i].number = i


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    code_dict = get_codes(tree)
    sum_w = 0
    sum_freq = 0

    for symbol in code_dict:
        sum_w += len(code_dict[symbol]) * freq_dict[symbol]
        sum_freq += freq_dict[symbol]

    return sum_w / sum_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # Source used to understand more about relationship between bits
    # and bytes: https://www.ime.usp.br/~pf/algorithms/chapters/bytes.html
    ret_bytes = []
    byte = ""
    start_ = 0
    end = 8

    for symbol in text:
        byte += codes[symbol]

        if len(byte) >= end:
            ret_bytes.append(bits_to_byte(byte[start_:end]))
            start_ += 8
            end += 8

    if start_ < len(byte) <= end:
        ret_bytes.append(bits_to_byte(byte[start_:]))

    return bytes(ret_bytes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    ret_lst = []
    int_nod_lst = all_internal_nodes(tree)

    for node in int_nod_lst:
        if node.left.is_leaf():
            ret_lst.append(0)
            ret_lst.append(node.left.symbol)
        else:
            ret_lst.append(1)
            ret_lst.append(node.left.number)

        if node.right.is_leaf():
            ret_lst.append(0)
            ret_lst.append(node.right.symbol)
        else:
            ret_lst.append(1)
            ret_lst.append(node.right.number)

    return bytes(ret_lst)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))

    result += compress_bytes(text, codes)

    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression
def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(1, 2, 1, 1), ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    root = node_lst[root_index]

    if root.l_type == 0 and root.r_type == 0:
        return HuffmanTree(None, HuffmanTree(root.l_data),
                           HuffmanTree(root.r_data))
    elif root.l_type == 0:
        r_tree = generate_tree_general(node_lst, root.r_data)
        return HuffmanTree(None, HuffmanTree(root.l_data), r_tree)
    elif root.r_type == 0:
        l_tree = generate_tree_general(node_lst, root.l_data)
        return HuffmanTree(None, l_tree, HuffmanTree(root.r_data))
    else:
        l_tree = generate_tree_general(node_lst, root.l_data)
        r_tree = generate_tree_general(node_lst, root.r_data)
        return HuffmanTree(None, l_tree, r_tree)


def num_internal_nodes(tree: HuffmanTree) -> int:
    """
    Returns number of internal nodes
    """
    if tree.is_leaf():
        return 0
    else:
        num_nodes = 1
        num_nodes += num_internal_nodes(tree.left)
        num_nodes += num_internal_nodes(tree.right)
        return num_nodes


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    root = node_lst[root_index]

    if root.l_type == 0 and root.r_type == 0:
        return HuffmanTree(None, HuffmanTree(root.l_data),
                           HuffmanTree(root.r_data))
    elif root.l_type == 0:
        r_tree = generate_tree_general(node_lst[:root_index], root_index - 1)
        return HuffmanTree(None, HuffmanTree(root.l_data), r_tree)
    elif root.r_type == 0:
        l_tree = generate_tree_postorder(node_lst, root_index - 1)
        return HuffmanTree(None, l_tree, HuffmanTree(root.r_data))
    else:
        r_tree = generate_tree_postorder(node_lst, root_index - 1)
        lr_num = root_index - num_internal_nodes(r_tree)
        # Number of leftmost node of r_tree

        l_tree = generate_tree_postorder(node_lst[:lr_num], lr_num - 1)
        return HuffmanTree(None, l_tree, r_tree)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    ret_list = [0] * size
    index = 0
    number_nodes(tree)
    node_lst = bytes_to_nodes(tree_to_bytes(tree))
    node = node_lst[-1]
    symbol = None

    for byte in text:
        bits = byte_to_bits(byte)

        for bit in bits:
            if bit == "0" and node.l_type == 0:
                symbol = node.l_data
            elif bit == "0":
                node = node_lst[node.l_data]
            elif node.r_type == 0:
                symbol = node.r_data
            else:
                node = node_lst[node.r_data]

            if symbol is not None:
                ret_list[index] = symbol
                symbol = None
                node = node_lst[-1]
                index += 1

            if index == size:
                return bytes(ret_list)

    return bytes(ret_list)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def get_leaf_nodes(tree: HuffmanTree) -> list[HuffmanTree]:
    """
    Return a dictionary mapping leaf nodes to length of leaf nodes
    """
    if tree.is_leaf():
        return [tree]
    else:
        ret_list = []
        ret_list.extend(get_leaf_nodes(tree.left))
        ret_list.extend(get_leaf_nodes(tree.right))

        return ret_list


def get_leaf_path_lengths(tree: HuffmanTree, height: int) -> list[int]:
    """
    Returns the length of each path from root nodes to
    leaf
    """
    if tree.is_leaf():
        return [height]
    else:
        ret_list = []
        ret_list.extend(get_leaf_path_lengths(tree.left, height + 1))
        ret_list.extend(get_leaf_path_lengths(tree.right, height + 1))

        return ret_list


def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    leaf_lst = get_leaf_nodes(tree)
    code_lst = get_leaf_path_lengths(tree, 0)

    for i in range(len(leaf_lst)):
        sym_1 = leaf_lst[i].symbol

        for j in range(len(leaf_lst)):
            sym_2 = leaf_lst[j].symbol

            if (freq_dict[sym_1] < freq_dict[sym_2]
                    and code_lst[i] < code_lst[j]):
                leaf_lst[i].symbol = sym_2
                leaf_lst[j].symbol = sym_1
                sym_1 = sym_2


if __name__ == "__main__":
    # import doctest
    #
    # doctest.testmod()
    #
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'allowed-io': ['compress_file', 'decompress_file'],
    #     'allowed-import-modules': [
    #         'python_ta', 'doctest', 'typing', '__future__',
    #         'time', 'utils', 'huffman', 'random'
    #     ],
    #     'disable': ['W0401']
    # })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
