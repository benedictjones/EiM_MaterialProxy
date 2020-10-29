import numpy as np
from functools import reduce
import time


# # # # # # # # # # # # # # # # # # # # #
# Fetches the number of perms that are generated, from the input sequence
def NPerms_FromSeq(seq):
    # computes the factorial of the length of
    return reduce(lambda x, y: x * y, range(1, len(seq) + 1), 1)


# # # # # # # # # # # # # # # # # # # # #
# Fetches the perm from a particular index (note starts from zero)
def IndexPerm_FromSeq(seq, index):
    # Returns the th permutation of  (in proper order)
    seqc = list (seq [:])
    result = []
    fact = NPerms_FromSeq(seq)

    if index >= fact:
        print("fact:", fact, "index:", index)
        raise ValueError("Error (FetchPerm.py): index was larger then the max number of perms")

    index %= fact
    while seqc:
        fact = fact / len(seqc)
        choice, index = index // fact, index % fact
        result += [seqc.pop(int(choice))]
    return result


#

#

#

# # # # # # # # # # # # # # # # # # # # #
# Fetches the number of perms that are generated, from the input sequence length (num nodes)
def NPerms(TotalNumInputs):
    # computes the factorial of the length of
    seq = np.arange(TotalNumInputs)
    return reduce(lambda x, y: x * y, range(1, len(seq) + 1), 1)


# # # # # # # # # # # # # # # # # # # # #
# Fetches the perm from a particular index (note starts from zero)
def IndexPerm(TotalNumInputs, index):

    index = int(index)
    #print("truncated inx:", index)

    # Returns the th permutation of  (in proper order)
    seq = np.arange(TotalNumInputs)
    seqc = list (seq [:])
    result = []
    fact = NPerms_FromSeq(seq)

    if index >= fact:
        print("fact:", fact, "index:", index)
        raise ValueError("Error (FetchPerm.py): index was larger then the max number of perms")

    index %= fact
    while seqc:
        fact = fact / len(seqc)
        choice, index = index // fact, index % fact
        result += [seqc.pop(int(choice))]
    return result


def reorder(arr, index):
    """ Function which re-orders an array acording to the input perm
    """
    # # Error Check inputs
    if len(arr) != len(index):
        print("Error (GetVoltageInputs): reorder input array length does not equal index length")
        raise ValueError(' reorder input array length does not equal index length')

    new_arr = []
    for loc in index:
        new_arr.append(arr[loc])

    return new_arr


def reorder_cols(matrix, index):
    """ Function which re-orders an array acording to the input perm
    """
    new_matrix = np.zeros(matrix.shape)
    for new_idx, old_idx in enumerate(index):
        print(new_idx, old_idx)
        new_matrix[:, new_idx] = matrix[:, old_idx]
    return new_matrix






'''
# # # # # # # # # # # # # # # # # # # # #
# Test script
# # # # # # # # # # # # # # # # # # # # #

# Print number of perms
sequence = np.arange(5)
print(sequence)
print("Number of perms:", NPerms_FromSeq(sequence))

i = 0
tic = time.time()
print(i, IndexPerm_FromSeq(sequence, i))
toc = time.time()
# # Print individual processor execution time
print("tic:", tic, " toc:", toc)
print("execution time:", toc - tic)

for i in range (79, 90):
    print(i, IndexPerm(5, i))
'''
