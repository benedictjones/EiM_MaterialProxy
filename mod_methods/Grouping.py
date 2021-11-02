import numpy as np

#

#

def group_genome(genome, gen_grouping):
    """
    Group the hdf5 saved genome, according to the passed in genome grouping
    list - reverting the genome to its original form.

    This can then be indexed using the gene_loc values.
    """

    grouped_genome = []

    #if len(gen_grouping) == 1:
    #    genome = np.asarray([genome])

    #rounded_genome = np.around(genome, decimals=3)
    #print("rounded_genome", rounded_genome)

    # print("\n\n\ngenome", genome)
    i = 0
    for loc, size in enumerate(gen_grouping):
        gen_group = []
        for c in range(size):
            gen_group.append(genome[i])
            i = i + 1

        grouped_genome.append(np.asarray(gen_group))

    # print("grouped_genome", grouped_genome)

    return grouped_genome

#

#


def fetch_genome_string(genome, gen_grouping):
    """
    Groups the raw (hdf5 saved) unformatted genome according to the passed
    in genome grouping list.

    Then spits out a formatted text version.
    """

    #print("\nGenome:", genome)
    #print("grouping:", gen_grouping)

    # # Round if wanted
    rounded_genome = np.around(genome, decimals=3)
    #print("rounded_genome", rounded_genome)

    # # Format genome into text
    best_genome_text = '['
    i = 0
    group = 0
    for size in gen_grouping:
        for c in range(size):

            best_genome_text = best_genome_text + str(rounded_genome[i])
            i = i + 1

            if c == size-1 and group != len(gen_grouping)-1:
                best_genome_text = best_genome_text + ' | '
            elif c != size-1:
                best_genome_text = best_genome_text + ', '
            else:
                best_genome_text = best_genome_text + ']'
        group = group + 1

    return best_genome_text
