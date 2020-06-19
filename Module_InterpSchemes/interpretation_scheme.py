# Import
import numpy as np


''' # NOTES
Different schemes can be used to interpret the readings (which is
the (weighted) sum of the output voltages).
These include:
    'band_binary' ( or 0 )
    'pn_binary' ( or 1 )

'''


def GetClass(genome, out_weight_gene_loc, Vout,
             IntpScheme, num_output, num_output_readings,
             OutWeight_gene, OutWeight_scheme,
             EN_Dendrite, NumDendrite):

    # # set the number of loops for the Dendrite for loop
    len_loop_dendrite = NumDendrite*EN_Dendrite
    if len_loop_dendrite == 0:
        len_loop_dendrite = 1

    # initialise variables
    reading = np.zeros(num_output_readings)
    y_out = []
    responceY = []

    if OutWeight_gene == 1:
        OW_genes = np.arange(out_weight_gene_loc[0], out_weight_gene_loc[1])
        #print("OW_genes:", OW_genes)

    ############################################################
    # Schemes
    ############################################################

    # Interpret for band binary
    if IntpScheme == 'band_binary' or IntpScheme == 0:

        # Find output reading for each output dimension
        for out_dim in range(num_output_readings):
            rY_OutDim = []

            if OutWeight_gene == 1:
                i = 0
                for gene in range(OW_genes[out_dim*num_output], OW_genes[(out_dim+1)*num_output]):

                    OutWeight = genome[gene]

                    # # sigmoid the outputs from each output node:
                    #v_sig = (1/(1+np.exp(-Vout[i]))) - 0.5
                    #reading[out_dim] = OutWeight*v_sig + reading[out_dim]

                    # # Reading from raw outputs
                    reading[out_dim] = OutWeight*Vout[i] + reading[out_dim]

                    #prev_reading = reading[out_dim]
                    #print("out_dim:", out_dim, " Weighted output:", genome[gene]*Vout[i])
                    #print("IS: ", reading[out_dim], " = ",  OutWeight, " * ", Vout[i], " + ", prev_reading, "  Actual weight:", genome[gene])
                    i = i + 1
            else:
                for i in range(num_output):
                    reading[out_dim] = Vout[i] + reading[out_dim]

            # print("reading dim", out_dim, " is:", reading[out_dim])

            # # Calculate the class output from the reading
            # Note: using bounded decision (nessercary for XOR gates)
            # ALSO! Determine class boundary closeness, 0=close to boundary
            if reading[out_dim] > -1 and reading[out_dim] < 1:
                y_out.append(2)
                responceY.append(1 - abs(reading[out_dim]))  # find weight as a decimal
            else:
                y_out.append(1)
                # responceY[loop, out_dim] = (abs(reading[out_dim]) -1)/((MaxOutputWeight*Vmax)-1)  # find weight as a decimal percentage of full range
                responceY.append(-(abs(reading[out_dim]) -1))  # find weight, ensuring opposite class is negative

        return y_out, responceY

    #

    # Interpret for +/- binary
    elif IntpScheme == 'pn_binary' or IntpScheme == 1:

        # Find output reading for each output dimension
        for out_dim in range(num_output_readings):
            NR_list = []

            if OutWeight_gene == 1:

                # # allow a network responce for each class
                for num_NR in range(len_loop_dendrite):
                    start = num_NR*(out_dim+1)*num_output
                    #print("start", start)
                    fin = (out_dim+1)*num_output + num_NR*(out_dim+1)*num_output-1
                    #print("fin", fin)
                    the_range = np.arange(OW_genes[start], OW_genes[fin]+1)  # find the current OW genes to apply to this dendrite reading
                    #print("the_range", the_range)

                    i = 0
                    for gene in the_range:

                        OutWeight = genome[gene]

                        # # sigmoid the outputs from each output node:
                        #v_sig = (1/(1+np.exp(-Vout[i]))) - 0.5
                        #reading[out_dim] = OutWeight*v_sig + reading[out_dim]

                        # # Reading from raw outputs
                        reading[out_dim] = OutWeight*Vout[i] + reading[out_dim]

                        #prev_reading = reading[out_dim]
                        #print("out_dim:", out_dim, " Weighted output:", genome[gene]*Vout[i])
                        #print("IS: ", reading[out_dim], " = ",  OutWeight, " * ", Vout[i], " + ", prev_reading, "  Actual weight:", genome[gene])
                        #print("IS: ", OutWeight, "Actual weight:", genome[gene])
                        i = i + 1
                    NR_list.append(reading[out_dim])
            else:
                for i in range(num_output):
                    reading[out_dim] = Vout[i] + reading[out_dim]
                NR_list.append(reading[out_dim])

            # # Calculate the class & boundary weight output from the reading
            sol_scheme = 'powers'

            if sol_scheme == 'normal':
                dendrite_mean = sum(NR_list)/len(NR_list)
                if dendrite_mean >= 0:
                    y_out.append(2)
                    responceY.append(dendrite_mean)
                else:
                    y_out.append(1)
                    responceY.append(dendrite_mean)

            elif sol_scheme == 'powers':
                total_red = 0
                i = 1

                for red in NR_list:
                    total_red = total_red + red**(i)
                    i = i + 1

                if total_red >= 0:
                    y_out.append(2)
                    responceY.append(total_red)
                else:
                    y_out.append(1)
                    responceY.append(total_red)

        return y_out, responceY

    # Highest Output wins protocol,
    # where output 1 is class 1 and output 2 is class 2, and so on
    elif IntpScheme == 'HOwins' or IntpScheme == 2:

        # Find output reading for each output dimension
        for out_dim in range(num_output_readings):

            for i in range(len(Vout)):
                if i == 0:
                    largest_voltage = Vout[i]
                    the_class = i + 1
                    bw = -1
                elif Vout[i] > largest_voltage:
                    largest_voltage = Vout[i]
                    the_class = i + 1
                    bw = 1

            y_out.append(the_class)
            responceY.append(bw)

        return y_out, responceY

    else:  # i.e input is wrong / not vailable
        print(" ")
        print("Error (interpretation_scheme): Invalid scheme")
        raise ValueError('(interpretation_scheme): Invalid scheme')
#

#

# fin
