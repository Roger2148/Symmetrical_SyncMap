#from keras.utils import np_utils
import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
#problems

from FixedChunkTest_v2 import *
from GraphWalkTest_v2 import *
from GraphWalkTest_v3_adj_mat import *
from GraphWalkTest_v2_karate_club import *
from GraphWalkTest_v2_dolphins import *
import sys

# neurons
from Symm_SyncMap_v4_new_submission import *

from VAE import *

save_dir = "output_files/"

save_filename = save_dir + "test"
save_truth_filename = save_dir + "test" + "_truth"

time_delay = 10

print("initializing...")
FixedChunkTest_v2(time_delay),
GraphWalkTest_v2(time_delay),
GraphWalkTest_v3_adj_mat(time_delay),
GraphWalkTest_karate_club(time_delay)
########################################################################################################################
'''
We should use GraphWalkTest_v2 or FixedChunkTest_v2
'''

ifDynamic = False

# env = FixedChunkTest_v2(time_delay, filename="data/UB_fixed_chunk_20_10_5.txt")
env = GraphWalkTest_v2(time_delay, filename="data/graph_prob.dot")
# env = GraphWalkTest_karate_club(time_delay)
# env = GraphWalkTest_dolphins(time_delay)
# env = GraphWalkTest_v3_adj_mat(time_delay) #SBM model
print("Environment No.1: ", env.name)


# # for sequence 2
if ifDynamic:
    env2 = FixedChunkTest_v2(time_delay, filename="data/fixed_chunk2.txt")
    # env2 = GraphWalkTest_v2(time_delay, filename="data/UB_prob_chunk_20_10_5.dot")
    print("Environment No.2: ", env2.name)


output_size = env.getOutputSize()
print("Output Size", output_size)

sequence_length = 200000

####### SyncMap #####
number_of_nodes = output_size
adaptation_rate = 0.001 * output_size
print("Adaptation rate:", adaptation_rate)
map_dimensions = [3]
space_size = 10 #0.25*output_size
print("space_size:", space_size)
dropout_rate=0.3
eps = 4.5 #space_size/5
# np.random.seed(42)
########################################################################################################################
number_test = 1
# NMI_list = np.zeros([number_test, 2])

# SyncMap_list = []
nmi_karate_2_icrl = []
nmi_karate_4_icrl = []
for map_dim in enumerate(map_dimensions):
    for n_t in range(number_test):
        print("Starting No.", n_t, "...")
        ####### SyncMap #####
        neuron_group = Symm_SyncMap_v4_new_submission(number_of_nodes, map_dim[1], adaptation_rate, space_size,
                                    sequence_length, dropout_rate, time_delay=10,
                               original_syncmap=False, state_memory=3, stochastic_pick=True,
                               positive_sto_pick_num=3, negative_sto_pick_num=3,
                               positive_state_memory_random_select_prob=0.3, negative_state_memory_random_select_prob=0.3,
                               eps=eps, ifdynamic=ifDynamic, movmean_window=1000)

        # TO USE ORIGINAL SYNCMAP
        # neuron_group = Symm_SyncMap_v4_new_submission(number_of_nodes, map_dim[1], adaptation_rate, space_size,
        #                             sequence_length, dropout_rate, time_delay=10,
        #                        original_syncmap=True, state_memory=2, stochastic_pick=True,
        #                        positive_sto_pick_num=2, negative_sto_pick_num=2,
        #                        positive_state_memory_random_select_prob=0.3, negative_state_memory_random_select_prob=0.3,
        #                        eps=eps, ifdynamic=ifDynamic, movmean_window=1)

        # Roger test
        syncmap_init = neuron_group.syncmap
        labels = neuron_group.organize()
        ####### SyncMap #####

        print("map latent dim: ", map_dim[1])
        print("getting Input data..")
        input_sequence, input_class = env.getSequence(sequence_length)

        if ifDynamic:
            input_sequence2, input_class2 = env2.getSequence(sequence_length)
            input_sequence = np.concatenate( (input_sequence,input_sequence2), axis=0)
            input_class = np.concatenate( (input_class, input_class2))

        # ==========================================
        print("training..")
        neuron_group.input(input_sequence)

        #### NMI 1 ###
        # label_list1 = []
        label_list2 = []
        # NMI_list_list1 = []
        NMI_list_list2 = []
        true_labels = env.trueLabel()
        # if testing Long-term Behavior Analysis
        # true_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        #                         1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        #                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
        switch_ = int(len(neuron_group.syncmap_history_list)/2)
        switch_counter = 0
        # for map1, map2 in zip(neuron_group.syncmap_history_list, neuron_group.syncmap_movmean_history_list):
        for map2 in neuron_group.syncmap_movmean_history_list:
            # temp_labels1 = DBSCAN(eps=neuron_group.eps, min_samples=2).fit_predict(map1)
            temp_labels2 = DBSCAN(eps=neuron_group.eps, min_samples=2).fit_predict(map2)

            # iclr karate club
            # true_labels_kc = np.array(
            #     [0, 0, 1, 0, 2, 2, 2, 0, 3, 1, 2, 0, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3])

            # label_list1.append(temp_labels1)
            label_list2.append(temp_labels2)

            if ifDynamic and switch_counter>=switch_:
                true_labels = env2.trueLabel()
                # true_labels = np.array([0,1,2,3,4,5,6,7,8,9,   0,1,2,3,4,5,6,7,8,9,    0,1,2,3,4,5,6,7,8,9,
                #                          0,1,2,3,4,5,6,7,8,9,   0,1,2,3,4,5,6,7,8,9,    0,1,2,3,4,5,6,7,8,9])
            # NMI_dbscan_temp1 = normalized_mutual_info_score(true_labels, temp_labels1)
            # NMI_list_list1.append(NMI_dbscan_temp1+0)
            NMI_dbscan_temp2 = normalized_mutual_info_score(true_labels, temp_labels2)

            NMI_list_list2.append(NMI_dbscan_temp2+0)
            switch_counter = switch_counter+1

            #############
        # create a NMI list for many tests
        if n_t==0:
            # NMI_all_test1 = np.zeros([number_test, len(NMI_list_list1)])
            NMI_all_test2 = np.zeros([number_test, len(NMI_list_list2)])
        # NMI_all_test1[n_t, :] = np.array(NMI_list_list1)
        NMI_all_test2[n_t, :] = np.array(NMI_list_list2)
        print("NMI = ", NMI_all_test2[n_t, -1])

        #iclr
        # nmi_list_icrl.append(NMI_list_list2[-1])
        # neuron_group.labels = env.trueLabel()
        # neuron_group.plot()
        # Z = linkage(neuron_group.temp_syncmap_movmean, "ward");
        # dendrogram(Z);
        # clu = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(
        #     neuron_group.temp_syncmap_movmean)
        # labels_hierarchy2 = clu.labels_
        # print("Learned Labels Hierachy: ", labels_hierarchy2)
        # print("True       Labels Hierachy: ", true_labels)
        # print("NMI_hierarchical_Clu: ", normalized_mutual_info_score(true_labels, labels_hierarchy2))

        # clu = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit(
        #     neuron_group.temp_syncmap_movmean)
        # labels_hierarchy4 = clu.labels_
        # print("Learned Labels Hierachy: ", labels_hierarchy4)
        # print("True       Labels Hierachy: ", true_labels_kc)
        # print("NMI_hierarchical_Clu: ", normalized_mutual_info_score(true_labels_kc, labels_hierarchy4))

        # nmi_karate_2_icrl.append(normalized_mutual_info_score(true_labels, labels_hierarchy2))
        # nmi_karate_4_icrl.append(normalized_mutual_info_score(true_labels_kc, labels_hierarchy4))

        pau=0




#####################################################################################################################
if save_filename is not None:

    with open(save_filename, "a+") as f:
        tmp = np.array2string(temp_labels2, precision=2, separator=',')
        f.write(tmp + "\n")
        f.closed

    if temp_labels2 is not None:
        with open(save_truth_filename, "a+") as f:
            tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
            f.write(tmp + "\n")
            f.closed
