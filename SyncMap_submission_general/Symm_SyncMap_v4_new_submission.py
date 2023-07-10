################################################################################ 
################################################################################ 

from keras.utils import np_utils
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from collections import deque

class Symm_SyncMap_v4_new_submission:

    def __init__(self, input_size, dimensions, adaptation_rate, space_size,
                 sequence_length, # necessary for recording history
                 dropout_rate=0.01, time_delay = 10,
                 original_syncmap=True, state_memory=2, stochastic_pick=False,
                 prob_select=False, regularization=False, freqency_control=False,
                 plus_update_freq_prob=0.8, minus_update_freq_prob=1,
                 positive_sto_pick_num=2, negative_sto_pick_num=2,
                 positive_state_memory_random_select_prob=0.3, negative_state_memory_random_select_prob=0.3,
                 eps=4.5, ifdynamic = False, movmean_window=5000):

        self.ifdynamic = ifdynamic

        if self.ifdynamic:
            sequence_length = (sequence_length)*2
        # else:
        #     sequence_length = sequence_length


        self.organized = False
        self.space_size = space_size
        self.dimensions = dimensions
        self.input_size = input_size
        # self.syncmap= np.zeros((input_size,dimensions))
        # I used normalization radius at 10 (space size)
        # np.random.seed(42)
        self.syncmap = np.random.rand(input_size, dimensions) * self.space_size - self.space_size/2
        self.adaptation_rate = adaptation_rate
        # self.syncmap= np.random.rand(dimensions, input_size)

        # Roger
        # bool values:
        self.stochastic_pick = stochastic_pick
        self.prob_select = prob_select
        self.regularization = regularization            # test
        self.original_syncmap = original_syncmap

        # Symm
        self.dropout_rate = dropout_rate
        self.syncmap_history = np.zeros([input_size, dimensions, sequence_length])
        self.syncmap_history_list = []
        self.syncmap_movmean_history_list = []
        self.movmean_window = movmean_window
        self.syncmap_movmean = deque(maxlen=self.movmean_window)
        self.temp_syncmap_movmean = self.syncmap+0
        self.time_delay = time_delay # used to generalize state memory
        self.plus_update_freq_prob = plus_update_freq_prob 
        self.minus_update_freq_prob = minus_update_freq_prob
        self.state_memory = state_memory
        self.positive_sto_pick_num = positive_sto_pick_num
        self.negative_sto_pick_num = negative_sto_pick_num
	    # probability parameter $P_r$ in the paper
        self.positive_state_memory_random_select_prob = positive_state_memory_random_select_prob
        self.negative_state_memory_random_select_prob = negative_state_memory_random_select_prob

        # test_parameter
        self.counter_plus = np.zeros(self.input_size)       # pre-running use
        self.counter_minus = np.zeros(self.input_size)      # pre-running use
        self.max_time_visit = 0         # test # pre-running use
        self.state_visit_ratio = np.zeros(self.input_size)  # pre-running use
        self.freq_control = freqency_control    # test
        # detecting global ROC (positive??)
        self.maxlen_ROC = 1
        self.pos_update_ROC_temp_window = deque(maxlen=self.maxlen_ROC)
        self.global_pos_update_ROC = []
        self.pairwise_dist_history = np.zeros((int(0.5 * self.input_size * (self.input_size - 1)), sequence_length))


        # novelty_strategy
        self.visit_matrix = np.zeros((sequence_length,self.input_size), dtype=bool)
        # self.adaptive_learning_rate = np.zeros(self.input_size)

        # DBSCAN
        self.eps = eps

        #


    def inputGeneral(self, x):
        print("[WARNING]: Symmetrical SyncMap requires all states in the input_sequence to have exactly the same time_delay,"
              " otherwise performance will decrease. \n")
        '''
        if using GraphWalkTest to generate Input sequence, do the following in main.py (add one value at the beginning):
        #####
        input_sequence, input_class = env.getSequence(sequence_length-1)
        input_sequence = np.concatenate( (input_sequence[0,:][:, np.newaxis].T, input_sequence), axis=0)
        temp1=np.zeros(sequence_length)
        temp1[1:sequence_length]=input_class
        temp1[0]=input_class[0]
        input_class = temp1
        #####
        '''


        sequence_size = x.shape[0]

        plus = x > 0.1 #Here the thresholding process is slightly different from that used in the paper. However, the final threshold results are the same.
        xmax = np.argmax(x, axis=1)
        for i1 in range(sequence_size):
            self.visit_matrix[i1, xmax[i1]] = True
        # minus = ~ plus

        # Roger
        dist_matrix = 1 # test
        plus_update_freq = np.random.rand(sequence_size, 1) <= self.plus_update_freq_prob   # test
        minus_update_freq = np.random.rand(sequence_size, 1) <= self.minus_update_freq_prob     # test
        positive_state_memory_random_select = np.random.rand(sequence_size, 1) <= self.positive_state_memory_random_select_prob
        negative_state_memory_random_select = np.random.rand(sequence_size, 1) <= self.negative_state_memory_random_select_prob

        # state memory setting
        past_state_num = self.state_memory-1
        past_state_set = np.zeros([past_state_num-1, self.input_size], dtype=bool)
        pre_plus = past_state_set.sum(axis=0, dtype=bool)
        #### !! ##
        ## if time_delay didn't fix, we will go with time_delay+1
        # actual_timedelay = self.time_delay + 1
        ## else
        # actual_timedelay = self.time_delay

        histroy = 0
        for i in tqdm(range(sequence_size)):


            # state memory changing:
            if i < self.state_memory * self.time_delay:
                pre_plus = plus[i, :]

                # this is to record the syncmap history
                # maximum = self.syncmap.max()
                # self.syncmap = self.space_size * self.syncmap / maximum
                # self.syncmap_history[:, :, i] = self.syncmap
                continue
            elif self.state_memory == 2:
                vplus = plus[i, :]
                pre_plus = np.zeros(vplus.shape, dtype=bool)
            elif i >= self.state_memory * self.time_delay:
                vplus, pre_plus = self.state_memory_generalization(i,plus,pre_plus,past_state_num,past_state_set)

                # # where there is a state transition happened, we modify and update its previous states
                # if (i % self.time_delay)==0:
                #     # vplus = plus[i, :]
                #     for j in range(past_state_num-1):
                #         past_state_set[(past_state_num-1) - (j+1), :] = plus[i - ((self.time_delay) * j + 1), :]
                #     pre_plus = past_state_set.sum(axis=0, dtype=bool)
                #     vplus = plus[i, :]
                #     vplus = np.logical_or(pre_plus, plus[i, :])
                # else:
                #     vplus = plus[i, :]
                #     vplus = np.logical_or(pre_plus, plus[i, :])

            vminus = ~ vplus

            ### test
            current_state = plus[i, :]
            if self.freq_control == True:
                # current_learning_rate = float((self.state_visit_ratio * current_state).sum()/2)
                current_state_temp = self.state_visit_ratio * current_state
                current_state_temp[current_state_temp.argmax()]=0
                current_learning_rate = float(current_state_temp.max())
            else:
                current_learning_rate = 1
            ### end test


            if self.original_syncmap == False:

                ### test
                if self.prob_select or self.regularization:
                    dist_matrix = pairwise_distances(self.syncmap)
                # vminus = self.dropout(vminus, self.dropout_rate).reshape(vplus.shape)
                # vminus = self.random_select_cn(vminus)
                ### end test

                if self.prob_select:
                    # vminus = self.prob_select_cn(vminus, dist_matrix, state_memory=self.state_memory)
                    vminus = self.prob_select_cn(vminus, dist_matrix, state_memory=self.state_memory)

                else:
                    if positive_state_memory_random_select[i,0]:
                        num_pick_temp_plus=self.positive_sto_pick_num
                    else:
                        num_pick_temp_plus=2

                    # have not consider
                    vplus = self.prob_select_cp_v1(vplus, num_pick=num_pick_temp_plus)
                    # vplus = self.random_select_cp(vplus, num_pick=num_pick_temp_plus)
                    vminus = ~vplus
                    if negative_state_memory_random_select[i,0]:
                        num_pick_temp_minus=self.negative_sto_pick_num
                    else:
                        num_pick_temp_minus=2
                    vminus = self.random_select_cn(vminus, num_pick=num_pick_temp_minus)

            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass <= 1:
                # Roger
                # this is to record the syncmap history
                # maximum = self.syncmap.max()
                # self.syncmap = self.space_size * self.syncmap / maximum
                # self.syncmap_history[:, :, i] = self.syncmap
                histroy=1
                continue

            if minus_mass <= 1:
                # Roger
                # this is to record the syncmap history
                # maximum = self.syncmap.max()
                # self.syncmap = self.space_size * self.syncmap / maximum
                # self.syncmap_history[:, :, i] = self.syncmap
                histroy = 1
                continue

            # print("vplus")
            # print(vplus)

            # Roger
            # vplus = self.random_select_cp(vplus, state_memory=3)
            # plus_mass = vplus.sum()

            center_plus = np.dot(vplus, self.syncmap) / plus_mass
            center_minus = np.dot(vminus, self.syncmap) / minus_mass

            # print(self.syncmap.shape)
            # exit()

            dist_plus = distance.cdist(center_plus[None, :], self.syncmap, 'euclidean')
            dist_minus = distance.cdist(center_minus[None, :], self.syncmap, 'euclidean')
            dist_plus = np.transpose(dist_plus)
            dist_minus = np.transpose(dist_minus)

            # Roger

            dist_bias_minus = 1
            dist_bias_plus = 1
            dist_minus_reg = dist_minus
            dist_plus_reg = dist_plus

            if self.regularization:
                dist_matrix_max = dist_matrix.max()
                dist_k = (1 / dist_matrix_max)
                # dist_k = dist_k + ((2 / dist_matrix_max) - (1 / dist_matrix_max))* ((i+1)/sequence_size)
                # dist_bias_minus = -dist_k * 1.1 * dist_minus + 1.5
                # dist_bias = dist_k * dist_minus + 0.5
                # dist_minus_reg = dist_bias_minus * dist_minus

                dist_bias_plus = -dist_k * dist_plus + 1.25
                # dist_plus_reg = dist_bias_plus * dist_plus

            # update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus + (self.syncmap - center_minus)/dist_minus)
            # update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus + (self.syncmap - center_plus)/dist_plus)
            # update_plus = vplus[:, np.newaxis] * (
            #         (center_plus - self.syncmap)/ dist_plus)  # + (self.syncmap - center_minus)/dist_minus)
            # update_minus = vminus[:, np.newaxis] * (
            #             (center_minus - self.syncmap) / dist_minus)  # + (self.syncmap - center_plus)/dist_plus)
            # Roger
            update_plus = vplus[:, np.newaxis] * ((center_plus - self.syncmap) * dist_bias_plus / dist_plus_reg)
            update_minus = vminus[:, np.newaxis] * ((center_minus - self.syncmap) * dist_bias_minus / dist_minus_reg)
            update = plus_update_freq[i,0]*update_plus - minus_update_freq[i,0]*update_minus
            # self.adaptation_rate = self.adaptation_rate
            # self.previous_syncmap = self.syncmap

            self.syncmap += self.adaptation_rate * update * current_learning_rate

            maximum = self.syncmap.max()
            self.syncmap = self.space_size * self.syncmap / maximum
            # instant history
            # if i % (10000+self.state_memory*self.time_delay) == 0 or i==sequence_size-1:
            if i % (10000) == 0 or i == sequence_size - 1  or histroy==1:
                self.syncmap_history_list.append(self.syncmap+0)

            # moving avg
            if i % self.time_delay == 0 or i==sequence_size-1:
                self.syncmap_movmean.append(self.syncmap+0)
                # self.temp_syncmap_movmean = sum(self.syncmap_movmean)/len(self.syncmap_movmean)
            # self.syncmap_history[:, :, i] = self.syncmap
            # temp_syncmap = self.syncmap + 1 - 1
            if i % 10000 == 0  or i==sequence_size-1 or histroy==1:
                self.temp_syncmap_movmean = sum(self.syncmap_movmean)/len(self.syncmap_movmean)
                self.syncmap_movmean_history_list.append(self.temp_syncmap_movmean+0)
                if histroy==1: #fix a bug
                    histroy=0
            # self.syncmap_history[:, :, i] = self.temp_syncmap_movmean

            # update_ROC detection
            # if i != 0:

            # gap = 5000
            # if i > gap:
            # # if i%gap==0:
            #     delta_update = np.linalg.norm(self.syncmap-self.syncmap_history[:, :, i-gap], axis=1)
            # else: delta_update = 0
            # self.pos_update_ROC_temp_window.append(np.sum(delta_update))

            # self.pos_update_ROC_temp_window.append(np.sum(np.abs(update_plus)))
            # self.pos_update_ROC_temp_window.append(np.sum(update_plus))
            # self.pos_update_ROC_temp_window.append(np.sum(dist_plus_reg*vplus[:,np.newaxis]))

            # self.global_pos_update_ROC.append(sum(self.pos_update_ROC_temp_window)/len(self.pos_update_ROC_temp_window))

            # dist = np.triu(pairwise_distances(self.syncmap))
            # self.pairwise_dist_history[:, i] = dist[np.where(dist != 0)]

            ########## END LOOP ###########
        # After for loop
        pause=1
        # self.global_pos_update_ROC_np = np.array([self.global_pos_update_ROC]).T

    def input(self, x):

        self.inputGeneral(x)

        return


    def organize(self, eps=1):

        self.organized = True
        # self.labels= DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap)
        # self.labels_snn = SNN(neighbor_num=9, min_shared_neighbor_proportion=0.4).fit_predict(self.syncmap)
        self.labels = DBSCAN(eps=self.eps, min_samples=2).fit_predict(self.syncmap)

        return self.labels

    def activate(self, x):
        '''
		Return the label of the index with maximum input value
		'''

        if self.organized == False:
            print("Activating a non-organized SyncMap")
            return

        # maximum output
        max_index = np.argmax(x)

        return self.labels[max_index]

    def plotSequence(self, input_sequence, input_class, filename="plot.png"):

        input_sequence = input_sequence[1:500]
        input_class = input_class[1:500]

        a = np.asarray(input_class)
        t = [i for i, value in enumerate(a)]
        c = [self.activate(x) for x in input_sequence]

        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.k')
        # plt.ylim([-0.01,1.2])

        plt.savefig(filename, quality=1, dpi=300)
        plt.show()
        plt.close()

    def plot(self, color=None, save=True, filename="plot_map1.png"):

        if color is None:
            color = self.labels

        #iclr

        labels_kc = np.array(
            [0, 0, 1, 0, 2, 2, 2, 0, 3, 1, 2, 0, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3])
        # color = labels_kc

        # print(self.syncmap)
        # print(self.syncmap)
        # print(self.syncmap[:,0])
        # print(self.syncmap[:,1])
        if self.dimensions == 2:
            # print(type(color))
            # print(color.shape)
            # ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
            ax = plt.scatter(self.temp_syncmap_movmean[:, 0], self.temp_syncmap_movmean[:, 1], c=color, s=80,
                             cmap=cm.Paired, alpha=0.5)
            # for i in range(len(self.syncmap[:,0])):
            #     plt.text(self.syncmap[i, 0], self.syncmap[i, 1], self.labels[i])
            # ax = plt.xlim(-10.5,10.5)
            # ax = plt.ylim(-10.5,10.5)
            # for i in range(self.temp_syncmap_movmean.shape[0]):
            #     plt.text(self.temp_syncmap_movmean[i, 0]-0.1,
            #              self.temp_syncmap_movmean[i, 1]-0.1,
            #              str(i + 1), fontsize="small")
            # plt.ylim([-11,11])
            # plt.xlim([-11,11])
            # plt.xticks(fontsize=13)
            # plt.yticks(fontsize=13)
            #
            # # karate ICLR
            # label_ = self.labels
            # label_ = labels_kc
            # cdict = {0: ""}
            # for i in range(self.temp_syncmap_movmean.shape[0]):
            #     plt.text(self.temp_syncmap_movmean[i, 0] + np.random.rand() * 0.2,
            #              self.temp_syncmap_movmean[i, 1] + 0.1,
            #              str(i + 1), fontsize="xx-large")








        if self.dimensions == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.syncmap[:, 0], self.syncmap[:, 1], self.syncmap[:, 2], c=color)
        # ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])

        if save == True:
            plt.savefig(filename)

        plt.show()
        plt.close()

    def save(self, filename):
        """save class as self.name.txt"""
        file = open(filename + '.txt', 'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self, filename):
        """try load self.name.txt"""
        file = open(filename + '.txt', 'r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)

    # Roger
    def dropout(self, X_, drop_probability=0.5):
        keep_probability = 1 - drop_probability
        mask = np.random.rand(X_.shape[0], 1) < keep_probability
        #############################
        #  Avoid division by 0 when scaling
        #############################
        # if keep_probability > 0.0:
        # 	scale = (1 / keep_probability)
        # else:
        # 	scale = 0.0
        # mask = np.repeat(mask, self.dimensions, axis=1)
        # return mask * X_  # * scale
        return np.multiply(mask, X_[:, None])

    def random_select_cp(self, X_, num_pick=3):

        if X_.sum()<num_pick:
            temp_num_pick = X_.sum()
        else:
            temp_num_pick = num_pick        # else:
        #     # if X_.size * self.dropout_rate <= state_memory
        #     num_pick = math.ceil(X_.size * (1 - self.dropout_rate))
        probability = X_ * (1 / X_.sum())
        temp = np.arange(X_.size)
        temp = np.random.choice(temp, temp_num_pick, replace=False, p=probability)
        mask = np.zeros(X_.shape, dtype=bool)
        mask[temp] = True
        output = mask * X_
        return output

    def prob_select_cp_v1(self, X_, num_pick=3):

        if X_.sum()<num_pick:
            temp_num_pick = X_.sum()
        else:
            temp_num_pick = num_pick
        # else:
        #     # if X_.size * self.dropout_rate <= state_memory
        #     num_pick = math.ceil(X_.size * (1 - self.dropout_rate))
        probability = X_ * (1 / X_.sum())
        temp = np.arange(X_.sum())+1
        temp_sum_denominator = temp.sum()
        probability[X_] = temp/temp_sum_denominator
        temp = np.arange(X_.size)
        temp = np.random.choice(temp, temp_num_pick, replace=False, p=probability)
        mask = np.zeros(X_.shape, dtype=bool)
        mask[temp] = True
        output = mask * X_
        return output

    def random_select_cn(self, X_, num_pick=2):
        # try:
        #     X_.sum() < num_pick
        # except ValueError:
        #     print("num_pick grater than vminus")
        if self.stochastic_pick:
            temp_num_pick = num_pick
        else:
            # if X_.size * self.dropout_rate <= state_memory
            temp_num_pick = math.ceil(X_.size * (1 - self.dropout_rate))
        probability = X_ * (1 / X_.sum())
        temp = np.arange(X_.size)
        temp = np.random.choice(temp, temp_num_pick, replace=False, p=probability)
        mask = np.zeros(X_.shape, dtype=bool)
        mask[temp] = True
        output = mask * X_
        return output

    def prob_select_cn(self, X_, dist_matrix, state_memory=2):
        if self.stochastic_pick:
            num_pick = state_memory - 1
        else:
            num_pick = math.ceil(X_.size * (1 - self.dropout_rate)) - 1  # x_.size-statememory ??

        probability = X_ * (1 / X_.sum())
        temp = np.arange(X_.size)
        temp = np.random.choice(temp, 1, replace=False, p=probability)
        mask = np.zeros(X_.shape, dtype=bool)
        mask[temp] = True
        dist_list = dist_matrix[temp[0], :]
        dist_list[mask] = 1
        dist_list_reciprocal = np.reciprocal(dist_list)
        dist_list_reciprocal[mask] = 0
        dist_list_reciprocal[~X_] = 0
        dist_list_reciprocal_cumsum = dist_list_reciprocal.cumsum()
        dist_denominator = dist_list_reciprocal_cumsum[-1]
        probability = dist_list_reciprocal / dist_denominator

        temp2 = np.arange(X_.size)
        temp2 = np.random.choice(temp2, num_pick, replace=False, p=probability)
        mask = np.zeros(X_.shape, dtype=bool)
        mask[temp] = True
        mask[temp2] = True

        output = mask * X_
        return output

    def pre_running(self, x):
        plus = x > 0.1
        minus = ~ plus

        sequence_size = x.shape[0]

        for i in tqdm(range(sequence_size)):
            vplus = plus[i, :]
            vminus = minus[i, :]
            self.counter_plus = self.counter_plus + vplus
            self.counter_minus = self.counter_minus + vminus

        self.counter_plus = np.round(self.counter_plus/2)
        self.counter_minus = np.round(self.counter_minus/2)
        self.max_time_visit = self.counter_plus.max()
        # temp_ratio = self.counter_plus/self.max_time_visit
        self.state_visit_ratio = self.max_time_visit / self.counter_plus
        temp1=1

    def state_memory_generalization(self, iter, plus, pre_plus,
                                    past_state_num, past_state_set):

        if (iter % self.time_delay) == 0:
            # vplus = plus[i, :]
            for j in range(past_state_num - 1):
                past_state_set[(past_state_num - 1) - (j + 1), :] = plus[iter -
                                                            ((self.time_delay) * j + 1), :]
            pre_plus = past_state_set.sum(axis=0, dtype=bool)
            vplus = plus[iter, :]
            vplus = np.logical_or(pre_plus, plus[iter, :])
        else:
            vplus = plus[iter, :]
            vplus = np.logical_or(pre_plus, plus[iter, :])

        return vplus, pre_plus

