#ifndef PRUNING_H
#define PRUNING_H

#include "darknet.h"

/*
 *  Apply soft filter pruning on a network (https://arxiv.org/pdf/1808.06866.pdf)
 *  net:        Network instance
 *  percentile: Percentage (value between 0 and 1) of how many filters should be pruned in each conv layer
 */
void prune_network(network *net);

/*
 *  Apply soft filter pruning on a an array of networks (https://arxiv.org/pdf/1808.06866.pdf)
 *  nets:       Array of network structs
 *  n:          Length of the network struct array
 *  percentile: Percentage (value between 0 and 1) of how many filters should be pruned in each conv layer
 */
void prune_networks(network **nets, int n);

#endif /* PRUNING_H */
