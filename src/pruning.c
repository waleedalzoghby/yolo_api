#include "pruning.h"
#include <math.h>

struct FilterInfo
{
    float norm;     // filter norm
    int idx;        // filter index
};

static int sfp_compare(const void *va, const void *vb)
{
    const struct FilterInfo *a = va;
    const struct FilterInfo *b = vb;
    float diff = a->norm - b->norm;

    if (diff < 0.0)
        return -1;

    return 1;
}

void prune_network(network *net)
{
    struct FilterInfo *filter_info;
    int i, j, num, num_weights_per_filter;

    // loop over every layer
    for (i=0; i<net->n; ++i) {

        // process only convolutional layers that are marked with the prune flag
        layer l = net->layers[i];
        if (l.type != CONVOLUTIONAL || !l.prune)
            continue;

        num_weights_per_filter = l.c*l.size*l.size;
        filter_info = malloc(l.n * sizeof(struct FilterInfo));

#ifdef GPU
        // fetch weights from GPU
        if (net->gpu_index >= 0) {
            cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
        }
#endif
        // compute L2 norm for each convolutional filter
        // loop over each filter
        for (j=0; j<l.n; ++j) {
            filter_info[j].norm = mag_array(&l.weights[j * num_weights_per_filter], num_weights_per_filter);
            filter_info[j].idx = j;
        }

        // sort L2 norms in ascending manner
        qsort(filter_info, l.n, sizeof(struct FilterInfo), sfp_compare);

        num = roundf(l.n * net->prune_rate);
        if (num > l.n)
            num = l.n;

        // force all weights in certain channels to zero based on sorted L2 norms
        // loop over prune_rate of filters
        for (j=0; j<num; ++j) {
            memset(&l.weights[filter_info[j].idx * num_weights_per_filter], 0, num_weights_per_filter * sizeof(float));
        }

        free(filter_info);

#ifdef GPU
        // push modified weights to GPU
        if (net->gpu_index >= 0) {
            cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        }
#endif
    }
}

void prune_networks(network **nets, int n)
{
    int i;

    for (i=0; i<n; ++i) {
        prune_network(nets[i]);
    }
}
