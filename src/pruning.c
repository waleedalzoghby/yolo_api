#include "pruning.h"
#include <math.h>

struct FilterInfo
{
    float norm;     // filter norm
    int idx;        // filter index
};

struct PruningArgs
{
    network *net;
    float percentile;
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

static void *prune_network_thread(void *varg)
{
    struct PruningArgs *arg = varg;
    prune_network(arg->net, arg->percentile);
    return NULL;
}

void prune_network(network *net, float percentile)
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

        num = roundf(l.n * percentile);
        if (num > l.n)
            num = l.n;

        // force all weights in certain channels to zero based on sorted L2 norms
        // loop over percentile of filters
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

void prune_networks(network **nets, int n, float percentile)
{
    int i;
    struct PruningArgs *args = malloc(n * sizeof(struct PruningArgs));
    pthread_t *threads = malloc(n * sizeof(struct PruningArgs));

    // start pruning each network in seperate thread
    for (i=0; i<n; ++i) {
        args[i].net = nets[i];
        args[i].percentile = percentile;
        if (pthread_create(&threads[i], NULL, prune_network_thread, &args[i]))
            error("Thread creation failed");
    }

    // join threads
    for (i=0; i<n; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(args);
    free(threads);
}
