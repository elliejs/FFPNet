#ifndef ffpnet
#define ffpnet

typedef unsigned char byte_t;
typedef enum { false, true, failure = 0, success = 1 } bool; //everyone wants booleans, except C89

typedef struct {
    int addr;
    int val;
} index_t;
typedef struct {
    byte_t* data;
    byte_t value;
} image_t;

typedef struct {
    double bias;
    double* weightBundle;
    int activation;
    double error;
} node_t;
typedef struct {
    node_t* nodeBundle;
} layer_t;
typedef struct {
    layer_t* layerBundle;
    int layerCount;
    int* layerSizes;        
    double eta; // η (Greek letter eta) is the variable for network learning rate
} net_t;


bool processImage   (net_t* net, image_t* inputImage);

void makeNet        (net_t* net);
void alterNet       (net_t* net);
void exportNet       (net_t* net);
void freeNet        (net_t* net);

double drand        (double range);
int ReLU            (double in);
index_t maxBundle   (int* inBundle, int inBundleSize);
double costFunction(node_t* node);


bool processImage (net_t* net, image_t* inputImage) {
    //set first layer activations from the image into the net
    for (int i = 0; i < net->layerSizes[0]; i++) {
        net->layerBundle[0].nodeBundle[i].activation = (int) inputImage->data[i];
    }
    
    //activate all net-based nodes
    for(int i = 1; i < net->layerCount; i++) {
        for(int j = 0; j < net->layerSizes[i]; j++) {
            double prevActivationBundle = 0;
            //get activations for current node from a loop of previous nodeBundle and nodeWeights.
            for(int k = 0; k < net->layerSizes[i - 1]; k++) {
                //printf("%d: %d\n", k, net->layerBundle[i - 1].nodeBundle[k].activation);
                prevActivationBundle += net->layerBundle[i - 1].nodeBundle[k].activation *
                                        net->layerBundle[i].nodeBundle[j].weightBundle[k];
                if (prevActivationBundle > DBL_MAX - 100) {
                    printf("warn: FFP-Net can't count that high :(");
                    exit(EXIT_FAILURE); //yikes. Hope we never trigger this one
                }
            }
            //add nodeBias to the looped total above
            net->layerBundle[i].nodeBundle[j].activation = ReLU(prevActivationBundle + net->layerBundle[i].nodeBundle[j].bias);
        }
    }
    
    int* guessBundle = malloc(sizeof(int) * net->layerSizes[net->layerCount - 1]);
    for (int i = 0; i < net->layerSizes[net->layerCount - 1]; i++) {
        guessBundle[i] = net->layerBundle[net->layerCount - 1].nodeBundle[i].activation;
    }
        
    free(inputImage->data);
    
    //calculate each node's error
    //last layer's error
    // δL = ∇aC ⊙ σ′(zL)
    for (int i = 0; i < net->layerSizes[net->layerCount - 1]; i++) {
        if (i != (int) inputImage->value) {
            net->layerBundle[net->layerCount - 1].nodeBundle[i].error = -1 * //flip for 0 - z (so we can do z - 0 math)
                                                                        net->eta *               
                                                                        net->layerBundle[net->layerCount - 1].nodeBundle[i].activation; //maybe use better cost function
                                                                        //(net->layerBundle[net->layerCount - 1].nodeBundle[i].activation ? 1 : 0);
        }
    }
    net->layerBundle[net->layerCount - 1].nodeBundle[(int) inputImage->value].error = 1; //we always like our "right" answer getting bigger
    
    //δl=((w^(l+1))^T * δ^(l+1)) ⊙ σ′(zl)
    if (net->layerCount > 2) {
        for (int i = net->layerCount - 2; i > 0; i--) {
            for (int j = 0; j < net->layerSizes[i]; j++) {
                double midLayerErr = 0;
                for (int k = 0; k < net->layerSizes[i + 1]; k++) {
                    midLayerErr += net->layerBundle[i + 1].nodeBundle[k].weightBundle[j] * net->layerBundle[i + 1].nodeBundle[k].error;
                }
                net->layerBundle[i].nodeBundle[j].error = midLayerErr * (net->layerBundle[i].nodeBundle[j].activation ? 1 : 0) * net->eta;
            }
        }
    }

    if (inputImage->value == maxBundle(guessBundle, net->layerSizes[net->layerCount - 1]).addr) {
        return success;
    } else {
        return failure;
    }
}

void makeNet (net_t* net) {
    
    net->layerBundle = malloc(sizeof(layer_t) * net->layerCount);
    
    //malloc loop
    for (int i = 0; i < net->layerCount; i++) {
        net->layerBundle[i].nodeBundle = malloc(sizeof(node_t) * net->layerSizes[i]);
        for (int j = 0; j < net->layerSizes[i]; j++) {
            //no weights for layer 1
            if (i) net->layerBundle[i].nodeBundle[j].weightBundle = malloc(sizeof(double) * net->layerSizes[i - 1]);
            //don't blame me, it's "economic". Basically, as long as i >= 1, we can do weights stuff. also protects i - 1 from weird dereference scariness.
        }
    }
    
    //set loop
    for (int i = 1; i < net->layerCount; i++) {
        for (int j = 0; j < net->layerSizes[i]; j++) { //no weights for layer 1
            for (int k = 0; k < net->layerSizes[i - 1]; k++) {
                double weight = drand(5);
                //printf("%d:%d:%d  %.2f\n", i, j, k, weight);
                net->layerBundle[i].nodeBundle[j].weightBundle[k] = weight; //set weights
            }
            net->layerBundle[i].nodeBundle[j].bias = drand(5); //set biases
        }
    }
    
    //The malloc loop needs to clear a space for the input layer of the net, where the image gets vectorized into, but the set loop can ignore that layer (by starting at i = 1) because it isn't part of the net's "thought" process. [the giant polynomial].
    
    //simplification: f(x) = 5x + 3 is analogous to -> net(image) = weight*node+bias. just a lot more terms.
    //f(x) is NOT *part* of the function 5x + 3, but it used when finding the answer to the function, therefore, it must be available, but doesn't need to have a value.
    
    return;
}

void alterNet (net_t* net) {
    for (int i = 1; i < net->layerCount; i++) {
        for (int j = 0; j < net->layerSizes[i]; j++) { //no weights for layer 1
            for (int k = 0; k < net->layerSizes[i - 1]; k++) {
                net->layerBundle[i].nodeBundle[j].weightBundle[k] += net->eta * 
                                                                     net->layerBundle[i-1].nodeBundle[k].activation *
                                                                     net->layerBundle[i].nodeBundle[j].error;
            }
            net->layerBundle[i].nodeBundle[j].bias += net->eta * 
                                                      net->layerBundle[i].nodeBundle[j].error;
        }
    }
    return;
}

void exportNet (net_t* net) {
    FILE *fo;
    fo = fopen("FFPNet.net", "wb");
    
    fwrite(&net->layerCount, sizeof(int), 1, fo);
    
    for (int i = 0; i < net->layerCount; i++) {
        fwrite(&net->layerSizes[i], sizeof(int), 1, fo);
        
        for (int j = 1; j < net->layerSizes[i]; j++) {
            fwrite(&net->layerBundle[i].nodeBundle[j].bias, sizeof(double), 1, fo);
            
            for (int k = 0; k < net->layerSizes[i - 1]; k++) {
                fwrite(&net->layerBundle[i].nodeBundle[j].weightBundle[k], sizeof(double), 1, fo);
            }
        }
    }
    fclose(fo);
    return;
}

void freeNet (net_t* net) {
    
    //malloc loop BUT IN REVERSE PRIORITY!
    for (int i = 0; i < net->layerCount; i++) {
        for (int j = 0; j < net->layerSizes[i]; j++) { //no weights for layer 1
            if (i) free(net->layerBundle[i].nodeBundle[j].weightBundle);
        }
        free(net->layerBundle[i].nodeBundle);
    }
    free(net->layerBundle);
    free(net->layerSizes);
    return;
}

double drand (double range) {
    int flipbit = rand() % 2 ? -1 : 1;
    return flipbit * (((double)rand() * range) / (double)RAND_MAX);
}

int ReLU (double in) {
    if (in <= 0) {
        return 0;
    } else {
        return (int)(in + 0.5);
    }
}

index_t maxBundle (int* inBundle, int inBundleSize) {
    int maxBundleVal = 0;
    int maxBundleAddr;
    index_t index;
    for (int i = 0; i < inBundleSize; i++) {
        if (inBundle[i] > maxBundleVal) {
            maxBundleVal = inBundle[i];
            maxBundleAddr = i;
        }
    }
    index.addr = maxBundleAddr;
    index.val = maxBundleVal;
    return index;
}

double costFunction(node_t* node) {
    
}

#endif
