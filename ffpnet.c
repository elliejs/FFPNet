#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <byteswap.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

#include "ffpnet.h"

bool main(int argc, byte_t* argv[]) {
    if (argc < 3) {
      printf("try with arguments: layersize0 ... layersizeN eta\nIt is highly suggested that layersizeN == 10 (so that the last layer captures perfectly 0..9)\n");
      return failure;
    }
    srand(time(NULL));

    //generic variables
    int correctGuesses = 0;

    net_t net;
    net.eta = strtod(argv[argc - 1], NULL);

    //start file manipulation
    FILE *fData, *fLabel;
    uint32_t row, col, imageSize, imageCount, labelCount; //image variable declarations

    fData  = fopen("./train-images", "rb");
    fLabel = fopen("./train-labels", "rb");

    //file manipulation is always finnicky, I want to make sure these are the right size ints.

    fseek(fLabel, 4, SEEK_SET);
    fread(&labelCount, 4, 1, fLabel);

    fseek(fData, 4, SEEK_SET);
    fread(&imageCount, 4, 1, fData);
    fread(&row, 4, 1, fData);
    fread(&col, 4, 1, fData);

    labelCount = __bswap_32 (labelCount);
    imageCount = __bswap_32 (imageCount);

    row = __bswap_32 (row);
    col = __bswap_32 (col);

    imageSize = row * col;
    //end file manipulation

    //double check that the labels match up with the images, at least numerically
    if (labelCount != imageCount) {
        printf("%d:%d", imageCount, labelCount);
        return failure; //Maybe FFPNet isn't reading the right manual.
    }

    //start layer accretion from stdin
    net.layerCount = argc - 1;
    net.layerSizes = malloc(sizeof(int) * net.layerCount);
    net.layerSizes[0] = imageSize;

    for (int i = 1; i < net.layerCount; i++) {
        net.layerSizes[i] = atoi(argv[i]);
    }
    //end layer accretion from stdin

    //start imageBundle accretion
    image_t* imageBundle = malloc(sizeof(image_t) * imageCount); //declare and init the whole input of images

    for (int i = 0; i < imageCount; i++) {
        imageBundle[i].data = malloc(sizeof(byte_t) * imageSize);

        fread(imageBundle[i].data, 1, imageSize, fData);
        fread(&imageBundle[i].value, 1, 1, fLabel);
    }
    //end imageBundle accretion

    makeNet(&net);

    int thisBatch = 0;
    //now for the fun stuff!
    for (int i = 0; i < imageCount; i++) {

//        prints the input image
//        for (int j = 0; j < row; j++) {
//            for (int k = 0; k < col; k++) {
//                imageBundle[i].data[k + j * col] ? printf("%c", '#') : printf("%c", ' ');
//                imageBundle[i].data[k + j * col] ? printf("%c", '#') : printf("%c", ' ');
//            }
//            printf("\n");
//        }
//        printf("\n");

        bool successfullyProcessed = processImage(&net, &imageBundle[i]);

        alterNet(&net);

        correctGuesses += successfullyProcessed;
        thisBatch += successfullyProcessed;

        if(i % 100 == 0) {
            printf("percent correct So Far:%0.2lf  this batch: %0.2lf\n", (double) correctGuesses / (double) i, (double) thisBatch / (double) 100);
            thisBatch = 0;
        }
        // free(imageBundle[i].data);
    }
    free(imageBundle);

    printf("\npercent correct (training): %0.2f\n\n", (double) correctGuesses / (double) imageCount);

    //export net for use later. This program creates a new net and would otherwise destroy it on exit. ExportNet creates persistence.
    exportNet(&net);

    //testing
    fclose(fData);
    fclose(fLabel);
    fData  = fopen("./test-images", "rb");
    fLabel = fopen("./test-labels", "rb");
    //file manipulation is always finnicky, I want to make sure these are the right size ints.
    fseek(fLabel, 4, SEEK_SET);
    fread(&labelCount, 4, 1, fLabel);

    fseek(fData, 4, SEEK_SET);
    fread(&imageCount, 4, 1, fData);
    fread(&row, 4, 1, fData);
    fread(&col, 4, 1, fData);

    labelCount = __bswap_32 (labelCount);
    imageCount = __bswap_32 (imageCount);

    row = __bswap_32 (row);
    col = __bswap_32 (col);

    imageSize = row * col;
    //end file manipulation

    //double check that the labels match up with the images, at least numerically
    if (labelCount != imageCount) {
        printf("%d:%d", imageCount, labelCount);
        return failure; //Maybe FFPNet isn't reading the right manual.
    }


    image_t proc = {data: malloc(sizeof(byte_t) * imageSize)};

    correctGuesses = 0;
    for (int i = 0; i < imageCount; i++) {
      fread(proc.data, 1, imageSize, fData);
      fread(&(proc.value), 1, 1, fLabel);
      proc.data = malloc(sizeof(byte_t) * imageSize);
      correctGuesses += processImage(&net, &proc);
    }

    printf("percent correct (testing): %.2f\n(%d out of %d)\n", (double)correctGuesses / (double)imageCount, correctGuesses, imageCount);


    //clean up
    fclose(fData);
    fclose(fLabel);
    freeNet(&net);

    return success;
}
