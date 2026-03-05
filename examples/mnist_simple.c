/**
 * MNIST Simple Example
 * Loads MNIST dataset, trains for X epochs, prints final loss
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define NNN_IMPLEMENTATION
#include "../nnn.h"

#define IMAGE_SIZE 784  // 28x28
#define NUM_CLASSES 10

typedef struct {
    float pixels[IMAGE_SIZE];
    uint8_t label;
} MnistSample;

typedef struct {
    MnistSample *samples;
    int count;
} MnistDataset;

uint32_t read_be32(FILE *f)
{
    uint8_t buf[4];
    fread(buf, 1, 4, f);
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

MnistDataset *mnist_load(const char *images_path, const char *labels_path)
{
    FILE *fimg = fopen(images_path, "rb");
    FILE *flbl = fopen(labels_path, "rb");
    if (!fimg || !flbl)
    {
        fprintf(stderr, "Cannot open MNIST files\n");
        if (fimg) fclose(fimg);
        if (flbl) fclose(flbl);
        return NULL;
    }

    // Read headers
    if (read_be32(fimg) != 0x803 || read_be32(flbl) != 0x801)
    {
        fprintf(stderr, "Invalid MNIST format\n");
        fclose(fimg); fclose(flbl);
        return NULL;
    }

    int img_count = read_be32(fimg);
    int lbl_count = read_be32(flbl);
    read_be32(fimg); // rows
    read_be32(fimg); // cols

    if (img_count != lbl_count)
    {
        fprintf(stderr, "Image/label count mismatch\n");
        fclose(fimg); fclose(flbl);
        return NULL;
    }

    MnistDataset *ds = malloc(sizeof(MnistDataset));
    ds->count = img_count;
    ds->samples = malloc(sizeof(MnistSample) * img_count);

    uint8_t buf[IMAGE_SIZE];
    for (int i = 0; i < img_count; i++)
    {
        fread(buf, 1, IMAGE_SIZE, fimg);
        for (int j = 0; j < IMAGE_SIZE; j++)
            ds->samples[i].pixels[j] = buf[j] / 255.0f;
        fread(&ds->samples[i].label, 1, 1, flbl);
    }

    fclose(fimg);
    fclose(flbl);
    printf("Loaded %d samples\n", ds->count);
    return ds;
}

void mnist_free(MnistDataset *ds)
{
    if (ds)
    {
        free(ds->samples);
        free(ds);
    }
}

int argmax(Mat *m)
{
    int idx = 0;
    float max = m->p_data[0];
    for (int i = 1; i < m->rows; i++)
    {
        if (m->p_data[i] > max)
        {
            max = m->p_data[i];
            idx = i;
        }
    }
    return idx;
}

float compute_accuracy(Network *net, MnistDataset *ds, int max_samples)
{
    int count = (max_samples < ds->count) ? max_samples : ds->count;
    int correct = 0;
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
            net->a[0]->p_data[j] = ds->samples[i].pixels[j];
        forward(net);
        if (argmax(net->a[net->n_layers - 1]) == ds->samples[i].label)
            correct++;
    }
    return (float)correct / count * 100.0f;
}

int main(int argc, char *argv[])
{
    srand(time(0));

    int epochs = 5;
    int train_samples = 10000;
    float lr = 0.001f;

    if (argc > 1) epochs = atoi(argv[1]);
    if (argc > 2) train_samples = atoi(argv[2]);

    printf("MNIST Training - Epochs: %d, Samples: %d, LR: %.4f\n\n", epochs, train_samples, lr);

    MnistDataset *ds = mnist_load("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");
    if (!ds) return 1;

    if (train_samples > ds->count) train_samples = ds->count;

    // Network: 784 -> 128 -> 64 -> 10
    size_t layers[] = {IMAGE_SIZE, 128, 64, NUM_CLASSES};
    Network *net = network_alloc(4, layers);
    network_randomize(net, -1, 1);

    Mat *target = mat_alloc(NUM_CLASSES, 1);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0;

        for (int i = 0; i < train_samples; i++)
        {
            MnistSample *s = &ds->samples[i];

            for (int j = 0; j < IMAGE_SIZE; j++)
                net->a[0]->p_data[j] = s->pixels[j];

            mat_init(target, 0);
            target->p_data[s->label] = 1.0f;

            forward(net);

            Mat *output = net->a[net->n_layers - 1];
            for (int j = 0; j < NUM_CLASSES; j++)
            {
                float diff = output->p_data[j] - target->p_data[j];
                total_loss += diff * diff;
            }

            backward(net, target, lr);
        }

        float avg_loss = total_loss / train_samples;
        float acc = compute_accuracy(net, ds, 1000);
        printf("Epoch %d/%d | Loss: %.6f | Accuracy: %.2f%%\n", epoch + 1, epochs, avg_loss, acc);
    }

    printf("\n=== Training Complete ===\n");
    printf("Final Accuracy: %.2f%%\n", compute_accuracy(net, ds, train_samples));

    printf("\nSample predictions:\n");
    for (int i = 0; i < 10; i++)
    {
        int idx = rand() % train_samples;
        MnistSample *s = &ds->samples[idx];

        for (int j = 0; j < IMAGE_SIZE; j++)
            net->a[0]->p_data[j] = s->pixels[j];
        forward(net);

        int pred = argmax(net->a[net->n_layers - 1]);
        printf("  [%d] predicted: %d, actual: %d %s\n",
               idx, pred, s->label, pred == s->label ? "" : "WRONG");
    }

    mat_free(target);
    network_free(net);
    mnist_free(ds);
    return 0;
}
