/**
 * AND Gate - Simple example
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NNN_IMPLEMENTATION
#include "../nnn.h"

float train_data[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};
#define DATA_SIZE 4

int main()
{
    srand(time(0));

    int epochs = 1000;
    float lr = 0.1f;

    // 2 inputs -> 1 output
    size_t layers[] = {2, 1};
    Network *net = network_alloc(2, layers);
    network_randomize(net, -1, 1);

    Mat *target = mat_alloc(1, 1);

    printf("Training AND gate for %d epochs...\n\n", epochs);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float loss = 0;
        for (int i = 0; i < DATA_SIZE; i++)
        {
            net->a[0]->p_data[0] = train_data[i][0];
            net->a[0]->p_data[1] = train_data[i][1];
            target->p_data[0] = train_data[i][2];

            forward(net);
            float diff = net->a[1]->p_data[0] - target->p_data[0];
            loss += diff * diff;
            backward(net, target, lr);
        }

        if ((epoch + 1) % 200 == 0)
            printf("Epoch %4d | Loss: %.6f\n", epoch + 1, loss / DATA_SIZE);
    }

    printf("\nResults:\n");
    for (int i = 0; i < DATA_SIZE; i++)
    {
        net->a[0]->p_data[0] = train_data[i][0];
        net->a[0]->p_data[1] = train_data[i][1];
        forward(net);
        printf("  %.0f AND %.0f = %.3f (expected: %.0f)\n",
               train_data[i][0], train_data[i][1],
               net->a[1]->p_data[0], train_data[i][2]);
    }

    mat_free(target);
    network_free(net);
    return 0;
}
