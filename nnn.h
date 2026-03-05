/**
 * Not nut networks
 * Simple NN header only library
 */
#ifndef NNN_H_
# define NNN_H_

# include <stdio.h>
# include <math.h>

# ifndef NNN_ALLOC
#  include <stdlib.h>
#  define NNN_ALLOC(size) malloc(size)
# endif /// NNN_ALLOC

# ifndef NNN_FREE
#  include <stdlib.h>
#  define NNN_FREE(ptr) free(ptr)
# endif /// NNN_FREE

#ifdef MAT_BOUNDS_CHECK
#define CHECK_BOUNDS(m,x,y) \
    if ((x)<0||(x)>=(m)->rows||(y)<0||(y)>=(m)->columns) return NULL
#else
#define CHECK_BOUNDS(m,r,c)
#endif // CHECK_BOUNDS

#ifdef DEBUG
#define debug(x) x
#else
#define debug(x)
#endif

/**
 * Matrix
 */
typedef struct Mat
{
	size_t	rows;
	size_t	columns;
	float	*p_data;
}	Mat;

/**
 * Network: array of layers
 */
typedef struct Network
{
    size_t  n_layers;
    Mat     **a;        // attivazioni: n_layers elementi
    Mat     **z;        // pre-attivazione: n_layers - 1 elementi (non serve per input)
    Mat     **weights;
    Mat     **biases;
}	Network;

// Matrix
Mat			        *mat_alloc(size_t rows, size_t columns);
void                mat_free(Mat *mat);
void			    mat_mul(Mat *out, const Mat *a, const Mat *b);
void                mat_add(Mat *out, const Mat *a, const Mat *b);
void			    mat_print(const Mat *mat);
void                mat_init(Mat *mat, float value);
static inline float *mat_at(Mat *mat, size_t x, size_t y);
void                mat_transpose(Mat *out, const Mat *mat);
void                mat_sub(Mat *out, const Mat *a, const Mat *b);
void                mat_hadamard(Mat *out, const Mat *a, const Mat *b);

// Network
Network             *network_alloc(size_t n_layers, size_t neurons[]);
void                network_free(Network *net);
void                network_print(const Network *net);
void                forward(Network *net);
void                network_randomize(Network *network, float min, float max);
float               ReLu(float z);
float               ReLu_derivative(float z);
float               randf(float min, float max);
void                backward(Network *net, Mat *target, float learning_rate);

#endif // NNN_H_

# ifdef NNN_IMPLEMENTATION

void forward(Network *net)
{
    for (int i = 0; i < net->n_layers - 1; i++)
    {
        // z[i] = W[i] * a[i] + b[i]
        mat_init(net->z[i], 0);
        mat_mul(net->z[i], net->weights[i], net->a[i]);
        mat_add(net->z[i], net->z[i], net->biases[i]);

        // a[i+1] = activation(z[i])
        for (int j = 0; j < net->z[i]->rows; j++)
            net->a[i + 1]->p_data[j] = ReLu(net->z[i]->p_data[j]);
    }
}

void backward(Network *net, Mat *target, float learning_rate)
{
    size_t L = net->n_layers - 1;   // Indice Output
    Mat *output = net->a[L];        // output della rete)
    Mat *z_last = net->z[L - 1];    // ultimo z (pre-attivazione)

    // Alloca delta con la stessa shape dell'output
    Mat *delta = mat_alloc(output->rows, 1);

    // delta = output - target
    mat_sub(delta, output, target);

    // delta = delta ⊙ ReLU'(z_last)
    for (size_t j = 0; j < delta->rows; j++)
        delta->p_data[j] *= ReLu_derivative(z_last->p_data[j]);
    for (int i = L - 1; i >= 0; i--)
    {
        Mat *a_T = mat_alloc(1, net->a[i]->rows);
        mat_transpose(a_T, net->a[i]);

        Mat *grad_w = mat_alloc(net->weights[i]->rows, net->weights[i]->columns);
        mat_init(grad_w, 0);  // azzera prima della moltiplicazione
        mat_mul(grad_w, delta, a_T);
        Mat *delta_new = NULL;
        if (i > 0)
        {
            Mat *w_T = mat_alloc(net->weights[i]->columns, net->weights[i]->rows);
            mat_transpose(w_T, net->weights[i]);

            delta_new = mat_alloc(net->a[i]->rows, 1);
            mat_init(delta_new, 0);
            mat_mul(delta_new, w_T, delta);

            // Moltiplica per la derivata dell'attivazione del layer precedente
            for (size_t j = 0; j < delta_new->rows; j++)
                delta_new->p_data[j] *= ReLu_derivative(net->z[i - 1]->p_data[j]);

            mat_free(w_T);
        }
        // Aggiorna pesi: W[i] = W[i] - lr * grad_w
        for (size_t j = 0; j < net->weights[i]->rows * net->weights[i]->columns; j++)
            net->weights[i]->p_data[j] -= learning_rate * grad_w->p_data[j];

        // Aggiorna bias: b[i] = b[i] - lr * delta
        for (size_t j = 0; j < net->biases[i]->rows; j++)
            net->biases[i]->p_data[j] -= learning_rate * delta->p_data[j];
        mat_free(a_T);
        mat_free(grad_w);
        mat_free(delta);

        // Passa al delta del layer precedente
        delta = delta_new;
    }
    if (delta != NULL)
        mat_free(delta);
}

float ReLu(float z)
{
    return (z < 0 ? 0.01f * z : z);  // Leaky ReLU
}

float ReLu_derivative(float z)
{
    return (z > 0 ? 1.0f : 0.01f);
}

float randf(float min, float max)
{
    return min + (float)rand() / RAND_MAX * (max - min);
}

void network_randomize(Network *network, float min, float max)
{
    (void)min; (void)max;  // unused, using He initialization instead
    for (size_t i = 0; i < network->n_layers - 1; i++)
    {
        Mat *w = network->weights[i];
        float scale = sqrtf(2.0f / w->columns);  // He initialization
        for (size_t j = 0; j < w->rows * w->columns; j++)
            w->p_data[j] = randf(-1, 1) * scale;

        Mat *b = network->biases[i];
        for (size_t j = 0; j < b->rows * b->columns; j++)
            b->p_data[j] = 0.0f;  // biases start at 0
    }
}

/**
 * Allocate a network
 */
// TODO: check allocations
Network *network_alloc(size_t n_layers, size_t neurons[])
{
    Network *network = (Network *)NNN_ALLOC(sizeof(Network));

    network -> n_layers = n_layers;
    network -> a       = (Mat **)NNN_ALLOC(sizeof(Mat *) * n_layers);
    network -> z       = (Mat **)NNN_ALLOC(sizeof(Mat *) * (n_layers - 1));
    network -> weights = (Mat **)NNN_ALLOC(sizeof(Mat *) * (n_layers - 1));
    network -> biases  = (Mat **)NNN_ALLOC(sizeof(Mat *) * (n_layers - 1));

    for (int i = 0; i < n_layers - 1; i++)
    {
        network -> a[i]       = mat_alloc(neurons[i], 1);
        network -> z[i]       = mat_alloc(neurons[i + 1], 1);
        network -> weights[i] = mat_alloc(neurons[i + 1], neurons[i]);
        network -> biases[i]  = mat_alloc(neurons[i + 1], 1);
    }
    network -> a[n_layers - 1] = mat_alloc(neurons[n_layers - 1], 1);

    return network;
}

/**
 * Free a matrix
 */
void mat_free(Mat *mat)
{
	if (!mat)
		return;
	NNN_FREE(mat->p_data);
	NNN_FREE(mat);
}


/**
 * Init a matrix with a value
 */
void mat_init(Mat *mat, float value)
{
    for (int i = 0; i < mat->rows * mat->columns; i++)
        mat->p_data[i] = value;
}

/**
 * Matrix addition
 */
void mat_add(Mat *out, const Mat *a, const Mat *b)
{
    //TODO: abstract matrix check
    if ((a->columns != b->columns) || (a->rows != b->rows))
    {
        fprintf(stderr, "Matrices not compatible");
        return ;
    }
    if (out->columns != a->columns || out->rows != a->rows)
    {
        fprintf(stderr, "Matrices not compatible");
        return ;
    }
    for (int i = 0; i < a->rows * a->columns; i++)
        out->p_data[i] = a->p_data[i] + b->p_data[i];
}

/**
 * Matrix subtraction
 */
void mat_sub(Mat *out, const Mat *a, const Mat *b)
{
    if ((a->columns != b->columns) || (a->rows != b->rows))
    {
        fprintf(stderr, "mat_sub: matrices not compatible\n");
        return;
    }
    if (out->columns != a->columns || out->rows != a->rows)
    {
        fprintf(stderr, "mat_sub: output not compatible\n");
        return;
    }
    for (size_t i = 0; i < a->rows * a->columns; i++)
        out->p_data[i] = a->p_data[i] - b->p_data[i];
}

/**
 * Hadamard product
 */
void mat_hadamard(Mat *out, const Mat *a, const Mat *b)
{
    if ((a->columns != b->columns) || (a->rows != b->rows))
    {
        fprintf(stderr, "mat_hadamard: matrices not compatible\n");
        return;
    }
    if (out->columns != a->columns || out->rows != a->rows)
    {
        fprintf(stderr, "mat_hadamard: output not compatible\n");
        return;
    }
    for (size_t i = 0; i < a->rows * a->columns; i++)
        out->p_data[i] = a->p_data[i] * b->p_data[i];
}

/**
 * Matrix multiplication (cache-friendly i-k-j order)
 */
void mat_mul(Mat *out, const Mat *a, const Mat *b)
{
	if (a->columns != b->rows)
	{
		fprintf(stderr, "Matrices not compatible. A cols: %zu B rows: %zu\n",
			a->columns, b->rows);
		return ;
	}
    if (out->rows != a->rows || out->columns != b->columns)
    {
        fprintf(stderr, "Output matrix not compatible. Expected (%zu, %zu), got (%zu, %zu)\n",
            a->rows, b->columns, out->rows, out->columns);
        return ;
    }

	// i-k-j order for cache efficiency
	for (size_t i = 0; i < a->rows; i++)
	{
		for (size_t k = 0; k < a->columns; k++)
		{
			float a_ik = *mat_at((Mat *)a, i, k);
	        for (size_t j = 0; j < b->columns; j++)
				*mat_at(out, i, j) += a_ik * *mat_at((Mat *)b, k, j);
		}
	}
}

/**
 * Transpose a matrix
 */
void mat_transpose(Mat *out, const Mat *mat)
{
    if (out->rows != mat->columns || out->columns != mat->rows)
    {
        fprintf(stderr, "Transpose: output size mismatch. Expected (%zu, %zu), got (%zu, %zu)\n",
            mat->columns, mat->rows, out->rows, out->columns);
        return;
    }
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->columns; j++)
        {
            out->p_data[j * out->columns + i] = mat->p_data[i * mat->columns + j];
        }
    }
}

/**
 * Allocate and returns a new Matrix
 */
Mat *mat_alloc(size_t rows, size_t columns)
{
	Mat *mat = NNN_ALLOC(sizeof(Mat));
	if (!mat)
		return NULL;
	mat->rows = rows;
	mat->columns = columns;
	mat->p_data = NNN_ALLOC(sizeof(float) * (rows * columns));
	if (!mat->p_data)
	{
		NNN_FREE(mat);
		return NULL;
	}
	return mat;
}

/**
 * Returns a pointer to the element at the coordinates
 */
static inline float *mat_at(Mat *mat, size_t x, size_t y)
{
	CHECK_BOUNDS(mat, x, y);
	return &mat->p_data[x * mat->columns + y];
}

/**
 * Print a matrix
 */
void mat_print(const Mat *mat)
{
	printf("┌ ");
	for (size_t c = 0; c < mat->columns; c++)
		printf("          ");
	printf("┐\n");

	for (size_t r = 0; r < mat->rows; r++)
	{
		printf("│ ");
		for (size_t c = 0; c < mat->columns; c++)
			printf("%8.4f  ", mat->p_data[r * mat->columns + c]);
		printf("│\n");
	}

	printf("└ ");
	for (size_t c = 0; c < mat->columns; c++)
		printf("          ");
	printf("┘\n");
}

/**
 * Print a network
 */
void network_print(const Network *net)
{
	printf("Network: %zu layers\n", net->n_layers);
	printf("════════════════════════════════════════\n");

	for (size_t i = 0; i < net->n_layers; i++)
	{
		printf("Layer %zu: %zu neurons\n", i, net->a[i]->rows);
		mat_print(net->a[i]);

		if (i < net->n_layers - 1)
		{
			printf("  Weights [%zu x %zu]:\n",
				net->weights[i]->rows, net->weights[i]->columns);
			mat_print(net->weights[i]);

			printf("  Biases [1 x %zu]:\n", net->biases[i]->columns);
			mat_print(net->biases[i]);

			printf("────────────────────────────────────────\n");
		}
	}
	printf("════════════════════════════════════════\n");
}

# endif
