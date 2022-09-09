#define THREADS_PER_BLOCK 512
#define BATCH_SIZE 128
#define LEARNING_RATE 0.01

#define index(x, y, col_sz) (x * col_sz + y)