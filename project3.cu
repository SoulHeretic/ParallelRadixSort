#include <assert.h>
#include <stdio.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(long long* data, long long count, int first, int step)
{
	assert(data != NULL);

	for(long long i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(long long i = count-1; i > 0; i--) //knuth shuffle
    {
        long long j = RAND_RANGE(i);
        long long k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
   // printf("%d\n", bits);
    return bits;
}

//Feel free to change the names of the kernels or define more kernels below if necessary

//define the histogram kernel here
__global__ void histogram(long long* d_input, int* partition_d, int partitionSize, long long rSize){

    long long tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < rSize){
        uint  nbits = ceil(log2((float(partitionSize)))); 
        uint h = bfe(d_input[tid], 0, nbits);

        atomicAdd(&(partition_d[h]), 1);   
    }
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int* partition_d, int* prefixScan_d, int partitionSize){

    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    int threads = threadIdx.x;

    int stride = 1;
    int span = 2*threads;

    if(tid < partitionSize/2 ){
        partition_d[span] = partition_d[span];
        partition_d[span+1] = partition_d[span+1];
        //For loop with bitshift
        for(int i = partitionSize >> 1; i > 0; i >>= 1){
            __syncthreads();
            if(threads < i){
                int temp1 = stride * (span+1) - 1;
                int temp2 = stride * (span+2) - 1;
                atomicAdd(&(partition_d[temp2]), (partition_d[temp1]));   //moved shared memory
            }
            __syncthreads();
            stride *= 2;    //propery adjust stride like in class
        }
        __syncthreads();

        if(threads == 0){
            partition_d[partitionSize - 1] = 0;
        }

        for(int i = 1; i < partitionSize; i *= 2){
            stride >>= 1;
            __syncthreads();

            if(threads < i){
                int temp1 = stride * (span+1)-1;
                int temp2 = stride * (span+2)-1;

                int t = partition_d[temp1];
                partition_d[temp1] = partition_d[temp2];
                atomicAdd(&(partition_d[temp2]), t);
            }
            __syncthreads();
        }
        __syncthreads();

        span = 2*threads;
        //Move results into prefix
        prefixScan_d[span] = partition_d[span];  
        prefixScan_d[span+1] = partition_d[span+1];
    }
    __syncthreads();
}

//define the reorder kernel here
__global__ void Reorder(long long* d_input, int* prefixScan_d, int partitionSize, long long rSize, long long* rOutput){
    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    //int threads = threadIdx.x;

    if(tid < rSize){
        uint  nbits = ceil(log2((float(partitionSize))));
        uint h = bfe(d_input[tid], 0, nbits);
        int atomic = atomicAdd(&(prefixScan_d[h]), 1); 
        rOutput[atomic] = d_input[tid];
    }
    __syncthreads();
}

int main(int argc, char const *argv[])
{
    long long rSize = atoi(argv[1]);
    int partitionSize = atoi(argv[2]);
    
    long long* h_input;
    long long* d_input;

    //Host memory allocations
    int* partition_h = (int *)malloc(sizeof(int)*partitionSize);
    int* partition_d;

    int* prefixScan_h = (int *)malloc(sizeof(int)*partitionSize);
    int* prefixScan_d;

    long long* rOutput_h = (long long *)malloc(sizeof(long long)*rSize);
    long long* rOutput;

    //Cuda Memory Allocations
    cudaMallocHost((void**)&h_input, sizeof(long long)*rSize);
    cudaMalloc((void**)&d_input, sizeof(long long)*rSize);
 
    dataGenerator(h_input, rSize, 0, 1);
    cudaMemcpy(d_input, h_input, sizeof(long long)*rSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&partition_d, sizeof(int) * partitionSize);
    cudaMemcpy(partition_d, partition_h, sizeof(int) * partitionSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&prefixScan_d, sizeof(int) * partitionSize);

    cudaMalloc((void**)&rOutput, sizeof(long long) *rSize);

    int blocks = ceil(rSize/(float)32);                 //Number of blocks to be used
    int prefix_blocks = ceil(partitionSize/(float)32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    histogram<<<blocks, 256, sizeof(int) * partitionSize>>>(d_input, partition_d, partitionSize, rSize);
    cudaDeviceSynchronize();
    cudaMemcpy(partition_h, partition_d, sizeof(int)*partitionSize, cudaMemcpyDeviceToHost);
 
    prefixScan<<<prefix_blocks, 256, sizeof(int) * partitionSize>>>(partition_d, prefixScan_d, partitionSize);
    cudaDeviceSynchronize();
    cudaMemcpy(prefixScan_h, prefixScan_d, sizeof(int)*partitionSize, cudaMemcpyDeviceToHost);

    Reorder<<<blocks, 32>>>(d_input, prefixScan_d, partitionSize, rSize, rOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(rOutput_h, rOutput, sizeof(long long)*rSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    //Print out the prefix and partition
    for(int i = 0; i < partitionSize; i++){  
        printf("partition %d: offset %d, number of keys %d\n", i,prefixScan_h[i],partition_h[i]);
    }
    printf("\n");
    printf("\n");
    //Print out the reordered partitions
    int k = 1;
    for(long long i = 0; i < rSize; i++){
        int j = rSize/partitionSize;
        if(i % j == 0){
            printf("\nPartition : %d\n", k);
            k++;
        }
        printf("%lld ", rOutput_h[i]);
        j++;
    }
    printf("\n");
    printf("\n");

    printf( "******** Total Running Time of All Kernels =  %0.5f ms *******\n", elapsedTime );

    cudaFreeHost(h_input);
    cudaFree(d_input);
    free(partition_h);
    cudaFree(partition_d);
    cudaFree(prefixScan_d);
    free(prefixScan_h);
    return 0;
}
