#include <stdio.h>
#include "MedianFilter.h"

#define BNX 16
#define BNY 16

#if defined Zero
#define KER "kernelFilterZero"
#elif defined Shrink
#define KER "kernelFilterShrink"
#elif defined Extend
#define KER "kernelFilterExtend"
#else
#define KER "kernelFilterDiscard"
#endif

#ifdef Bubble
#define MDN "medianBubble"
#else
#define MDN "medianNoSort"
#endif

#define cuCheck(arg) getCudaError(__FILE__, __LINE__, arg)

inline void getCudaError(const char *file, const int line, cudaError_t cuerr)
{
  if(cuerr != cudaSuccess)
  {
    printf("cudaError at %s(%d): %s\n", file, line, cudaGetErrorString(cuerr));
    cudaDeviceReset();
    exit(2);
  }
}

int main(int argc, char *argv[])
{
  const int nInX = 600;
  const int nInY = 600;
  size_t sizeXY = nInX*nInY*sizeof(float);

  float *hostInput, *hostOutput;
  hostInput  = new float[nInX*nInY] ();
  hostOutput = new float[nInX*nInY] ();

  float *deviceInput, *deviceOutput;
  cudaSetDevice(0);
  cudaMalloc((float**) &deviceInput , sizeXY);
  cudaMalloc((float**) &deviceOutput, sizeXY);

  FILE* fp = fopen("example/data_orig.bin", "rb");
  if( ! fp)
  {
    printf("Fatal error at %s(%d): fail to open file example/data_orig.bin",
      __FILE__, __LINE__);
    exit(1);
  }
  fread(hostInput, sizeXY, 1, fp);
  fclose(fp);
  cuCheck( cudaMemcpy(deviceInput, hostInput, sizeXY, cudaMemcpyHostToDevice) );

  dim3 Block(BNX, BNY, 1);
  dim3 Grid(ceil(1.0F*nInX/BNX), ceil(1.0F*nInY/BNY), 1);
  float elapsedTime;
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  // Warmup Launching
  cudaEventRecord(startEvent, 0);
  warmUp <<< Grid, Block >>> (nInX, nInY, deviceInput, deviceOutput);
  cuCheck( cudaGetLastError() );
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

  // Kernel Launching
  cudaEventRecord(startEvent, 0);
#if defined Zero
  kernelFilterZero <<< Grid, Block >>> (nInX, nInY, deviceInput, deviceOutput);
#elif defined Shrink
  kernelFilterShrink <<< Grid, Block >>> (nInX, nInY, deviceInput, deviceOutput);
#elif defined Extend
  kernelFilterExtend <<< Grid, Block >>> (nInX, nInY, deviceInput, deviceOutput);
#else
  kernelFilterDiscard <<< Grid, Block >>> (nInX, nInY, deviceInput, deviceOutput);
#endif
  cuCheck( cudaGetLastError() );
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

  printf("Elapsed time of %s with %s @%dx%d is: %5.3f ms.\n", KER, MDN,
    LSM, LSM, elapsedTime);

  cuCheck( cudaMemcpy(hostOutput, deviceOutput, sizeXY, cudaMemcpyDeviceToHost) );
  fp = fopen("example/data_filt.bin", "wb");
  fwrite(hostOutput, sizeXY, 1, fp);
  fclose(fp);

  delete [] hostInput;
  delete [] hostOutput;

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaDeviceReset();

  return 0;
}
