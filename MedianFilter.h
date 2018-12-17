#ifndef _HEADER_MEDIANFILTER
#define _HEADER_MEDIANFILTER

#ifndef HSM
#define HSM 3
#endif

#define LSM (HSM*2 + 1)
#define FSM (LSM*LSM)

__device__ float medianBubble(float *dataQueue, int lengthQueue)
{
  int minValueIndex;
  float bufferData;
  int i, j;

  for(j = 0; j <= (lengthQueue - 1)/2; j++)
  {
    minValueIndex = j;
    for(i = j + 1; i < lengthQueue; i++)
      if(dataQueue[i] < dataQueue[minValueIndex])
        minValueIndex = i;

    bufferData = dataQueue[j];
    dataQueue[j] = dataQueue[minValueIndex];
    dataQueue[minValueIndex] = bufferData;
  }

  return dataQueue[(lengthQueue - 1)/2];
}

__device__ float medianNoSort(float *dataQueue, int indexStart, int indexEnd,
  int lengthQueue)
{
  int leftIndex, rightIndex;
  float pivotData, medianResult;
  leftIndex = indexStart;
  rightIndex = indexEnd;
  pivotData = dataQueue[indexStart];

  while(leftIndex < rightIndex)
  {
    while(leftIndex < rightIndex && dataQueue[rightIndex] >= pivotData)
      rightIndex--;
    dataQueue[leftIndex] = dataQueue[rightIndex];
    while(leftIndex < rightIndex && dataQueue[leftIndex] <= pivotData)
      leftIndex++;
    dataQueue[rightIndex] = dataQueue[leftIndex];
  }
  if(leftIndex > (lengthQueue - 1)/2)
    medianResult = medianNoSort(dataQueue, indexStart, leftIndex - 1, lengthQueue);
  else if(leftIndex == (lengthQueue - 1)/2)
    medianResult = pivotData;
  else
    medianResult = medianNoSort(dataQueue, leftIndex + 1, indexEnd, lengthQueue);

  return medianResult;
}

__global__ void warmUp(int nInX, int nInY, float *dataInput, float *dataOutput)
{
  int indexInX = blockDim.x*blockIdx.x + threadIdx.x;
  int indexInY = blockDim.y*blockIdx.y + threadIdx.y;
  float filterWindow[FSM];
  int numCount, iCol, iRow;

  if(indexInX >= HSM && indexInX < nInX - HSM
    && indexInY >= HSM && indexInY < nInY - HSM)
  {
    numCount = 0;
    for(iRow = indexInY - HSM; iRow <= indexInY + HSM; iRow++)
      for(iCol = indexInX - HSM; iCol <= indexInX + HSM; iCol++)
      {
        filterWindow[numCount] = dataInput[iRow*nInX + iCol];
        numCount++;
      }

#ifdef Bubble
    dataOutput[indexInY*nInX + indexInX] = medianBubble(filterWindow, numCount);
#else
    dataOutput[indexInY*nInX + indexInX] = medianNoSort(filterWindow, 0,
      numCount - 1, numCount);
#endif
  }
}

__global__ void kernelFilterDiscard(int nInX, int nInY, float *dataInput,
  float *dataOutput)
{
  int indexInX = blockDim.x*blockIdx.x + threadIdx.x;
  int indexInY = blockDim.y*blockIdx.y + threadIdx.y;
  float filterWindow[FSM];
  int numCount, iCol, iRow;

  if(indexInX >= HSM && indexInX < nInX - HSM
    && indexInY >= HSM && indexInY < nInY - HSM)
  {
    numCount = 0;
    for(iRow = indexInY - HSM; iRow <= indexInY + HSM; iRow++)
      for(iCol = indexInX - HSM; iCol <= indexInX + HSM; iCol++)
      {
        filterWindow[numCount] = dataInput[iRow*nInX + iCol];
        numCount++;
      }

#ifdef Bubble
    dataOutput[indexInY*nInX + indexInX] = medianBubble(filterWindow, FSM);
#else
    dataOutput[indexInY*nInX + indexInX] = medianNoSort(filterWindow, 0, FSM - 1, FSM);
#endif
  }
}

__global__ void kernelFilterZero(int nInX, int nInY, float *dataInput,
  float *dataOutput)
{
  int indexInX = blockDim.x*blockIdx.x + threadIdx.x;
  int indexInY = blockDim.y*blockIdx.y + threadIdx.y;
  float filterWindow[FSM];
  int numCount, iCol, iRow;

  if(indexInX < nInX && indexInY < nInY)
  {
    numCount = 0;
    for(iRow = indexInY - HSM; iRow <= indexInY + HSM; iRow++)
      for(iCol = indexInX - HSM; iCol <= indexInX + HSM; iCol++)
      {
        if(iCol < 0 || iCol >= nInX || iRow < 0 || iRow >= nInY)
          filterWindow[numCount] = 0.0F;
        else
          filterWindow[numCount] = dataInput[iRow*nInX + iCol];
        numCount++;
      }

#ifdef Bubble
    dataOutput[indexInY*nInX + indexInX] = medianBubble(filterWindow, FSM);
#else
    dataOutput[indexInY*nInX + indexInX] = medianNoSort(filterWindow, 0, FSM - 1, FSM);
#endif
  }
}

__global__ void kernelFilterShrink(int nInX, int nInY, float *dataInput,
  float *dataOutput)
{
  int indexInX = blockDim.x*blockIdx.x + threadIdx.x;
  int indexInY = blockDim.y*blockIdx.y + threadIdx.y;
  float filterWindow[FSM];
  int numCount, iCol, iRow;

  if(indexInX < nInX && indexInY < nInY)
  {
    numCount = 0;
    for(iRow = indexInY - HSM; iRow <= indexInY + HSM; iRow++)
      for(iCol = indexInX - HSM; iCol <= indexInX + HSM; iCol++)
        if(iCol >= 0 && iCol < nInX && iRow >= 0 && iRow < nInY)
        {
          filterWindow[numCount] = dataInput[iRow*nInX + iCol];
          numCount++;
        }

#ifdef Bubble
    dataOutput[indexInY*nInX + indexInX] = medianBubble(filterWindow, numCount);
#else
    dataOutput[indexInY*nInX + indexInX] = medianNoSort(filterWindow, 0,
      numCount - 1, numCount);
#endif
  }
}

__global__ void kernelFilterExtend(int nInX, int nInY, float *dataInput,
  float *dataOutput)
{
  int indexInX = blockDim.x*blockIdx.x + threadIdx.x;
  int indexInY = blockDim.y*blockIdx.y + threadIdx.y;
  float filterWindow[FSM];
  int localLeft, localRight, localBottom, localTop;
  int globalLeft, globalRight, globalBottom, globalTop;
  int iCol, iRow;

  if(indexInX < nInX && indexInY < nInY)
  {
    globalLeft = max(0, indexInX - HSM);
    globalRight = min(nInX - 1, indexInX + HSM);
    globalBottom = max(0, indexInY - HSM);
    globalTop = min(nInY - 1, indexInY + HSM);

    localLeft = HSM - (indexInX - globalLeft);
    localRight = HSM + 1 + (globalRight - indexInX);
    localBottom = HSM - (indexInY - globalBottom);
    localTop = HSM + 1 + (globalTop - indexInY);

    for(iRow = localBottom; iRow < localTop; iRow++)
      for(iCol = localLeft; iCol < localRight; iCol++)
        filterWindow[iRow*LSM + iCol]
          = dataInput[(iRow - localBottom + globalBottom)*nInX
          + (iCol - localLeft + globalLeft)];

    // extend toward bottom
    for(iRow = 0; iRow < localBottom; iRow++)
      for(iCol = localLeft; iCol < localRight; iCol++)
        filterWindow[iRow*LSM + iCol]
          = filterWindow[localBottom*LSM + iCol];
    // extend toward top
    for(iRow = localTop; iRow < LSM; iRow++)
      for(iCol = localLeft; iCol < localRight; iCol++)
        filterWindow[iRow*LSM + iCol]
          = filterWindow[localTop*LSM + iCol];

    for(iRow = 0; iRow < LSM; iRow++)
    {
      // extend toward left
      for(iCol = 0; iCol < localLeft; iCol++)
        filterWindow[iRow*LSM + iCol]
          = filterWindow[iRow*LSM + localLeft];
      // extend toward right
      for(iCol = localRight; iCol < LSM; iCol++)
        filterWindow[iRow*LSM + iCol]
          = filterWindow[iRow*LSM + localRight];
    }

#ifdef Bubble
    dataOutput[indexInY*nInX + indexInX] = medianBubble(filterWindow, FSM);
#else
    dataOutput[indexInY*nInX + indexInX] = medianNoSort(filterWindow, 0, FSM - 1, FSM);
#endif
  }
}

#endif
