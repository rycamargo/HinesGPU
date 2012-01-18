/*
 * SynapticComm.cu
 *
 *  Created on: 10/12/2010
 *      Author: rcamargo
 */

#include "Connections.hpp"
#include <cstdio>
#include <cassert>

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

// Put here so that eclipse will not see this function as an error in the code
extern void __syncthreads(void);

extern void checkCUDAError(const char *msg);

// TODO: remove connections from Connections NEW
ConnGpu* createGpuConnections( ConnectionInfo *connInfo, int destType, int *nNeurons, int nGroups ) {

	// Contains the structures with the connections for each neuron group
	ConnGpu *connGpuTypeHost = (ConnGpu *)malloc(nGroups*sizeof(ConnGpu));

	int nConnectionsTotal[nGroups];
	int nNeuronsPerGroup =  nNeurons[destType]/nGroups;
	int nGroupsExtraNeuron = (nNeurons[destType] % nGroups);
	int nNeuronsExtraGroups = nGroupsExtraNeuron * (nNeuronsPerGroup + 1);

	int nNeuronsInPreviousGroups = 0;
	for (int group=0; group<nGroups; group++) {

		ConnGpu & connGpu = connGpuTypeHost[group];
		nConnectionsTotal[group] = 0;

		connGpu.nNeuronsGroup = nNeuronsPerGroup;
		connGpu.nNeuronsInPreviousGroups = nNeuronsInPreviousGroups;

		if ( group < nGroupsExtraNeuron ) connGpu.nNeuronsGroup++;

		nNeuronsInPreviousGroups += connGpu.nNeuronsGroup;
	}

	/**
	 * Counts the total number of connections for the group
	 */
	for (int conn=0; conn < connInfo->nConnections; conn++) {

		if (connInfo->dest[conn] / CONN_NEURON_TYPE == destType) {

			int destNeuron = connInfo->dest[conn] % CONN_NEURON_TYPE;
			int group = destNeuron / nNeuronsPerGroup;
			if (nNeurons[destType] % nGroups != 0) {
				if (destNeuron < nNeuronsExtraGroups)
					group = destNeuron / (nNeuronsPerGroup+1);
				else
					group = nGroupsExtraNeuron + ((destNeuron - nNeuronsExtraGroups) / nNeuronsPerGroup);
			}
			nConnectionsTotal[group]++;
		}
	}




	for (int group=0; group<nGroups; group++) {

		ConnGpu & connGpu = connGpuTypeHost[group];
		connGpu.nConnectionsTotal = nConnectionsTotal[group];

		checkCUDAError("Allocation error 0 at [SynapticComm.cfor (int neuron = 0; neuron < tInfo->nNeurons[type]; neuron++) {u]:");
		/**
		 * Allocates the memory to keep the connection information in the GPU and CPU
		 */
		cudaMalloc( (void **) &(connGpu.srcDevice), 	nConnectionsTotal[group]*sizeof(int) );
		connGpu.srcHost = (int *)malloc( nConnectionsTotal[group]*sizeof(int) );
		checkCUDAError("Allocation error 1 at [SynapticComm.cu]:");

		cudaMalloc( (void **) &(connGpu.destDevice), 	nConnectionsTotal[group]*sizeof(int) );
		connGpu.destHost = (int *)malloc( nConnectionsTotal[group]*sizeof(int) );
		checkCUDAError("Allocation error 2 at [SynapticComm.cu]:");

		cudaMalloc( (void **) &(connGpu.synapseDevice),	nConnectionsTotal[group]*sizeof(ucomp) );
		connGpu.synapseHost = (ucomp *)malloc( nConnectionsTotal[group]*sizeof(ucomp) );
		checkCUDAError("Allocation error 3 at [SynapticComm.cu]:");

		cudaMalloc( (void **) &(connGpu.weightDevice),	nConnectionsTotal[group]*sizeof(ftype) );
		connGpu.weightHost = (ftype *)malloc( nConnectionsTotal[group]*sizeof(ftype) );
		checkCUDAError("Allocation error 4 at [SynapticComm.cu]:");

		cudaMalloc( (void **) &(connGpu.delayDevice),	nConnectionsTotal[group]*sizeof(ftype) );
		connGpu.delayHost = (ftype *)malloc( nConnectionsTotal[group]*sizeof(ftype) );
		checkCUDAError("Allocation error 5 at [SynapticComm.cu]:");
	}

	/**
	 * Copies the connection info data to the host memory
	 */
	int memPosList[nGroups];
	for (int group=0; group<nGroups; group++)
		memPosList[group] = 0;


	for (int conn=0; conn < connInfo->nConnections; conn++) {

		if (connInfo->dest[conn] / CONN_NEURON_TYPE == destType) {

			int destNeuron = connInfo->dest[conn] % CONN_NEURON_TYPE;
			int group = destNeuron / nNeuronsPerGroup;
			if (nNeurons[destType] % nGroups != 0) {
				if (destNeuron < nNeuronsExtraGroups)
					group = destNeuron / (nNeuronsPerGroup+1);
				else
					group = nGroupsExtraNeuron + ((destNeuron - nNeuronsExtraGroups) / nNeuronsPerGroup);
			}

			ConnGpu & connGpu = connGpuTypeHost[group];
			int memPos = memPosList[group];

			connGpu.srcHost    [memPos] = connInfo->source [conn];
			connGpu.destHost   [memPos]	= connInfo->dest   [conn];  // TODO: can move to another vector
			connGpu.synapseHost[memPos]	= connInfo->synapse[conn];  // TODO: can move to another vector
			connGpu.weightHost [memPos] = connInfo->weigth [conn];  // TODO: can move to another vector
			connGpu.delayHost  [memPos]	= connInfo->delay  [conn];  // TODO: can move to another vector

			memPosList[group]++;
		}

	}

	/**
	 * Copies the connection info data to the device memory
	 */
	for (int group=0; group<nGroups; group++) {

		assert (memPosList[group] == nConnectionsTotal[group]);

		ConnGpu & connGpu = connGpuTypeHost[group];
		cudaMemcpy( connGpu.srcDevice, 		connGpu.srcHost, 	 nConnectionsTotal[group]*sizeof(int),	  cudaMemcpyHostToDevice);
		cudaMemcpy( connGpu.destDevice, 	connGpu.destHost, 	 nConnectionsTotal[group]*sizeof(int),	  cudaMemcpyHostToDevice);
		cudaMemcpy( connGpu.synapseDevice, 	connGpu.synapseHost, nConnectionsTotal[group]*sizeof(ucomp), cudaMemcpyHostToDevice);
		cudaMemcpy( connGpu.weightDevice, 	connGpu.weightHost,  nConnectionsTotal[group]*sizeof(ftype), cudaMemcpyHostToDevice);
		cudaMemcpy( connGpu.delayDevice, 	connGpu.delayHost, 	 nConnectionsTotal[group]*sizeof(ftype), cudaMemcpyHostToDevice);

		checkCUDAError("Memcopy error at [SynapticComm.cu]:");
	}

	int nConnectionsAllGroups = 0;
	for (int group=0; group<nGroups; group++)
		nConnectionsAllGroups += connGpuTypeHost[group].nConnectionsTotal;
	printf ("Number of connections to type %d is %d (%dk).\n", destType, nConnectionsAllGroups, nConnectionsAllGroups/1000);


	return connGpuTypeHost;
}

__device__ void updateActivationListPos (
		ftype *activationList, ucomp* activationListPos, int activationListSize, ucomp cStep,
		ftype currTime, ftype dt, ucomp synapse, ftype spikeTime, ftype delay, ftype weight, int destNeuron, int nNeurons, ftype *freeMem) {

	//int neuron = blockIdx.x * blockDim.x + threadIdx.x;

	ftype fpos = (spikeTime + delay - currTime) / dt;

	ucomp pos  = ( activationListPos[synapse] + (ucomp)fpos + 1 ) % activationListSize;
	pos       += synapse * activationListSize;

	ucomp nextPos  = ( pos + 1 ) % activationListSize;
	nextPos       += synapse * activationListSize;

	ftype diff = fpos - (int)fpos;

	activationList[    pos * nNeurons + destNeuron] += (weight / dt) * ( 1 - diff );
	activationList[nextPos * nNeurons + destNeuron] += (weight / dt) * diff;


//	ftype *posValue      = freeMem;
//	ftype *nextPosValue  = posValue + blockDim.x;
//	ftype *posValueO     = nextPosValue + blockDim.x;
//	ftype *nextPosValueO = posValueO + blockDim.x;
//	ucomp *posToUpdate   = (ucomp *)(nextPosValueO + blockDim.x);
//	ucomp *posToUpdateO  = posToUpdate  + blockDim.x;
//	ucomp *cStepThread   = posToUpdateO + blockDim.x;
//
//
//	posToUpdate[threadIdx.x]   = pos;
//	cStepThread[threadIdx.x]   = cStep;
//
//	posValue[threadIdx.x]      = (weight / dt) * ( 1 - diff );
//	nextPosValue[threadIdx.x]  = (weight / dt) * diff;
//	posValueO[threadIdx.x]     = (weight / dt) * ( 1 - diff );
//	nextPosValueO[threadIdx.x] = (weight / dt) * diff;
//
//	for (int i=threadIdx.x + 1; i<blockDim.x; i++)
//		if ( cStep == cStepThread[i] && pos == posToUpdateO[i] ) {
//			posValue[threadIdx.x]     += posValueO[i];
//			nextPosValue[threadIdx.x] += nextPosValueO[i];
//			posToUpdate[i] += 1; // just need to change the value by any amount
//		}
//
//	__syncthreads();
//
//	if (pos == posToUpdate[threadIdx.x]) {
//		activationList[    pos * nNeurons + destNeuron] += posValue[threadIdx.x];
//		activationList[nextPos * nNeurons + destNeuron] += nextPosValue[threadIdx.x];
//	}



//	for (int i=0; i<blockDim.x; i++) {
//
//		if (threadIdx.x == i) {
			//activationList[    pos * nNeurons + destNeuron] += (weight / dt) * ( 1 - diff );
			//activationList[nextPos * nNeurons + destNeuron] += (weight / dt) * diff;
//		}
//	}

}


/**
 * Updates the global activation list
 * TODO: change ConnGpu connGpuDev to reference
 */
__device__ void updateActivationList( HinesStruct *hList,
		int nNeurons, ConnGpu connGpuDev,
		ftype **genSpikeTimeListDev, ucomp **nGeneratedSpikesDev,
		ftype *randomSpikeTimesDev,  int *randomSpikeDestDev, ftype *freeMem) {

	ucomp cStep = 123;

	int spikeTimeListSize = GENSPIKETIMELIST_SIZE;

	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;

	ftype *activationList  = hList[0].activationList;     // global list
	int activationListSize = hList[0].activationListSize; // global value

	if (threadIdx.x >= connGpuDev.nNeuronsGroup) return;
	//HinesStruct & h = hList[neuron];
	ftype dt = hList[neuron].dt;
	ftype currTime = hList[neuron].currStep * dt;

	for (int iConn = 0; iConn < connGpuDev.nConnectionsTotal; iConn += blockDim.x) {

		if (iConn+threadIdx.x < connGpuDev.nConnectionsTotal) {

			int srcNeuron = connGpuDev.srcDevice[iConn+threadIdx.x];
			int srcType   = srcNeuron/CONN_NEURON_TYPE;
			srcNeuron = srcNeuron%CONN_NEURON_TYPE;
			int nSpikesSource = nGeneratedSpikesDev[ srcType ][ srcNeuron ];

			ftype weight   = connGpuDev.weightDevice [iConn+threadIdx.x];
			int destNeuron = connGpuDev.destDevice   [iConn+threadIdx.x]%CONN_NEURON_TYPE;
			ucomp synapse  = connGpuDev.synapseDevice[iConn+threadIdx.x];
			ftype delay    = connGpuDev.delayDevice  [iConn+threadIdx.x];

			ftype *genSpikeTimes = genSpikeTimeListDev[srcType] + spikeTimeListSize * srcNeuron; // Put in Shared

			for (int i = 0; i < nSpikesSource; i++) {

				updateActivationListPos( activationList, hList[destNeuron].activationListPos, activationListSize, cStep,
						currTime, dt, synapse, genSpikeTimes[i], delay, weight, destNeuron, nNeurons, freeMem );
			}
		}

		cStep++;
	}


	/**
	 * Copy the random spikes
	 * Only works when random spikes are delivered to synapse 0
	 * TODO: What happens if a position outside the vector is accessed?
	 */
	int iRnd = 0;
	int destNeuron = randomSpikeDestDev[ threadIdx.x ] % CONN_NEURON_TYPE;
	int maxNeuron  = connGpuDev.nNeuronsGroup + connGpuDev.nNeuronsInPreviousGroups;
	while (destNeuron >= 0 && destNeuron < maxNeuron) {
		if ( destNeuron >= connGpuDev.nNeuronsInPreviousGroups ) {
			// synapse=0, delay=0, weight = 1
			updateActivationListPos( activationList, hList[destNeuron].activationListPos, activationListSize, cStep,
					currTime, dt, 0, randomSpikeTimesDev[iRnd+threadIdx.x], 0, 1, destNeuron, nNeurons, freeMem );
		}
		iRnd += blockDim.x;
		destNeuron = randomSpikeDestDev[iRnd+threadIdx.x]%CONN_NEURON_TYPE ;
		cStep++;
	}

}


__global__ void performCommunicationsG(int nNeurons, ConnGpu *connGpuListDev,
		ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *randomSpikeTimesDev, int *randomSpikeDestDev) {

	int group = blockIdx.x;
	ConnGpu connGpuDev = connGpuListDev[group];


	extern __shared__ ftype sharedMem[];
	ftype *freeMem =sharedMem;
//	int nSynapses = 2;
//	int *nReceivedSpikesShared = (int *)sharedMem;
//	int *nSpikesToKeepShared = nReceivedSpikesShared + connGpuDev.nNeuronsGroup * nSynapses;
//	int *sharedMemNext = nSpikesToKeepShared + connGpuDev.nNeuronsGroup * nSynapses;
//	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;

	updateActivationList( hList, nNeurons, connGpuDev, genSpikeTimeListDev, nGeneratedSpikesDev,
			randomSpikeTimesDev, randomSpikeDestDev, freeMem);
}






__global__ void performCommunicationsG(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, int *nReceivedSpikesGlobal0, int *nReceivedSpikesGlobal1) {

}

__global__ void performCommunicationsG_Step1(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory) {

}

__global__ void performCommunicationsG_Step2(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory) {

}





