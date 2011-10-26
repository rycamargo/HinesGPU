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
ConnGpu* createGpuConnections( MPIConnectionInfo *connInfo, int destType, int *nNeurons, int nGroups ) {

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

		checkCUDAError("Allocation error 0 at [SynapticComm.cu]:");
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


/**
 * Counts the number of spikes received at each synapse by each neuron of a given type
 * Used only to test if the implementation of createGpuConnections is working
 */
int **countReceivedSpikesCpu(ConnGpu *connGpuList, int nNeurons, int nGroups, ucomp **nGeneratedSpikes) {

	int nSynapses = 2;

	int **nReceivedSpikes = new int *[nSynapses];
	nReceivedSpikes[0] = new int[nNeurons];
	nReceivedSpikes[1] = new int[nNeurons];
	for (int i=0; i<nNeurons; i++) {
		nReceivedSpikes[0][i] = 0;
		nReceivedSpikes[1][i] = 0;
	}

	int typeTmp = connGpuList[0].destHost[0]/CONN_NEURON_TYPE;

	int nConsideredSynapses = 0;
	int nAddedSpikes = 0;

	for (int group = 0; group < nGroups; group++) {
		ConnGpu & connGpu = connGpuList[group];
		for (int iConn = 0; iConn < connGpu.nConnectionsTotal; iConn++) {
			assert (typeTmp == connGpu.destHost[iConn]/CONN_NEURON_TYPE);
			nReceivedSpikes[ connGpu.synapseHost[iConn] ][ connGpu.destHost[iConn]%CONN_NEURON_TYPE ] +=
					nGeneratedSpikes[ connGpu.srcHost[iConn]/CONN_NEURON_TYPE ][ connGpu.srcHost[iConn]%CONN_NEURON_TYPE ];

			nConsideredSynapses++;
			nAddedSpikes += nGeneratedSpikes[ connGpu.srcHost[iConn]/CONN_NEURON_TYPE ][ connGpu.srcHost[iConn]%CONN_NEURON_TYPE ];
		}
		//printf("###\n");
	}

	printf ("nConsideredSynapses = %d    nAddedSpikes = %d\n", nConsideredSynapses, nAddedSpikes);

	//printf("###\n");

	return nReceivedSpikes;

}

/**
 * Count the number of spikes delivered to each neuron
 * TODO: change ConnGpu connGpuDev to reference
 */
__device__ void countReceivedSpikesG(int nNeurons, ConnGpu connGpuDev, int *nReceivedSpikesShared, ucomp **nGeneratedSpikesDev, int *randomSpikeDestDev) {

	int rcvSpkListSize = connGpuDev.nNeuronsGroup * blockDim.x;

	for (int iConn = 0; iConn < connGpuDev.nConnectionsTotal; iConn += blockDim.x) {

		if (iConn+threadIdx.x < connGpuDev.nConnectionsTotal) {

			int destNeuron = (connGpuDev.destDevice[iConn+threadIdx.x]%CONN_NEURON_TYPE) - connGpuDev.nNeuronsInPreviousGroups;
			int threadStartPos = connGpuDev.synapseDevice[iConn+threadIdx.x] * rcvSpkListSize + threadIdx.x * connGpuDev.nNeuronsGroup;
			int srcNeuron = connGpuDev.srcDevice[iConn+threadIdx.x];
			nReceivedSpikesShared[ threadStartPos + destNeuron ] +=
				nGeneratedSpikesDev[ srcNeuron/CONN_NEURON_TYPE ][ srcNeuron%CONN_NEURON_TYPE ];
		}
	}

	int threadStartPosRnd = threadIdx.x * connGpuDev.nNeuronsGroup;
	int iRnd = 0;
	int destNeuron = randomSpikeDestDev[ threadIdx.x ];
	int maxNeuron = connGpuDev.nNeuronsInPreviousGroups + connGpuDev.nNeuronsGroup;
	while (destNeuron >= 0 && destNeuron < maxNeuron) {
		destNeuron -= connGpuDev.nNeuronsInPreviousGroups;
		if (destNeuron >= 0)
			nReceivedSpikesShared[ threadStartPosRnd + destNeuron ]++;
		iRnd += blockDim.x;
		destNeuron = randomSpikeDestDev[iRnd+threadIdx.x];
	}

	/**
	 * Here we consider nThreads == nNeuronsGroup;
	 * Reduce data for synapse 0
	 */
	for (int i = 1; i < blockDim.x; i++) {

		if (threadIdx.x < connGpuDev.nNeuronsGroup) {
			nReceivedSpikesShared[threadIdx.x] +=
					nReceivedSpikesShared[i * connGpuDev.nNeuronsGroup + threadIdx.x];
		}
	}

	/**
	 * Here we consider nThreads == nNeuronsGroup;
	 * Reduce data for synapse 1
	 */
	if (threadIdx.x < connGpuDev.nNeuronsGroup)
		nReceivedSpikesShared[connGpuDev.nNeuronsGroup + threadIdx.x] = 0;
	for (int i = 0; i < blockDim.x; i++) {
		if (threadIdx.x < connGpuDev.nNeuronsGroup) {
			nReceivedSpikesShared[connGpuDev.nNeuronsGroup + threadIdx.x] +=
					nReceivedSpikesShared[rcvSpkListSize + i * connGpuDev.nNeuronsGroup + threadIdx.x];
		}
	}

	//nReceivedSpikesShared[connGpuDev.nNeuronsInPreviousGroups + threadIdx.x] = nReceivedSpikesShared[threadIdx.x];


}


/**
 * Move the current spikes to accommodate the new spikes
 */
__device__ void moveCurrentSpikesG(HinesStruct *hList, ConnGpu connGpuDev, int nNeurons, ftype *spikeListGlobal,
		ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		int *startPosCurr, int *startPosNew) {

	//int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;
	if (threadIdx.x >= connGpuDev.nNeuronsGroup) return;
	HinesStruct & h = hList[neuron];
	ftype currTime = h.currStep * h.dt;

	startPosNew[connGpuDev.nNeuronsGroup + threadIdx.x]  =
			startPosCurr[threadIdx.x] + startPosNew[threadIdx.x] +
			startPosCurr[connGpuDev.nNeuronsGroup + threadIdx.x];
	startPosCurr[connGpuDev.nNeuronsGroup + threadIdx.x] =
			startPosCurr[threadIdx.x] + startPosNew[threadIdx.x];
	startPosNew[threadIdx.x]  = startPosCurr[threadIdx.x];
	startPosCurr[threadIdx.x] = 0;

	int *synSpikeListPos = spikeListPosGlobal + neuron * h.synapseListSize;

	/**
	 * Scans the spike list, copying the new generated spikes and the existing ones.
	 */
	int synapseListSize = 2;
	for (int syn=0; syn < synapseListSize ; syn++) {

		/*
		 * Move the current spikes to the their final positions
		 * TODO: Works only with 2 synapses, when synapse 0 is AMPA and synapse 1 is GABA!!!
		 */
		ftype remThresh = currTime - (3 * (h.tau[2*syn] + h.tau[2*syn+1]) );
		int synPos = syn * connGpuDev.nNeuronsGroup + threadIdx.x;

		if ( startPosCurr[synPos] <= synSpikeListPos[syn]) {

			int pos = startPosCurr[synPos];
			int spk = synSpikeListPos[syn];
			int lastSpk = synSpikeListPos[syn] + startPosNew[synPos] - startPosCurr[synPos];

			for (; spk < lastSpk; spk++) {
				// Copy only the spikes not expired
				if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
					spikeListGlobal[pos * nNeurons + neuron]  = spikeListGlobal[spk * nNeurons + neuron];
					weightListGlobal[pos * nNeurons + neuron] = weightListGlobal[spk * nNeurons + neuron];
					pos++;
				}
				//else spikeListGlobal[spk * nNeurons + neuron] = 0;
			}

		}

		else {

			int pos = startPosNew[synPos]-1;
			int spk = synSpikeListPos[syn] + startPosNew[synPos] - startPosCurr[synPos] - 1;
			int lastSpk = synSpikeListPos[syn];

			for (; spk >= lastSpk; spk--) {
				// Copy only the spikes not expired
				if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
					spikeListGlobal[pos * nNeurons + neuron]  = spikeListGlobal[spk * nNeurons + neuron];
					weightListGlobal[pos * nNeurons + neuron] = weightListGlobal[spk * nNeurons + neuron];
					pos--;
				}
				//else spikeListGlobal[spk * nNeurons + neuron] = 0;
			}

		}

	}

}


/**
 * Count the number of current spikes to keep in the spikeList
 */
__device__ void countCurrentSpikesG(HinesStruct *hList, ConnGpu connGpuDev, int nNeurons,
		ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		int *nSpikesToKeepShared) {

	//int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;
	if (threadIdx.x >= connGpuDev.nNeuronsGroup) return;
	HinesStruct & h = hList[neuron];
	ftype currTime = h.currStep * h.dt;

	//int rcvSpkListSize = connGpuDev.nNeuronsGroup * blockDim.x;
	int *synSpikeListPos = spikeListPosGlobal + neuron * h.synapseListSize;
	// SYNAPSE_AMPA 0, SYNAPSE_GABA 1
	// TODO: Works only when synapse 0 is AMPA and synapse 1 is GABA!!!
	int synapseListSize = 2;
	for (int syn = 0; syn<synapseListSize; syn++) {

		int nSpikesToKeep = 0;
		ftype remThresh = currTime - (3 * (h.tau[2*syn] + h.tau[2*syn+1]) );

		int spk = synSpikeListPos[syn];
		int lastSpk = (syn < synapseListSize-1) ? synSpikeListPos[syn+1] : spikeListSizeGlobal[neuron];
		int spkMovePos = -1;
		for (; spk < lastSpk && spkMovePos == -1; spk++) {
			if (spikeListGlobal[spk * nNeurons + neuron] > remThresh)
				nSpikesToKeep++;
			else
				spkMovePos = spk;
		}
		for (; spk < lastSpk; spk++) {
			if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
				nSpikesToKeep++;
				spikeListGlobal[spkMovePos * nNeurons + neuron]  = spikeListGlobal[spk* nNeurons + neuron];
				weightListGlobal[spkMovePos * nNeurons + neuron] = weightListGlobal[spk* nNeurons + neuron];
				spkMovePos++;
			}

		}

		if (syn==0) nSpikesToKeepShared[threadIdx.x] = nSpikesToKeep;
		else nSpikesToKeepShared[connGpuDev.nNeuronsGroup + threadIdx.x] = nSpikesToKeep;
	}
}

/**
 * Copy the new spikes to the vector
 */
__device__ void deliverGeneratedSpikesG(ConnGpu connGpuDev, int nNeurons, int *sharedMem, int *startPosNew,
		ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev, ftype *randomSpikeTimesDev, int *randomSpikeDestDev,
		ftype *spikeListGlobal, ftype *weightListGlobal) {

	int spikeTimeListSize = 5; // #############################################

	//const int nSynapses = 2;
	int *srcNeuronShared    = sharedMem;
	int *destNeuronShared   = srcNeuronShared  + blockDim.x;
	int *nSpikesSrcShared   = destNeuronShared + blockDim.x;
	int *newPosThreadShared = nSpikesSrcShared + blockDim.x;

	int srcNeuron, srcType, nSpikesSource;
	int destNeuron,	destSyn, synPosL, synPosG;
	ftype weight, delay, *genSpikeTimes;

	/**
	 * Copy the spikes received from other neurons
	 */
	for (int iConn = 0; iConn < connGpuDev.nConnectionsTotal; iConn += blockDim.x) {

		__syncthreads();

		nSpikesSrcShared[threadIdx.x]   = 0;
		newPosThreadShared[threadIdx.x] = threadIdx.x;

		if (iConn+threadIdx.x < connGpuDev.nConnectionsTotal) {

			srcNeuron = connGpuDev.srcDevice[iConn+threadIdx.x];
			srcType   = srcNeuron/CONN_NEURON_TYPE;
			srcNeuron = srcNeuron%CONN_NEURON_TYPE;

			nSpikesSource = nGeneratedSpikesDev[ srcType ][ srcNeuron ];

			weight = connGpuDev.weightDevice[iConn+threadIdx.x];
			destNeuron = connGpuDev.destDevice[iConn+threadIdx.x]%CONN_NEURON_TYPE;
			destSyn = connGpuDev.synapseDevice[iConn+threadIdx.x];
			synPosL = destSyn * connGpuDev.nNeuronsGroup + (destNeuron-connGpuDev.nNeuronsInPreviousGroups);
			synPosG = startPosNew[synPosL];
			delay = connGpuDev.delayDevice[iConn+threadIdx.x];
			genSpikeTimes = genSpikeTimeListDev[srcType] + spikeTimeListSize * srcNeuron; // Put in Shared

			srcNeuronShared[threadIdx.x]    = srcNeuron + srcType * CONN_NEURON_TYPE;
			destNeuronShared[threadIdx.x]   = destNeuron + destSyn * CONN_NEURON_TYPE;
			nSpikesSrcShared[threadIdx.x]   = nSpikesSource;
		}

		__syncthreads();

		if (nSpikesSource > 0 && iConn+threadIdx.x < connGpuDev.nConnectionsTotal) {
			//if ( srcNeuronShared[0] != srcNeuronShared[threadIdx.x] ) { // only valid if there are no repeated connections
				for (int i=0; i < threadIdx.x; i++ ) {
					if ( destNeuronShared[i] == destNeuronShared[threadIdx.x] ) {
						synPosG += nSpikesSrcShared[i];
						newPosThreadShared[i] = threadIdx.x;
					}
				}
			//}

			for (int i = 0; i < nSpikesSource; synPosG++, i++) {
				spikeListGlobal[ synPosG * nNeurons  + destNeuron ] = genSpikeTimes[i] + delay;
				weightListGlobal[ synPosG * nNeurons + destNeuron ] = weight;
			}

			if (threadIdx.x == newPosThreadShared[threadIdx.x])
				startPosNew[synPosL] = synPosG;
		}
	}

	__syncthreads();

	/**
	 * Copy the random spikes
	 * Only works when random spikes are delivered to synapse 0
	 */
	{
		int iRnd = 0;
		int destNeuron = randomSpikeDestDev[ threadIdx.x ]%CONN_NEURON_TYPE;
		int maxNeuron = connGpuDev.nNeuronsGroup + connGpuDev.nNeuronsInPreviousGroups;
		while (destNeuron >= 0 && destNeuron < maxNeuron) {
			destNeuron -= connGpuDev.nNeuronsInPreviousGroups;
			destNeuronShared[threadIdx.x] = destNeuron;
			if ( destNeuron >= 0 ) {
				int spikePos = startPosNew[destNeuron] * nNeurons +
						(destNeuron + connGpuDev.nNeuronsInPreviousGroups);

				if (threadIdx.x > 0) {
					for (int i = 1; i <= threadIdx.x && destNeuronShared[threadIdx.x - i] == destNeuron; i++) {
						spikePos += nNeurons;
						startPosNew[destNeuron]++;
					}
				}

				spikeListGlobal[  spikePos ] = randomSpikeTimesDev[iRnd+threadIdx.x];
				weightListGlobal[ spikePos ] = 1;
				startPosNew[destNeuron]++;

			}

			iRnd += blockDim.x;
			destNeuron = randomSpikeDestDev[iRnd+threadIdx.x]%CONN_NEURON_TYPE ;
		}
	}

}


// HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal
__global__ void performCommunicationsG(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, int *nReceivedSpikesGlobal0, int *nReceivedSpikesGlobal1) {


	int group = blockIdx.x;
	ConnGpu connGpuDev = connGpuListDev[group];
	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;

	extern __shared__ ftype sharedMem[];

	int nSynapses = 2;
	int *nReceivedSpikesShared = (int *)sharedMem;
	int *nSpikesToKeepShared = nReceivedSpikesShared + connGpuDev.nNeuronsGroup * nSynapses;
	int *sharedMemNext = nSpikesToKeepShared + connGpuDev.nNeuronsGroup * nSynapses;

	int receivedSpikesSize = connGpuDev.nNeuronsGroup * blockDim.x;

	for (int i=0; i < receivedSpikesSize; i += blockDim.x) {
		nReceivedSpikesShared[i + threadIdx.x] = 0; // synapse 0
		nReceivedSpikesShared[i + threadIdx.x + receivedSpikesSize] = 0; // synapse 1
	}

	/**
	 * Counts the number of spikes from other neurons and from a random source that will be added
	 */
	countReceivedSpikesG(nNeurons, connGpuDev, nReceivedSpikesShared, nGeneratedSpikesDev, randomSpikeDestDev);

	/**
	 * TODO: Remove Me [MPI]
	 * Used only during debugging to check the number of received spikes per process
	 */
	__syncthreads();
	if (threadIdx.x == 0) {
		int nTmp0 = 0, nTmp1 = 0;
		for (int i = 0; i < connGpuDev.nNeuronsGroup; i++) {
			nTmp0 += nReceivedSpikesShared[i];
			nTmp1 += nReceivedSpikesShared[i + connGpuDev.nNeuronsGroup];
		}
		nReceivedSpikesGlobal0[group] = nTmp0;
		nReceivedSpikesGlobal1[group] = nTmp1;
	}

	/**
	 * Counts the number of current spikes in spikeList to keep for the next kernel call
	 */
	nSpikesToKeepShared[threadIdx.x] = 0;
	nSpikesToKeepShared[connGpuDev.nNeuronsGroup + threadIdx.x] = 0;

	countCurrentSpikesG(hList, connGpuDev, nNeurons, spikeListGlobal, weightListGlobal, spikeListPosGlobal, spikeListSizeGlobal, nSpikesToKeepShared);

	/**
	 * Copy the data to the CPU for debugging
	 * Here we consider nThreads == nNeuronsGroup;
	 */
	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		spikeListSizeGlobal[neuron] =
				nReceivedSpikesShared[threadIdx.x] + nReceivedSpikesShared[threadIdx.x + connGpuDev.nNeuronsGroup] +
				nSpikesToKeepShared[threadIdx.x]   + nSpikesToKeepShared[threadIdx.x + connGpuDev.nNeuronsGroup];
	}

	/**
	 * Move the current spikes in the spikeList vector
	 */
	int *startPosCurr = nSpikesToKeepShared;
	int *startPosNew = nReceivedSpikesShared;

	moveCurrentSpikesG(hList, connGpuDev, nNeurons, spikeListGlobal, weightListGlobal,
			spikeListPosGlobal, spikeListSizeGlobal, startPosCurr, startPosNew);

	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		spikeListPosGlobal[2*neuron]   = startPosCurr[threadIdx.x];
		spikeListPosGlobal[2*neuron+1] = startPosCurr[threadIdx.x + connGpuDev.nNeuronsGroup];
	}

	deliverGeneratedSpikesG(connGpuDev, nNeurons, sharedMemNext, startPosNew,
			nGeneratedSpikesDev, genSpikeTimeListDev, randomSpikeTimesDev, randomSpikeDestDev,
			spikeListGlobal, weightListGlobal);

}










// HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal
__global__ void performCommunicationsG_Step1(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory) {


	int group = blockIdx.x;
	ConnGpu connGpuDev = connGpuListDev[group];
	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;

	extern __shared__ ftype sharedMem[];

	int nSynapses = 2;
	int *nReceivedSpikesShared = (int *)sharedMem;
	int *nSpikesToKeepShared = nReceivedSpikesShared + connGpuDev.nNeuronsGroup * nSynapses;

	int receivedSpikesSize = connGpuDev.nNeuronsGroup * blockDim.x;

	for (int i=0; i < receivedSpikesSize; i += blockDim.x) {
		nReceivedSpikesShared[i + threadIdx.x] = 0; // synapse 0
		nReceivedSpikesShared[i + threadIdx.x + receivedSpikesSize] = 0; // synapse 1
	}

	/**
	 * Counts the number of spikes from other neurons and from a random source that will be added
	 */
	countReceivedSpikesG(nNeurons, connGpuDev, nReceivedSpikesShared, nGeneratedSpikesDev, randomSpikeDestDev);

	/**
	 * Counts the number of current spikes in spikeList to keep for the next kernel call
	 */
	nSpikesToKeepShared[threadIdx.x] = 0;
	nSpikesToKeepShared[connGpuDev.nNeuronsGroup + threadIdx.x] = 0;

	countCurrentSpikesG(hList, connGpuDev, nNeurons, spikeListGlobal, weightListGlobal, spikeListPosGlobal, spikeListSizeGlobal, nSpikesToKeepShared);

	/**
	 * Copy the data to the CPU for debugging
	 * Here we consider nThreads == nNeuronsGroup;
	 */
	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		spikeListSizeGlobal[neuron] =
				nReceivedSpikesShared[threadIdx.x] + nReceivedSpikesShared[threadIdx.x + connGpuDev.nNeuronsGroup] +
				nSpikesToKeepShared[threadIdx.x]   + nSpikesToKeepShared[threadIdx.x + connGpuDev.nNeuronsGroup];
	}

	/**
	 * Move the current spikes in the spikeList vector
	 */
	int *startPosCurr = nSpikesToKeepShared;
	int *startPosNew = nReceivedSpikesShared;

	moveCurrentSpikesG(hList, connGpuDev, nNeurons, spikeListGlobal, weightListGlobal,
			spikeListPosGlobal, spikeListSizeGlobal, startPosCurr, startPosNew);

	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		spikeListPosGlobal[2*neuron]   = startPosCurr[threadIdx.x];
		spikeListPosGlobal[2*neuron+1] = startPosCurr[threadIdx.x + connGpuDev.nNeuronsGroup];
	}

	/*
	 * Used only for benchmarking
	 */
	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		tmpDevMemory[neuron] = startPosNew[threadIdx.x];
		tmpDevMemory[nNeurons + neuron] = startPosNew [connGpuDev.nNeuronsGroup + threadIdx.x];
	}

}

__global__ void performCommunicationsG_Step2(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory) {

	int group = blockIdx.x;
	ConnGpu connGpuDev = connGpuListDev[group];
	extern __shared__ ftype sharedMem[];
	int neuron = connGpuDev.nNeuronsInPreviousGroups + threadIdx.x;
	int nSynapses = 2;

	int *startPosNew = (int *)sharedMem;
	int *sharedMemNext = startPosNew + connGpuDev.nNeuronsGroup * nSynapses;

	/*
	 * Used only for benchmarking
	 */
	if (threadIdx.x < connGpuDev.nNeuronsGroup ) {
		startPosNew[threadIdx.x] = tmpDevMemory[neuron];
		startPosNew [connGpuDev.nNeuronsGroup + threadIdx.x] = tmpDevMemory[nNeurons + neuron];
	}


	deliverGeneratedSpikesG(connGpuDev, nNeurons, sharedMemNext, startPosNew,
			nGeneratedSpikesDev, genSpikeTimeListDev, randomSpikeTimesDev, randomSpikeDestDev,
			spikeListGlobal, weightListGlobal);

}



