#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include "GpuSimulationControl.hpp"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>

#ifdef MPI_GPU_NN
#include <mpi.h>
#endif

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

extern __global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal);
extern ConnGpu* createGpuConnections( ConnectionInfo *connInfoList, int destType, int *nNeurons, int nGroups );
extern int **countReceivedSpikesCpu(ConnGpu *connGpuList, int nNeurons, int nGroups, ucomp **nGeneratedSpikes);
extern __global__ void performCommunicationsG(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, int *nReceivedSpikesGlobal0, int *nReceivedSpikesGlobal1);
__global__ void performCommunicationsG_Step1(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory);
__global__ void performCommunicationsG_Step2(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev, ftype *tmpDevMemory);


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

//===================================================================================================

GpuSimulationControl::GpuSimulationControl(ThreadInfo *tInfo) {

	this->tInfo = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void GpuSimulationControl::prepareSynapses() {

	int *nNeurons = tInfo->nNeurons;
	SynapticData* synData = sharedData->synData;

	/**
	 * Prepare the synaptic channels and spike generation
	 */
	int totalTypes = synData->totalTypes;

	pthread_mutex_lock (sharedData->mutex);
	if (synData->spikeListDevice == 0) {
		synData->spikeListDevice   = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->weightListDevice  = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->spikeListPosDevice  = (int **) malloc (sizeof(int *) * totalTypes);
		synData->spikeListSizeDevice = (int **) malloc (sizeof(int *) * totalTypes);


		synData->spikeListGlobal     = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->weightListGlobal    = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->spikeListPosGlobal  =  (int **) malloc (sizeof(int *) * totalTypes);
		synData->spikeListSizeGlobal = (int **) malloc (sizeof(int *) * totalTypes);

		synData->vmListHost    = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->vmListDevice  = (ftype **) malloc (sizeof(ftype *) * totalTypes);

		/*
		 * [MPI] Used in the CPU and GPU version to distribute the spike list among the processes
		 */
		synData->genSpikeTimeListHost     = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->nGeneratedSpikesHost     = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		synData->genSpikeTimeListDevice   = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->nGeneratedSpikesDevice   = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		if (benchConf.simProcMode == NN_GPU || benchConf.simCommMode == NN_GPU) {
			synData->genSpikeTimeListGpusDev  = (ftype ***) malloc (sizeof(ftype **) * sharedData->nThreadsCpu);
			synData->genSpikeTimeListGpusHost = (ftype ***) malloc (sizeof(ftype **) * sharedData->nThreadsCpu);
			synData->nGeneratedSpikesGpusDev  = (ucomp ***) malloc (sizeof(ucomp **) * sharedData->nThreadsCpu);
			synData->nGeneratedSpikesGpusHost = (ucomp ***) malloc (sizeof(ucomp **) * sharedData->nThreadsCpu);
		}
	}
	pthread_mutex_unlock (sharedData->mutex);


	/**
	 * Prepare the delivered spike related lists
	 * - spikeListPos and spikeListSize
	 */
	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {
		synData->spikeListGlobal[type] = 0;
		synData->weightListGlobal[type] = 0;
		int totalListPosSize = 1 + sharedData->matrixList[type][0].synapticChannels->synapseListSize * nNeurons[type];
		synData->spikeListPosGlobal[type]  = (int *) malloc(sizeof(int) * totalListPosSize);
		synData->spikeListSizeGlobal[type] = (int *) malloc(sizeof(int) * (nNeurons[type]+1));
		cudaMalloc ((void **) &(synData->spikeListPosDevice[type]),  sizeof(int)  * totalListPosSize);
		cudaMalloc ((void **) &(synData->spikeListSizeDevice[type]), sizeof(int)  * (nNeurons[type]+1));

		synData->vmListHost[type] = (ftype *) malloc(sizeof(ftype) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->vmListDevice[type]), sizeof(ftype)  * nNeurons[type]);

		for (int i=0; i<totalListPosSize; i++)
			synData->spikeListPosGlobal[type][i] = 0;
		for (int i=0; i<nNeurons[type]+1; i++)
			synData->spikeListSizeGlobal[type][i] = 0;

		cudaMemcpy(synData->spikeListPosDevice[type], synData->spikeListPosGlobal[type],
				sizeof(int) * totalListPosSize, cudaMemcpyHostToDevice);
		cudaMemcpy(synData->spikeListSizeDevice[type], synData->spikeListSizeGlobal[type],
				sizeof(int) * (nNeurons[type]+1), cudaMemcpyHostToDevice);
	}

	/**
	 * Prepare the lists containing the generated spikes during each kernel call
	 */
	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {
		int spikeTimeListSize = GENSPIKETIMELIST_SIZE;

		synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
		cudaMalloc ((void **) &(synData->genSpikeTimeListDevice[type]), sizeof(ftype) * spikeTimeListSize * nNeurons[type]);

		synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->nGeneratedSpikesDevice[type]), sizeof(ucomp) * nNeurons[type]);

		for (int neuron = 0; neuron < nNeurons[type]; neuron++ ) {
			HinesStruct & h = sharedData->hList[type][neuron];
			h.spikeTimes  = synData->genSpikeTimeListDevice[type] + spikeTimeListSize * neuron;
			h.nGeneratedSpikes = synData->nGeneratedSpikesDevice[type];// + neuron;
		}
	}
}

int GpuSimulationControl::prepareExecution(int type) {

	int nNeurons = tInfo->nNeurons[type];
	int nKernelSteps = kernelInfo->nKernelSteps;

	HinesStruct **hListPtr = &(sharedData->hList[type]);

	HinesStruct *hList = (HinesStruct *)malloc(nNeurons*sizeof(HinesStruct)); //new HinesStruct[nNeurons];

	HinesMatrix & m0 = sharedData->matrixList[type][0];
	int nComp = m0.nComp;
	int nCompActive = m0.activeChannels->getCompListSize();
	int nSynaptic = m0.synapticChannels->synapseListSize;

	int sharedMemMatrix    = sizeof(ftype) * (3*nComp + m0.mulListSize + m0.leftListSize); //+ nComp*nComp;
	int sharedMemSynaptic  = sizeof(ftype) * 4 * m0.synapticChannels->nChannelTypes;
	int sharedMemTotal = sharedMemMatrix + sharedMemSynaptic;

	int exclusiveMemMatrix   = sizeof(ftype) * (5*nComp + m0.leftListSize);
	int exclusiveMemActive   = sizeof(ftype) * (nCompActive*7);
	//int exclusiveMemSynaptic = sizeof(ftype) * m.spikeTimeListSize;
	int exclusiveMemTotal = exclusiveMemMatrix + exclusiveMemActive + sizeof(ftype)*nComp*nKernelSteps;

	/**
	 * Allocates memory to all neurons
	 */
	printf("TotalMem = %10.3f MB %d %d %d\n",(sharedMemTotal + exclusiveMemTotal * nNeurons)/(1024.*1024.), sharedMemTotal, exclusiveMemTotal, nNeurons);
	ftype *memory;
	cudaMalloc((void **)(&(memory)), sharedMemTotal + exclusiveMemTotal * nNeurons);

	/**
	 * Copies the shared ftype data from matrix[0] to the GPU
	 */
	ftype *sharedMemMatrixAddress = m0.Cm;
	cudaMemcpy(memory, sharedMemMatrixAddress, sharedMemMatrix, cudaMemcpyHostToDevice);
	ftype *sharedMemSynapticAddress = m0.synapticChannels->tau;
	cudaMemcpy(memory+sharedMemMatrix/sizeof(ftype), sharedMemSynapticAddress, sharedMemSynaptic, cudaMemcpyHostToDevice);

	/**
	 * Allocates and copies shared data of type ucomp from matrix[0] to the GPU
	 */
	ucomp *ucompSharedMemory;
	cudaMalloc((void **)(&(ucompSharedMemory)), sizeof(ucomp) * ((m0.mulListSize + m0.leftListSize) * 2 + nComp + nCompActive + 3*nSynaptic));
	ucomp *ucompActiveAddress   = ucompSharedMemory + (m0.mulListSize + m0.leftListSize) * 2 + nComp;
	ucomp *ucompSynapticAddress = ucompActiveAddress + nCompActive;

	cudaMemcpy(ucompSharedMemory,    m0.ucompMemory, sizeof(ucomp) * ((m0.mulListSize + m0.leftListSize) * 2 + nComp), cudaMemcpyHostToDevice);
	cudaMemcpy(ucompActiveAddress,   m0.activeChannels->getCompList(), sizeof(ucomp) * nCompActive, cudaMemcpyHostToDevice);
	cudaMemcpy(ucompSynapticAddress, m0.synapticChannels->synapseCompList, sizeof(ucomp) * nSynaptic*3, cudaMemcpyHostToDevice);

	/**
	 * Prepare the MatrixStruct h for each neuron in the GPU
	 */
	for (int neuron = 0; neuron < nNeurons; neuron++ ) {

		HinesMatrix & m = sharedData->matrixList[type][neuron];

		HinesStruct & h = hList[neuron];
		h.type = type;

		/**
		 * Shared memory
		 */
		h.memoryS = memory;
		h.Cm = h.memoryS;
		h.Ra = h.Cm + nComp;
		h.Rm = h.Ra + nComp;
		h.leftList = h.Rm + nComp;
		h.mulList  = h.leftList + m.leftListSize; // Used only when triangAll = 0

		/**
		 * Memory allocated per thread
		 */
		h.memoryE = memory + sharedMemTotal/sizeof(ftype) + neuron*exclusiveMemTotal/sizeof(ftype);
		ftype *exclusiveAddressM = m.rhsM;
		cudaMemcpy(h.memoryE, exclusiveAddressM, exclusiveMemMatrix, cudaMemcpyHostToDevice);
		// must match the order in HinesMatrix.cpp
		h.rhsM = h.memoryE	;
		h.vmList = h.rhsM + nComp;
		h.vmTmp = h.vmList + nComp;
		h.curr = h.vmTmp + nComp;
		h.active = h.curr + nComp;
		h.triangList = h.active + nComp; // triangularized list
		h.vmTimeSerie = h.triangList + m.leftListSize;

		h.currStep = m.currStep;
		h.vRest = m.vRest;
		h.dx = m.dx;
		h.nComp = m.nComp;
		h.dt = m.dt;
		h.triangAll = m.triangAll;

		h.mulListSize = m.mulListSize;
		h.leftListSize = m.leftListSize;
		h.mulListComp    = ucompSharedMemory;
		h.mulListDest    = h.mulListComp  + h.mulListSize;
		h.leftListLine   = h.mulListDest  + h.mulListSize;
		h.leftListColumn = h.leftListLine + h.leftListSize;
		h.leftStartPos   = h.leftListColumn + h.leftListSize;

		// must match the order in ActiveChannels.cpp
		if (nCompActive > 0) {
			cudaMemcpy(h.vmTimeSerie + nComp*nKernelSteps, m.activeChannels->memory, exclusiveMemActive, cudaMemcpyHostToDevice);

			h.n = h.vmTimeSerie + nComp*nKernelSteps;
			h.h = h.n + nCompActive;
			h.m = h.h + nCompActive;
			h.gNaBar = h.m + nCompActive;
			h.gKBar  = h.gNaBar + nCompActive;
			h.gNaChannel  = h.gKBar + nCompActive;
			h.gKChannel  = h.gNaChannel + nCompActive;

			h.ELeak = m.activeChannels->ELeak;
			h.EK = m.activeChannels->EK;
			h.ENa = m.activeChannels->ENa;

			h.compListSize = nCompActive;
			h.compList = ucompActiveAddress;

			checkCUDAError("Memory Allocation3:");
		}

		if (m.synapticChannels != 0) {

			h.synapseListSize = m.synapticChannels->synapseListSize;
			h.nChannelTypes   = m.synapticChannels->nChannelTypes;
			h.synapseCompList = ucompSynapticAddress;
			h.synapseTypeList = h.synapseCompList + h.synapseListSize;
			//h.synSpikeListPos = h.synapseTypeList + h.synapseListSize;

			h.spikeListSize 	= 0;
			h.spikeList     	= 0;
			h.synapseWeightList = 0;

			h.tau = memory+sharedMemMatrix/sizeof(ftype);
			h.gmax = h.tau  + 2 * m.synapticChannels->nChannelTypes;
			h.esyn = h.gmax + m.synapticChannels->nChannelTypes;

			h.lastSpike 		= m.lastSpike;
			h.spikeTimeListSize = m.spikeTimeListSize;
			h.threshold        = m.threshold;
			h.minSpikeInterval = m.minSpikeInterval;

		}
		h.nNeurons = nNeurons;

		//if (benchConf.simCommMode == NN_GPU)
		sharedData->matrixList[type][neuron].freeMem();
	}

	*hListPtr = hList;

	return 0;
}

int GpuSimulationControl::updateSpikeListSizeGlobal(int type, int maxSpikesNeuron)
{

	int spikeListSizeMax = 0;
	int synapsePosListTmp = 0;
	for (int n = 0; n < tInfo->nNeurons[type]; n++) {

		//int synapsePosListTmp = 0;
		int synapseListSize = sharedData->matrixList[type][0].synapticChannels->synapseListSize;
		HinesMatrix & m = sharedData->matrixList[type][n];

		if (m.synapticChannels->spikeListSize > spikeListSizeMax)
			spikeListSizeMax = m.synapticChannels->spikeListSize;

		/**
		 * Stores the number of spikes delivered to each neuron
		 */
		sharedData->synData->spikeListSizeGlobal[type][n] = m.synapticChannels->spikeListSize;

		/**
		 * Copies the information about the start position of the spikes at each synapse
		 */
		for (int i=0; i<synapseListSize; i++)
			sharedData->synData->spikeListPosGlobal[type][synapsePosListTmp + i] =
					m.synapticChannels->synSpikeListPos[i];
		synapsePosListTmp += synapseListSize;


		if (spikeListSizeMax > maxSpikesNeuron) {
			printf ("Neuron with %d spikes, more than the max of %d.\n", spikeListSizeMax, maxSpikesNeuron);
			assert(false);
		}
	}

	return spikeListSizeMax;
}

void GpuSimulationControl::transferSynapticSpikeInfoToGpu(int type, int spikeListSizeMax) {

	SynapticData *& synData = sharedData->synData;
    int synapseListSize = sharedData->matrixList[type][0].synapticChannels->synapseListSize;

    checkCUDAError("cp2a:");
    cudaMemcpy(synData->spikeListDevice[type], synData->spikeListGlobal[type], sizeof (ftype) * spikeListSizeMax * tInfo->nNeurons[type], cudaMemcpyHostToDevice);
    checkCUDAError("cp2b:");
    cudaMemcpy(synData->weightListDevice[type], synData->weightListGlobal[type], sizeof (ftype) * spikeListSizeMax * tInfo->nNeurons[type], cudaMemcpyHostToDevice);
    checkCUDAError("cp2c:");
    cudaMemcpy(synData->spikeListPosDevice[type], synData->spikeListPosGlobal[type], sizeof (int) * (1 + synapseListSize * tInfo->nNeurons[type]), cudaMemcpyHostToDevice);
    checkCUDAError("cp2d:");
    cudaMemcpy(synData->spikeListSizeDevice[type], synData->spikeListSizeGlobal[type], sizeof (int) * (tInfo->nNeurons[type] + 1), cudaMemcpyHostToDevice);
    checkCUDAError("cp2e:");

}

void GpuSimulationControl::generateRandomSpikes( int type, RandomSpikeInfo & randomSpkInfo )
{

	int kernelSteps = kernelInfo->nKernelSteps;
	ftype dt = sharedData->dt;

	randomSpkInfo.nRandom = 0;
    for (int neuron = 0; neuron < tInfo->nNeurons[type]; neuron++) {
				HinesMatrix & m = sharedData->matrixList[type][neuron];

				if ((tInfo->kStep + kernelSteps)*m.dt > 9.9999 && sharedData->typeList[type] == PYRAMIDAL_CELL) {
					int32_t spkTime;

					random_r(sharedData->randBuf[tInfo->threadNumber], &spkTime);
					ftype rate = (sharedData->inputSpikeRate) * (kernelSteps * dt);
					if (spkTime / (float) RAND_MAX < rate ) {
						spkTime = (tInfo->kStep + kernelSteps)*dt + (ftype)spkTime/RAND_MAX * (kernelSteps * dt);

						if (benchConf.simCommMode == NN_GPU) {
							assert(randomSpkInfo.nRandom < randomSpkInfo.listSize);
							randomSpkInfo.spikeTimes[randomSpkInfo.nRandom] = spkTime;
							randomSpkInfo.spikeDest[randomSpkInfo.nRandom] = neuron;
						}
						randomSpkInfo.nRandom++;
						if (benchConf.simCommMode == NN_CPU)
							m.synapticChannels->addSpike(0, spkTime, 1);
					}
				}
			}
}

void GpuSimulationControl::performCPUCommunication(int type, int maxSpikesNeuron, int nRandom) {

    int totalNumberSpikes = 0;

    for(int neuron = 0;neuron < tInfo->nNeurons[type];neuron++){
        HinesMatrix & m = sharedData->matrixList[type][neuron];

        /**
         * Updates the spike list when using CPU communications
         */
        if (benchConf.simProcMode == NN_CPU)
        	m.synapticChannels->updateSpikeList(sharedData->dt * (tInfo->kStep + kernelInfo->nKernelSteps));
        else
        	m.synapticChannels->updateSpikeListGpu(sharedData->dt * (tInfo->kStep + kernelInfo->nKernelSteps),
        			sharedData->synData->spikeListGlobal[type], sharedData->synData->weightListGlobal[type],
        			maxSpikesNeuron, tInfo->nNeurons[type], neuron, type);

        totalNumberSpikes += m.synapticChannels->spikeListSize;

        // Used to print spike statistics in the end of the simulation
        sharedData->spkStat->addReceivedSpikes(type, neuron, m.synapticChannels->getAndResetNumberOfAddedSpikes());
    }

    /**
     * Copy synaptic and spike info into the GPU
     */
    if (benchConf.simProcMode == NN_GPU && benchConf.simCommMode == NN_CPU) {
        // The max number of spikes delivered to a single neuron
        int spikeListSizeMax = updateSpikeListSizeGlobal(type, maxSpikesNeuron);
    	transferSynapticSpikeInfoToGpu(type, spikeListSizeMax);
    }
}

void GpuSimulationControl::performGPUCommunications(int type, RandomSpikeInfo & randomSpkInfo) {

	/**
	 * Transfers the list of random spikes to the GPU
	 */
	for(int i = randomSpkInfo.nRandom;i < randomSpkInfo.listSize;i++){
		randomSpkInfo.spikeTimes[i] = -1;
		randomSpkInfo.spikeDest[i] = -1;
	}

	SynapticData *synData = sharedData->synData;
	int threadNumber = tInfo->threadNumber;
	int *nBlocksComm  = kernelInfo->nBlocksComm;
	int *nThreadsComm = kernelInfo->nThreadsComm;

	ftype *randomSpikeTimesDev;
	int *randomSpikeDestDev;
	cudaMalloc((void**)(&randomSpikeTimesDev), sizeof (ftype) * randomSpkInfo.listSize);
	cudaMalloc((void**)(&randomSpikeDestDev), sizeof (int) * randomSpkInfo.listSize);
	cudaMemcpy(randomSpikeTimesDev, randomSpkInfo.spikeTimes, sizeof (ftype) * randomSpkInfo.listSize, cudaMemcpyHostToDevice);
	cudaMemcpy(randomSpikeDestDev, randomSpkInfo.spikeDest, sizeof (int) * randomSpkInfo.listSize, cudaMemcpyHostToDevice);

	if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {

		uint64 connTmp = 0;
		if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE)
			connTmp = gettimeInMilli();

		/**
		 * TODO: Remove Me [MPI]
		 * Used only during debugging to check the number of received spikes per process
		 */
		int *nReceivedSpikesDev0, *nReceivedSpikesDev1, *nReceivedSpikesHost0, *nReceivedSpikesHost1;
		cudaMalloc ( (void **) &nReceivedSpikesDev0, sizeof(int) * nBlocksComm[type]);
		cudaMalloc ( (void **) &nReceivedSpikesDev1, sizeof(int) * nBlocksComm[type]);
		nReceivedSpikesHost0 = (int *)malloc(sizeof(int) * nBlocksComm[type]);
		nReceivedSpikesHost1 = (int *)malloc(sizeof(int) * nBlocksComm[type]);

		//printf (" type=%d %d %d %d\n", type, tInfo->nNeurons[type], nThreadsComm, sharedMemSizeComm);
		performCommunicationsG <<<nBlocksComm[type], nThreadsComm[type], kernelInfo->sharedMemSizeComm>>>(
				tInfo->nNeurons[type], sharedData->connGpuListDevice[type],
				synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
				sharedData->hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
				synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
				randomSpikeTimesDev, randomSpikeDestDev, nReceivedSpikesDev0, nReceivedSpikesDev1);

		/**
		 * TODO: Remove Me [MPI]
		 * Used only during debugging to check the number of received spikes per process
		 */
		cudaMemcpy(nReceivedSpikesHost0, nReceivedSpikesDev0, sizeof(int) * nBlocksComm[type], cudaMemcpyDeviceToHost);
		cudaMemcpy(nReceivedSpikesHost1, nReceivedSpikesDev1, sizeof(int) * nBlocksComm[type], cudaMemcpyDeviceToHost);
		long nReceived0 = 0, nReceived1 = 0;
		for (int k=0; k < nBlocksComm[type]; k++ ) {
			nReceived0 += nReceivedSpikesHost0[k];
			nReceived1 += nReceivedSpikesHost1[k];
		}
		printf("nReceived=0:%5ld|1:%5ld for type %d\n", nReceived0, nReceived1, type);
		cudaFree ( nReceivedSpikesDev0);
		cudaFree ( nReceivedSpikesDev1);
		free(nReceivedSpikesHost0);
		free(nReceivedSpikesHost1);


		if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {
			cudaThreadSynchronize();
			bench.connRead += (gettimeInMilli()-connTmp);
		}

	}

	else if (benchConf.gpuCommBenchMode == GPU_COMM_DETAILED) {

		ftype *tmpDevMemory;
		cudaMalloc ( (void **) &tmpDevMemory, sizeof(ftype) * tInfo->nNeurons[type] * 2 );

		uint64 connTmp = 0;
		if (threadNumber == 0)
			connTmp = gettimeInMilli();
		//int sharedMemComSize = sharedMemorySize/10;
		performCommunicationsG_Step1 <<<nBlocksComm[type], nThreadsComm[type], kernelInfo->sharedMemSizeComm>>> (
				tInfo->nNeurons[type], sharedData->connGpuListDevice[type],
				synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
				sharedData->hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
				synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
				randomSpikeTimesDev, randomSpikeDestDev, tmpDevMemory);

		cudaThreadSynchronize();

		if (threadNumber == 0) {
			bench.connRead += (gettimeInMilli()-connTmp);
			connTmp = gettimeInMilli();
		}

		performCommunicationsG_Step2 <<<nBlocksComm[type], nThreadsComm[type], kernelInfo->sharedMemSizeComm>>> (
				tInfo->nNeurons[type], sharedData->connGpuListDevice[type],
				synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
				sharedData->hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
				synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
				randomSpikeTimesDev, randomSpikeDestDev, tmpDevMemory);

		cudaThreadSynchronize();

		if (threadNumber == 0)
			bench.connWrite += (gettimeInMilli()-connTmp);

		cudaFree (tmpDevMemory);
	}
	//int **countReceivedSpikesCpu(synData->v, nNeurons[type], nBlocksComm[type], synData->nGeneratedSpikesHost);
	cudaThreadSynchronize();
	cudaFree(randomSpikeTimesDev);
	cudaFree(randomSpikeDestDev);
	cudaThreadSynchronize();
}

void GpuSimulationControl::performGpuNeuronalProcessing() {

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

		int nThreadsProc =  tInfo->nNeurons[type]/kernelInfo->nBlocksProc[type];
		if (tInfo->nNeurons[type] % kernelInfo->nBlocksProc[type]) nThreadsProc++;

		checkCUDAError("Before SolveMatrixG Kernel:");
		SynapticData *synData = sharedData->synData;

		//printf (" type=%d %d %d\n", type, nThreadsProc, kernelInfo->sharedMemSizeProc);

		solveMatrixG<<<kernelInfo->nBlocksProc[type], nThreadsProc, kernelInfo->sharedMemSizeProc>>>(
				sharedData->hGpu[type], kernelInfo->nKernelSteps, tInfo->nNeurons[type],
				synData->spikeListDevice[type], synData->weightListDevice[type],
				synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
				synData->vmListDevice[type]);

//		printf("%d\n", type);
	}

}

void GpuSimulationControl::checkVmValues()
{
	SynapticData *synData = sharedData->synData;
    for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
					for (int n = 0; n < tInfo->nNeurons[type]; n++)
						if ( synData->vmListHost[type][n] < -500 || 500 < synData->vmListHost[type][n] || synData->vmListHost[type][n] == 0.000000000000) {
							printf("********* type=%d neuron=%d %.2f\n", type, n, synData->vmListHost[type][n]);
							assert(false);
						}
}



void GpuSimulationControl::prepareGpuSpikeDeliveryStructures()
{
	SynapticData *synData = sharedData->synData;
	int threadNumber = tInfo->threadNumber;

    synData->genSpikeTimeListGpusHost[threadNumber] = (ftype**)(malloc(sizeof (ftype*) * tInfo->totalTypes));
    for (int type = 0; type < tInfo->totalTypes; type++) {
			int genSpikeTimeListSize = GENSPIKETIMELIST_SIZE;
			// The device memory for the types of the current thread are already allocated
			if (tInfo->startTypeThread <= type && type < tInfo->endTypeThread)
				synData->genSpikeTimeListGpusHost[threadNumber][type] = synData->genSpikeTimeListDevice[type];
			else
				cudaMalloc ((void **) &(synData->genSpikeTimeListGpusHost[threadNumber][type]),
						sizeof(ftype) * genSpikeTimeListSize * tInfo->nNeurons[type]);
		}
    // Copies the list of pointers to the genSpikeLists of each type
    cudaMalloc((void**)(&synData->genSpikeTimeListGpusDev[threadNumber]), sizeof (ftype*) * tInfo->totalTypes);
    cudaMemcpy(synData->genSpikeTimeListGpusDev[threadNumber], synData->genSpikeTimeListGpusHost[threadNumber], sizeof (ftype*) * tInfo->totalTypes, cudaMemcpyHostToDevice);
    synData->nGeneratedSpikesGpusHost[threadNumber] = (ucomp**)(malloc(sizeof (ucomp*) * tInfo->totalTypes));
    for(int type = 0;type < tInfo->totalTypes;type++){
        if(tInfo->startTypeThread <= type && type < tInfo->endTypeThread)
            synData->nGeneratedSpikesGpusHost[threadNumber][type] = synData->nGeneratedSpikesDevice[type];

        else
            cudaMalloc((void**)(&(synData->nGeneratedSpikesGpusHost[threadNumber][type])), sizeof (ucomp) * tInfo->nNeurons[type]);

    }
    // Copies the list of pointers to the nGeneratedSpikes of each type
    cudaMalloc((void**)(&synData->nGeneratedSpikesGpusDev[threadNumber]), sizeof (ucomp*) * tInfo->totalTypes);
    cudaMemcpy(synData->nGeneratedSpikesGpusDev[threadNumber], synData->nGeneratedSpikesGpusHost[threadNumber], sizeof (ucomp*) * tInfo->totalTypes, cudaMemcpyHostToDevice);
}

void GpuSimulationControl::createGpuCommunicationStructures()
{

	int *nBlocksComm = kernelInfo->nBlocksComm;

    for(int destType = tInfo->startTypeThread;destType < tInfo->endTypeThread;destType++){
        sharedData->connGpuListHost[destType] =
        		createGpuConnections(sharedData->connInfo, destType, tInfo->nNeurons, nBlocksComm[destType]);

        cudaMalloc((void**)(&(sharedData->connGpuListDevice[destType])), nBlocksComm[destType] * sizeof (ConnGpu));
        cudaMemcpy(sharedData->connGpuListDevice[destType], sharedData->connGpuListHost[destType], nBlocksComm[destType] * sizeof (ConnGpu), cudaMemcpyHostToDevice);

        kernelInfo->nThreadsComm[destType] = sharedData->connGpuListHost[destType][0].nNeuronsGroup;
    }
}

void GpuSimulationControl::addReceivedSpikesToTargetChannelCPU()
{

	cudaThreadSynchronize(); // Used to prevent reads of host memory before the copy

	if( benchConf.simProcMode == NN_CPU && tInfo->nProcesses == 1) {

		for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

			for (int source = 0; source < tInfo->nNeurons[type]; source++) {

				ucomp nGeneratedSpikes = sharedData->matrixList[type][source].nGeneratedSpikes;
				if (nGeneratedSpikes > 0) {
					ftype *spikeTimes = sharedData->matrixList[type][source].spikeTimes;

					// Used to print spike statistics in the end of the simulation
					sharedData->spkStat->addGeneratedSpikes(type, source, spikeTimes, nGeneratedSpikes);
					std::vector<Conn> & connList = sharedData->connection->getConnArray(source + type*CONN_NEURON_TYPE);
					for (int conn=0; conn<connList.size(); conn++) {
						Conn & connStruct = connList[conn];
						SynapticChannels *targetSynapse = sharedData->matrixList[ connStruct.dest / CONN_NEURON_TYPE ][ connStruct.dest % CONN_NEURON_TYPE ].synapticChannels;
						targetSynapse->addSpikeList(connStruct.synapse, nGeneratedSpikes, spikeTimes, connStruct.delay, connStruct.weigth);
					}
				}

			}
		}
	}

	else {

		ConnectionInfo *connInfo = sharedData->connInfo;

		int conn 	= tInfo->threadNumber       * connInfo->nConnections/sharedData->nThreadsCpu;
		int endConn = (tInfo->threadNumber + 1) * connInfo->nConnections/sharedData->nThreadsCpu;
		if (tInfo->threadNumber == sharedData->nThreadsCpu-1)
			endConn = connInfo->nConnections;

		for ( ; conn < endConn; conn++) {

			int dType   = connInfo->dest[conn] / CONN_NEURON_TYPE;
			if (dType < tInfo->startTypeProcess || dType >= tInfo->endTypeProcess)
				continue;

			int dNeuron = connInfo->dest[conn]   % CONN_NEURON_TYPE;
			int sType   = connInfo->source[conn] / CONN_NEURON_TYPE;
			int sNeuron = connInfo->source[conn] % CONN_NEURON_TYPE;

			ucomp nGeneratedSpikes = sharedData->synData->nGeneratedSpikesHost[sType][sNeuron];
			if (nGeneratedSpikes > 0) {
				ftype *spikeTimes = sharedData->synData->genSpikeTimeListHost[sType] + GENSPIKETIMELIST_SIZE * sNeuron;

				SynapticChannels *targetSynapse = sharedData->matrixList[ dType ][ dNeuron ].synapticChannels;
				targetSynapse->addSpikeList(connInfo->synapse[conn], nGeneratedSpikes, spikeTimes, connInfo->delay[conn], connInfo->weigth[conn]);
			}
		}
	}
}

void GpuSimulationControl::copyGeneratedSpikeListsToGPU()
{

	// Consider that each thread has its own device
	for (int type=0; type < tInfo->totalTypes; type++) {
		if (type < tInfo->startTypeThread || tInfo->endTypeThread <= type) {

			cudaMemcpy(sharedData->synData->genSpikeTimeListGpusHost[tInfo->threadNumber][type],
					sharedData->synData->genSpikeTimeListHost[type],
					sizeof(ftype) * GENSPIKETIMELIST_SIZE *tInfo-> nNeurons[type], cudaMemcpyHostToDevice);

			cudaMemcpy(sharedData->synData->nGeneratedSpikesGpusHost[tInfo->threadNumber][type],
					sharedData->synData->nGeneratedSpikesHost[type],
					sizeof(ucomp) * tInfo->nNeurons[type], cudaMemcpyHostToDevice);
		}
	}
}

void GpuSimulationControl::readGeneratedSpikesFromGPU()
{
    /*--------------------------------------------------------------
		 * Reads information from spike sources
		 *--------------------------------------------------------------*/
    if(benchConf.verbose == 1)
        printf("Getting spikes %d\n", tInfo->threadNumber);

    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){
//        cudaMemcpy(sharedData->synData->genSpikeTimeListHost[type],
//        		sharedData->synData->genSpikeTimeListGpusHost[tInfo->threadNumber][type],
//        		sizeof (ftype) * sharedData->hList[type][0].spikeTimeListSize * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

//        cudaMemcpy(sharedData->synData->nGeneratedSpikesHost[type],
//        		sharedData->synData->nGeneratedSpikesGpusHost[tInfo->threadNumber][type],
//        		sizeof (ucomp) * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

        cudaMemcpy(sharedData->synData->genSpikeTimeListHost[type], sharedData->synData->genSpikeTimeListDevice[type],
        		sizeof (ftype) * sharedData->hList[type][0].spikeTimeListSize * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

        cudaMemcpy(sharedData->synData->nGeneratedSpikesHost[type], sharedData->synData->nGeneratedSpikesDevice[type],
        		sizeof (ucomp) * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

        checkCUDAError("Synapses2:");
    }

	/*--------------------------------------------------------------
	 * Used only for Debugging
	 *--------------------------------------------------------------*/
//	int nSpikes[tInfo->totalTypes];
//	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {
//		nSpikes[type] = 0;
//		for (int source = 0; source < tInfo->nNeurons[type]; source++) {
//			nSpikes[type] += sharedData->synData->nGeneratedSpikesHost[type][source];
//		}
//		printf("type=%d nSpikes=%d\n", type, nSpikes[type]);
//	}
	//--------------------------------------------------------------

}

void GpuSimulationControl::configureGpuKernel()
{

    /*--------------------------------------------------------------
	 * Select the device attributed to each thread
	 *--------------------------------------------------------------*/
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice((tInfo->threadNumber + (2 * tInfo->currProcess) + 2) % nDevices);
    cudaGetDevice(&(tInfo->deviceNumber));
    tInfo->prop = new struct cudaDeviceProp;
    cudaGetDeviceProperties(tInfo->prop, tInfo->deviceNumber);

    checkCUDAError("Device selection:");
    //--------------------------------------------------------------
    /*--------------------------------------------------------------
	 * Configure number of threads and shared memory size for each kernel
	 *--------------------------------------------------------------*/
    kernelInfo->maxThreadsProc = 64;
    kernelInfo->maxThreadsComm = 16;
    kernelInfo->sharedMemSizeProc = 15 * 1024; // Valid for capability 1.x (16kB)
    kernelInfo->sharedMemSizeComm = 15 * 1024; // Valid for capability 1.x (16kB)
    if(tInfo->prop->major == 2){
        kernelInfo->maxThreadsProc = 256;
        kernelInfo->maxThreadsComm = 32; // or can be 64
        kernelInfo->sharedMemSizeProc = 47 * 1024; // Valid for capability 2.x (48kB)
        kernelInfo->sharedMemSizeComm = 15 * 1024; // Valid for capability 2.x (48kB)
    }
    //--------------------------------------------------------------
    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){
        // Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsProc condition
        kernelInfo->nBlocksProc[type] = tInfo->prop->multiProcessorCount * (tInfo->nNeurons[type] / kernelInfo->maxThreadsProc / tInfo->prop->multiProcessorCount);
        if(tInfo->nNeurons[type] % kernelInfo->maxThreadsProc != 0 || (tInfo->nNeurons[type] / kernelInfo->maxThreadsProc) % kernelInfo->maxThreadsProc != 0)
            kernelInfo->nBlocksProc[type] += tInfo->prop->multiProcessorCount;

    }

    for(int destType = tInfo->startTypeThread;destType < tInfo->endTypeThread;destType++){

    	// Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsComm condition
    	kernelInfo->nBlocksComm[destType] = tInfo->prop->multiProcessorCount * (tInfo->nNeurons[destType] / kernelInfo->maxThreadsComm / tInfo->prop->multiProcessorCount);
        if(tInfo->nNeurons[destType] % kernelInfo->maxThreadsComm != 0 || (tInfo->nNeurons[destType] / kernelInfo->maxThreadsComm) % kernelInfo->maxThreadsComm != 0)
        	kernelInfo->nBlocksComm[destType] += tInfo->prop->multiProcessorCount;
    }

    //	if (nComp0 <= 4) nThreads = (sizeof (ftype) == 4) ? 196 : 96;
    //	else if (nComp0 <= 8) nThreads = (sizeof (ftype) == 4) ? 128 : 64;
    //	else if (nComp0 <= 12) nThreads = (sizeof (ftype) == 4) ? 96 : 32;
    //	else if (nComp0 <= 16) nThreads = (sizeof (ftype) == 4) ? 64 : 32;

}

void GpuSimulationControl::updateSharedDataInfo()
{
    sharedData->dt = sharedData->matrixList[tInfo->startTypeThread][0].dt;
    sharedData->hList = new HinesStruct*[tInfo->totalTypes];
    sharedData->hGpu = new HinesStruct*[tInfo->totalTypes];
    sharedData->synData = new SynapticData;
    sharedData->synData->spikeListDevice = 0;
    sharedData->synData->totalTypes = tInfo->totalTypes;
    sharedData->neuronInfoWriter = new NeuronInfoWriter(tInfo);
    sharedData->spkStat = new SpikeStatistics(tInfo->nNeurons, tInfo->totalTypes, tInfo->sharedData->typeList, tInfo->startTypeProcess, tInfo->endTypeProcess);
    kernelInfo->nThreadsComm = new int[tInfo->totalTypes];
    kernelInfo->nBlocksComm = new int[tInfo->totalTypes];
    kernelInfo->nBlocksProc = new int[tInfo->totalTypes];
}

/***************************************************************************
 * Controls the simulation and kernel calls
 ***************************************************************************/
//int GpuSimulationControl::launchGpuExecution() {
//
//    int *nNeurons = tInfo->nNeurons;
//    int startTypeThread = tInfo->startTypeThread;
//    int endTypeThread = tInfo->endTypeThread;
//    int threadNumber = tInfo->threadNumber;
//
//    pthread_mutex_lock(sharedData->mutex);
//    if(sharedData->hList == 0)
//    	updateSharedDataInfo();
//    pthread_mutex_unlock(sharedData->mutex);
//
//    //Synchronize threads before starting
//    syncCpuThreads();
//
//    bench.matrixSetup = gettimeInMilli();
//    bench.matrixSetupF = (bench.matrixSetup - bench.start) / 1000.;
//
//    /*--------------------------------------------------------------
//	 * Configure the Device and GPU kernel information
//	 *--------------------------------------------------------------*/
//    configureGpuKernel();
//
//    /*--------------------------------------------------------------
//	 * Initializes the benchmark counters
//	 *--------------------------------------------------------------*/
//    if(threadNumber == 0){
//        bench.totalHinesKernel = 0;
//        bench.totalConnRead = 0;
//        bench.totalConnWait = 0;
//        bench.totalConnWrite = 0;
//    }
//    /*--------------------------------------------------------------
//	 * Allocates the memory on the GPU for neuron information and transfers the data
//	 *--------------------------------------------------------------*/
//    for(int type = startTypeThread;type < endTypeThread;type++){
//        printf("GPU allocation with %d neurons, %d comparts on device %d thread %d process %d.\n", nNeurons[type], sharedData->matrixList[type][0].nComp, tInfo->deviceNumber, threadNumber, tInfo->currProcess);
//        prepareExecution(type);
//    }
//
//    /*--------------------------------------------------------------
//	 * Allocates the memory on the GPU for the communications and transfers the data
//	 *--------------------------------------------------------------*/
//    prepareSynapses();
//    SynapticData *synData = sharedData->synData;
//    int nKernelSteps = kernelInfo->nKernelSteps;
//
//    /*--------------------------------------------------------------
//	 * Sends the complete data to the GPUs
//	 *--------------------------------------------------------------*/
//    for(int type = startTypeThread;type < endTypeThread;type++){
//        cudaMalloc((void**)((((&(sharedData->hGpu[type]))))), sizeof (HinesStruct) * nNeurons[type]);
//        cudaMemcpy(sharedData->hGpu[type], sharedData->hList[type], sizeof (HinesStruct) * nNeurons[type], cudaMemcpyHostToDevice);
//        checkCUDAError("Memory Allocation:");
//    }
//
//    /*--------------------------------------------------------------
//	 * Prepare the spike list in the format used in the GPU
//	 *--------------------------------------------------------------*/
//    int maxSpikesNeuron = 5000; //5000;
//    for(int type = startTypeThread;type < endTypeThread;type++){
//        int neuronSpikeListSize = maxSpikesNeuron * nNeurons[type];
//        synData->spikeListGlobal[type] = (ftype*)((((malloc(sizeof (ftype) * neuronSpikeListSize)))));
//        synData->weightListGlobal[type] = (ftype*)((((malloc(sizeof (ftype) * neuronSpikeListSize)))));
//        cudaMalloc((void**)((((&(synData->spikeListDevice[type]))))), sizeof (ftype) * neuronSpikeListSize);
//        cudaMalloc((void**)((((&(synData->weightListDevice[type]))))), sizeof (ftype) * neuronSpikeListSize);
//        if(type == 0)
//            printf("Spike List size of %.3f MB for each type.\n", sizeof (ftype) * neuronSpikeListSize / 1024. / 1024.);
//
//    }
//
//    /*--------------------------------------------------------------
//	 * Creates the connections between the neurons
//	 *--------------------------------------------------------------*/
//    if (threadNumber == 0) {
//		sharedData->connection = new Connections();
//		sharedData->connection->connectRandom (tInfo );
//
//		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
//			sharedData->connGpuListHost   = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
//			sharedData->connGpuListDevice = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
//		}
//
//		sharedData->connInfo = sharedData->connection->getConnectionInfo();
//	}
//
//    /*--------------------------------------------------------------
//	 * [MPI] Send the connection list to other MPI processes
//	 *--------------------------------------------------------------*/
//#ifdef MPI_GPU_NN
//    if(threadNumber == 0)
//        mpiAllGatherConnections();
//#endif
//
//    /*--------------------------------------------------------------
//	 * Guarantees that all connections have been setup
//	 *--------------------------------------------------------------*/
//    syncCpuThreads();
//
//    /*--------------------------------------------------------------
//	 * Creates the connection list for usage in the GPU communication
//	 *--------------------------------------------------------------*/
//    if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM)
//    	createGpuCommunicationStructures();
//
//    /*--------------------------------------------------------------
//	 * Prepare the lists of generated spikes used for GPU spike delivery
//	 *--------------------------------------------------------------*/
//    if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM)
//    	prepareGpuSpikeDeliveryStructures();
//
//    /*--------------------------------------------------------------
//	 * [MPI] Prepare the genSpikeListTime to receive the values from other processes
//	 *--------------------------------------------------------------*/
//#ifdef MPI_GPU_NN
//    if (threadNumber == 0) {
//		for (int type = 0; type < tInfo->totalTypes; type++) {
//			if ( type < tInfo->startTypeProcess || tInfo->endTypeProcess <= type ) {
//				int spikeTimeListSize = GENSPIKETIMELIST_SIZE;
//				synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
//				synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
//			}
//		}
//	}
//#endif
//
//    /*--------------------------------------------------------------
//	 * Synchronize threads before beginning [Used only for Benchmarking]
//	 *--------------------------------------------------------------*/
//    //syncCpuThreads(sharedData);
//    printf("Launching GPU kernel with %d blocks and %d (+1) threads per block for types %d-%d for thread %d "
//    		"on device %d [%s|%d.%d|MP=%d|G=%dMB|S=%dkB].\n", kernelInfo->nBlocksProc[startTypeThread],
//    		nNeurons[startTypeThread] / kernelInfo->nBlocksProc[startTypeThread], startTypeThread, endTypeThread - 1,
//    		threadNumber, tInfo->deviceNumber, tInfo->prop->name, tInfo->prop->major, tInfo->prop->minor,
//    		tInfo->prop->multiProcessorCount, (int)((tInfo->prop->totalGlobalMem / 1024 / 1024)),
//    		(int)((tInfo->prop->sharedMemPerBlock / 1024)));
//
//    if(threadNumber == 0){
//        bench.execPrepare = gettimeInMilli();
//        bench.execPrepareF = (bench.execPrepare - bench.matrixSetup) / 1000.;
//    }
//
//    /*--------------------------------------------------------------
//	 * Solves the matrix for n steps
//	 *--------------------------------------------------------------*/
//    ftype dt = sharedData->dt;
//    int nSteps = sharedData->totalTime / dt;
//
//    for (kStep = 0; kStep < nSteps; kStep += nKernelSteps) {
//
//		// Synchronizes the thread to wait for the communication
//
//		if (threadNumber == 0 && kStep % 100 == 0)
//			printf("Starting Kernel %d -----------> %d \n", threadNumber, kStep);
//
//		if (threadNumber == 0) // Benchmarking
//			bench.kernelStart  = gettimeInMilli();
//
//		for (int type = startTypeThread; type < endTypeThread; type++) {
//
//			int nThreadsProc =  nNeurons[type]/kernelInfo->nBlocksProc[type];
//			if (nNeurons[type] % kernelInfo->nBlocksProc[type]) nThreadsProc++;
//
//			checkCUDAError("Before SolveMatrixG Kernel:");
//			solveMatrixG<<<kernelInfo->nBlocksProc[type], nThreadsProc, kernelInfo->sharedMemSizeProc>>>(
//					sharedData->hGpu[type], nKernelSteps, nNeurons[type],
//					synData->spikeListDevice[type], synData->weightListDevice[type],
//					synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
//					synData->vmListDevice[type]);
//		}
//
//		cudaThreadSynchronize();
//
//		if (threadNumber == 0) // Benchmarking
//			bench.kernelFinish = gettimeInMilli();
//
//		/*--------------------------------------------------------------
//		 * Reads information from spike sources fromGPU
//		 *--------------------------------------------------------------*/
//		readGeneratedSpikesFromGPU();
//
//		/*--------------------------------------------------------------
//		 * Adds the generated spikes to the target synaptic channel
//		 * Used only for communication processing in the CPU
//		 *--------------------------------------------------------------*/
//		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM)
//			addReceivedSpikesToTargetChannelCPU();
//
//		if (threadNumber == 0 && benchConf.gpuCommMode == CPU_COMM)
//			bench.connRead = gettimeInMilli();
//		else if (threadNumber == 0) {
//			bench.connRead = 0;
//			bench.connWrite = 0;
//		}
//		/*--------------------------------------------------------------
//		 * Synchronize threads before communication
//		 *--------------------------------------------------------------*/
//		syncCpuThreads();
//
//		/*--------------------------------------------------------------
//		 * [MPI] Send the list of generated spikes to other processes
//		 *--------------------------------------------------------------*/
//#ifdef MPI_GPU_NN
//		broadcastGeneratedSpikesMPISync();
//#endif
//
//		if (threadNumber == 0)
//			bench.connWait = gettimeInMilli();
//
//		/*--------------------------------------------------------------
//		 * Used to print spike statistics in the end of the simulation
//		 *--------------------------------------------------------------*/
//		if (threadNumber == 0)
//			for (int type=0; type < tInfo->totalTypes; type++)
//				for (int c=0; c<nNeurons[type]; c++)
//					sharedData->spkStat->addGeneratedSpikes(type, c, NULL, synData->nGeneratedSpikesHost[type][c]);
//
//		/*--------------------------------------------------------------
//		 * Copy the Vm from GPUs to the CPU memory
//		 *--------------------------------------------------------------*/
//		if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1)
//			for (int type = startTypeThread; type < endTypeThread; type++)
//				cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);
//
//		/*--------------------------------------------------------------
//		 * Writes Vm to file at the end of each kernel execution
//		 *--------------------------------------------------------------*/
//		if (benchConf.assertResultsAll == 1)
//			checkVmValues();
//
//		/*--------------------------------------------------------------
//		 * Check if Vm is ok for all neurons
//		 *--------------------------------------------------------------*/
//		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1)
//			sharedData->neuronInfoWriter->writeVmToFile(kStep);
//
//
//		/*--------------------------------------------------------------
//		 * Copy the generatedSpikeList to the GPUs
//		 *--------------------------------------------------------------*/
//		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM)
//			copyGeneratedSpikeListsToGPU();
//
//		/*-------------------------------------------------------
//		 * Perform Communications
//		 *-------------------------------------------------------*/
//		for (int type = startTypeThread; type < endTypeThread; type++) {
//
//			/*-------------------------------------------------------
//			 *  Generates random spikes for the network
//			 *-------------------------------------------------------*/
//			RandomSpikeInfo randomSpkInfo;
//			randomSpkInfo.listSize = sharedData->inputSpikeRate * nKernelSteps * dt * nNeurons[type] * 3;
//			randomSpkInfo.spikeTimes = new ftype[ randomSpkInfo.listSize ];
//			randomSpkInfo.spikeDest = new int[ randomSpkInfo.listSize ];
//			generateRandomSpikes(type, randomSpkInfo);
//
//			/*-------------------------------------------------------
//			 * Perform GPU Communications
//			 *-------------------------------------------------------*/
//			if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM)
//				performGPUCommunications(type, randomSpkInfo);
//
//			delete []randomSpkInfo.spikeTimes;
//			delete []randomSpkInfo.spikeDest;
//
//			/*-------------------------------------------------------
//			 * Perform CPU Communications
//			 *-------------------------------------------------------*/
//			if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM)
//				performCPUCommunication(type, maxSpikesNeuron, randomSpkInfo.nRandom);
//		}
//
//		if (threadNumber == 0)
//			if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE || benchConf.gpuCommMode == CPU_COMM)
//				bench.connWrite = gettimeInMilli();
//
//		if (threadNumber == 0 && benchConf.printSampleVms == 1)
//			sharedData->neuronInfoWriter->writeSampleVm(kStep);
//
//		if (benchConf.printAllSpikeTimes == 1)
//			if (threadNumber == 0) // Uses only data from SpikeStatistics::addGeneratedSpikes
//				sharedData->spkStat->printKernelSpikeStatistics((kStep+nKernelSteps)*dt);
//
//		if (threadNumber == 0)
//			updateBenchmark();
//
//
//	}
//    // --------------- Finished the simulation ------------------------------------
//
//	if (threadNumber == 0) {
//		bench.execExecution  = gettimeInMilli();
//		bench.execExecutionF = (bench.execExecution - bench.execPrepare)/1000.;
//	}
//
//	if (threadNumber == 0) {
//		//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[0])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[0])[nKernelSteps-1]);
//		//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[1])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[1])[nKernelSteps-1]);
//	}
//
//	// Used to print spike statistics in the end of the simulation
//	if (threadNumber == 0)
//		sharedData->spkStat->printSpikeStatistics("spikeGpu.dat", sharedData->totalTime, bench, tInfo->startTypeProcess, tInfo->endTypeProcess);
//
//	// TODO: Free CUDA Memory
//	for (int type = startTypeThread; type < endTypeThread; type++) {
//		cudaFree(synData->spikeListDevice[type]);
//		cudaFree(synData->weightListDevice[type]);
//		free(synData->spikeListGlobal[type]);
//		free(synData->weightListGlobal[type]);
//	}
//
//	if (threadNumber == 0) {
//		delete[] kernelInfo->nBlocksComm;
//		delete[] kernelInfo->nThreadsComm;
//		delete sharedData->neuronInfoWriter;
//	}
//
//	printf("Finished GPU execution.\n" );
//
//	return 0;
//}
