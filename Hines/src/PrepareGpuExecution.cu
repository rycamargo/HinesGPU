#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include <cassert>
#include <cstdlib>
#include <pthread.h>

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

extern __global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal);
extern ConnGpu* createGpuConnections( Connections *conn, int destType, int nTypes, int *nNeurons, int nGroups );
extern int **countReceivedSpikesCpu(ConnGpu *connGpuList, int nNeurons, int nGroups, ucomp **nGeneratedSpikes);
extern __global__ void performCommunicationsG(int nNeurons, ConnGpu *connGpuListDev, ucomp **nGeneratedSpikesDev, ftype **genSpikeTimeListDev,
		HinesStruct *hList, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListSizeGlobal,
		ftype *randomSpikeTimesDev, int *randomSpikeDestDev);
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

void prepareSynapses(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int threadDevice, int nDevices) {

	HinesMatrix **matrixList = sharedData->matrixList;
	HinesStruct **hList = sharedData->hList;
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

		synData->spikeTimeListHost     = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->spikeTimeListDevice   = (ftype **) malloc (sizeof(ftype *)  * totalTypes);

		synData->nGeneratedSpikesHost     = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);
		synData->nGeneratedSpikesDevice   = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
			synData->spikeTimeListGpusDev  = (ftype ***) malloc (sizeof(ftype **) * nDevices);
			synData->spikeTimeListGpusHost = (ftype ***) malloc (sizeof(ftype **) * nDevices);
			synData->nGeneratedSpikesGpusDev  = (ucomp ***) malloc (sizeof(ucomp **) * nDevices);
			synData->nGeneratedSpikesGpusHost = (ucomp ***) malloc (sizeof(ucomp **) * nDevices);
		}
	}
	pthread_mutex_unlock (sharedData->mutex);


	/**
	 * Prepare the delivered spike related lists
	 * - spikeListPos and spikeListSize
	 */
	for (int type = startType; type < endType; type++) {
		synData->spikeListGlobal[type] = 0;
		synData->weightListGlobal[type] = 0;
		int totalListPosSize = 1 + matrixList[type][0].synapticChannels->synapseListSize * nNeurons[type];
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
	for (int type = startType; type < endType; type++) {
		int spikeTimeListSize = matrixList[type][0].spikeTimeListSize;

		synData->spikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
		cudaMalloc ((void **) &(synData->spikeTimeListDevice[type]), sizeof(ftype) * spikeTimeListSize * nNeurons[type]);

		synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->nGeneratedSpikesDevice[type]), sizeof(ucomp) * nNeurons[type]);

		for (int neuron = 0; neuron < nNeurons[type]; neuron++ ) {
			HinesStruct & h = hList[type][neuron];
			h.spikeTimes  = synData->spikeTimeListDevice[type] + spikeTimeListSize * neuron;
			h.nGeneratedSpikes = synData->nGeneratedSpikesDevice[type];// + neuron;
		}
	}
}

int prepareExecution(HinesMatrix *matrixList, int nNeurons, int kernelSteps, HinesStruct **hListPtr, int type) {

	HinesStruct *hList = (HinesStruct *)malloc(nNeurons*sizeof(HinesStruct)); //new HinesStruct[nNeurons];

	HinesMatrix & m0 = matrixList[0];
	int nComp = m0.nComp;
	int nCompActive = m0.activeChannels->getCompListSize();
	int nSynaptic = m0.synapticChannels->synapseListSize;

	int sharedMemMatrix    = sizeof(ftype) * (3*nComp + m0.mulListSize + m0.leftListSize); //+ nComp*nComp;
	int sharedMemSynaptic  = sizeof(ftype) * 4 * m0.synapticChannels->nChannelTypes;
	int sharedMemTotal = sharedMemMatrix + sharedMemSynaptic;

	int exclusiveMemMatrix   = sizeof(ftype) * (5*nComp + m0.leftListSize);
	int exclusiveMemActive   = sizeof(ftype) * (nCompActive*7);
	//int exclusiveMemSynaptic = sizeof(ftype) * m.spikeTimeListSize;
	int exclusiveMemTotal = exclusiveMemMatrix + exclusiveMemActive + sizeof(ftype)*nComp*kernelSteps;

	/**
	 * Allocates memory to all neurons
	 */
	printf("TotalMem = %10.3f MB\n",(sharedMemTotal + exclusiveMemTotal * nNeurons)/(1024.*1024.));
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

		HinesMatrix & m = matrixList[neuron];

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
			cudaMemcpy(h.vmTimeSerie + nComp*kernelSteps, m.activeChannels->memory, exclusiveMemActive, cudaMemcpyHostToDevice);

			h.n = h.vmTimeSerie + nComp*kernelSteps;
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
		matrixList[neuron].freeMem();
	}

	*hListPtr = hList;

	return 0;
}

/***************************************************************************
 * This part is executed in every integration step
 ***************************************************************************/
int launchGpuExecution(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int totalTypes, int threadNumber) {

	HinesMatrix **matrixList = sharedData->matrixList;



	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
//	int threadDevice = (threadNumber+1)%nDevices;
//	cudaSetDevice(threadDevice);
	int threadDevice = threadNumber;
	cudaSetDevice((threadNumber+1)%nDevices);
	checkCUDAError("Device selection:");

	ftype totalTime = sharedData->totalTime;
	int kernelSteps = sharedData->nKernelSteps;

	FILE *nSpkfile 	  = fopen("nSpikeKernel.dat", "w");
	FILE *lastSpkfile = fopen("lastSpikeKernel.dat", "w");

	/**
	 * Only one Gpu should update it
	 */
	pthread_mutex_lock (sharedData->mutex);
	if (sharedData->hList == 0) {
		sharedData->hList   = new HinesStruct *[totalTypes];
		sharedData->hGpu    = new HinesStruct *[totalTypes];
		sharedData->synData = new SynapticData;
		sharedData->synData->spikeListDevice = 0;
		sharedData->synData->totalTypes = totalTypes;

		if (threadNumber == 0) {
			bench.totalHinesKernel = 0;
			bench.totalConnRead    = 0;
			bench.totalConnWait    = 0;
			bench.totalConnWrite   = 0;
		}
	}
	pthread_mutex_unlock (sharedData->mutex);

	HinesStruct **hList   = sharedData->hList;
	HinesStruct **hGpu  = sharedData->hGpu;

	// Allocates the memory on the GPU and transfers the data
	for (int type = startType; type < endType; type++) {
		prepareExecution(matrixList[type], nNeurons[type], kernelSteps, &(hList[type]), type);
		if (benchConf.verbose)
			printf("GPU allocation with %d neurons, %d comparts on device %d.\n", nNeurons[type], matrixList[type][0].nComp, threadDevice);
	}

	ftype dt = matrixList[startType][0].dt;
	int nSteps = totalTime / dt;

	prepareSynapses(sharedData, nNeurons, startType, endType, threadDevice, nDevices);
	SynapticData *synData = sharedData->synData;

	/**
	 * Send data to Gpu
	 */
	for (int type = startType; type < endType; type++) {
		cudaMalloc((void **)&(hGpu[type]), sizeof(HinesStruct)*nNeurons[type]);
		cudaMemcpy(hGpu[type], hList[type], sizeof(HinesStruct)*nNeurons[type], cudaMemcpyHostToDevice);
		checkCUDAError("Memory Allocation:");
	}

	int nComp0 = matrixList[startType][0].nComp;
	int timeSerieMemSize = sizeof(ftype) * (nComp0 * kernelSteps);

	/**
	 * Perform the execution
	 */
	int nThreads = 64; // 64
//	if (nComp0 <= 4) nThreads = (sizeof (ftype) == 4) ? 196 : 96;
//	else if (nComp0 <= 8) nThreads = (sizeof (ftype) == 4) ? 128 : 64;
//	else if (nComp0 <= 12) nThreads = (sizeof (ftype) == 4) ? 96 : 32;
//	else if (nComp0 <= 16) nThreads = (sizeof (ftype) == 4) ? 64 : 32;

	int nBlocks = nNeurons[startType]/nThreads;
	if (nNeurons[startType]%nThreads != 0) nBlocks++;
	if (nBlocks == 1) nThreads = nNeurons[startType];
	printf("Launching GPU kernel with %d blocks and %d threads per block types:%d-%d on device %d.\n", nBlocks, nThreads, startType, endType, (threadNumber+1)%nDevices);

	FILE *outFile;
	outFile = fopen("sampleVm.dat", "w");

	FILE *vmKernelFile;
	vmKernelFile = fopen("vmKernel.dat", "w");

	int nVmTimeSeries = 4;
	ftype **vmTimeSerie  = (ftype **)malloc(sizeof(ftype *) * nVmTimeSeries);
	for (int k=0; k<nVmTimeSeries; k++)
		vmTimeSerie[k] = (ftype *)malloc(timeSerieMemSize);

	int sharedMemorySize = 15 * 1024; // Valid for CUDA 1.3 (16kb)

	if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom (sharedData->pyrConnRatio, sharedData->inhConnRatio,
				sharedData->typeList, totalTypes, nNeurons, sharedData, threadNumber);

		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
			sharedData->connGpuListHost   = (ConnGpu **)malloc(totalTypes * sizeof(ConnGpu *));
			sharedData->connGpuListDevice = (ConnGpu **)malloc(totalTypes * sizeof(ConnGpu *));
		}
	}

	/**
	 * Prepare the spike list in the format used in the GPU
	 */
	int maxSpikesNeuron = 3000; //5000;
	for (int type = startType; type < endType; type++) {
		int neuronSpikeListSize = maxSpikesNeuron * nNeurons[type];
		synData->spikeListGlobal[type]  = (ftype *) malloc( sizeof(ftype) * neuronSpikeListSize );
		synData->weightListGlobal[type] = (ftype *) malloc( sizeof(ftype) * neuronSpikeListSize );
		cudaMalloc ((void **) &(synData->spikeListDevice[type] ), sizeof(ftype) * neuronSpikeListSize );
		cudaMalloc ((void **) &(synData->weightListDevice[type]), sizeof(ftype) * neuronSpikeListSize );

		if (type == 0) printf ("Spike List size of %.3f MB for each type.\n", sizeof(ftype) * neuronSpikeListSize / 1024. / 1024.);
	}

	/**
	 * Guarantees that all connectionshave been setup
	 */
	pthread_mutex_lock (sharedData->mutex);
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);

	/**
	 * Creates the connections for usage in the GPU communications mode
	 */
	int nGroups[totalTypes];
	if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
		for (int type = startType; type < endType; type++) {
			int ng = 16;
			nGroups[type] = ng * ((nNeurons[type] / ng / ng) + 1);
			if (nNeurons[type] % ng == 0  && (nNeurons[type] / ng) % ng == 0) nGroups[type] -= ng;
			sharedData->connGpuListHost[type] =
					createGpuConnections( sharedData->connection, type, totalTypes, nNeurons, nGroups[type]);
			cudaMalloc( (void **)&(sharedData->connGpuListDevice[type]), nGroups[type] * sizeof(ConnGpu) );
			cudaMemcpy(sharedData->connGpuListDevice[type], sharedData->connGpuListHost[type], nGroups[type] * sizeof(ConnGpu), cudaMemcpyHostToDevice);

		}

	}

	/**
	 * Prepare the spike control lists used for GPU spike delivery
	 */
	if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
		synData->spikeTimeListGpusHost[threadDevice] = (ftype **) malloc( sizeof(ftype *) * totalTypes );
		for (int type = 0; type < totalTypes; type++) {
			int spikeTimeListSize = matrixList[type][0].spikeTimeListSize;
			if (startType <= type && type < endType)
				synData->spikeTimeListGpusHost[threadDevice][type] = synData->spikeTimeListDevice[type];
			else
				cudaMalloc ((void **) &(synData->spikeTimeListGpusHost[threadDevice][type]),
						sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
		}
		cudaMalloc( (void **) &synData->spikeTimeListGpusDev[threadDevice], sizeof(ftype *) * totalTypes);
		cudaMemcpy(synData->spikeTimeListGpusDev[threadDevice], synData->spikeTimeListGpusHost[threadDevice],
				sizeof(ftype *) * totalTypes, cudaMemcpyHostToDevice);

		synData->nGeneratedSpikesGpusHost[threadDevice] = (ucomp **) malloc( sizeof(ucomp *) * totalTypes );
		for (int type = 0; type < totalTypes; type++) {
			if (startType <= type && type < endType)
				synData->nGeneratedSpikesGpusHost[threadDevice][type] = synData->nGeneratedSpikesDevice[type];
			else
				cudaMalloc ((void **) &(synData->nGeneratedSpikesGpusHost[threadDevice][type]), sizeof(ucomp) * nNeurons[type]);
		}
		cudaMalloc( (void **) &synData->nGeneratedSpikesGpusDev[threadDevice], sizeof(ucomp *) * totalTypes);
		cudaMemcpy(synData->nGeneratedSpikesGpusDev[threadDevice], synData->nGeneratedSpikesGpusHost[threadDevice],
				sizeof(ucomp *) * totalTypes, cudaMemcpyHostToDevice);
	}

	/**
	 * Synchronize threads before beginning
	 */
	pthread_mutex_lock (sharedData->mutex);
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);


	if (threadNumber == 0) {
		bench.execPrepare  = gettimeInMilli();
		bench.execPrepareF = (bench.execPrepare - bench.matrixSetup)/1000.;
	}


	/**
	 * Solves the matrix for n steps
	 */
	for (int kStep = 0; kStep < nSteps; kStep += kernelSteps) {

		if (threadNumber == 0 && kStep % 100 == 0)
			printf("Starting Kernel %d -----------> %d \n", threadNumber, kStep);

		if (threadNumber == 0) // Benchmarking
			bench.kernelStart  = gettimeInMilli();

		for (int type = startType; type < endType; type++) {

			checkCUDAError("Before SolveMatrixG Kernel:");
			solveMatrixG<<<nBlocks,nThreads, sharedMemorySize>>>(hGpu[type], kernelSteps, nNeurons[type],
					synData->spikeListDevice[type], synData->weightListDevice[type],
					synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
					synData->vmListDevice[type]);
			checkCUDAError("SolveMatrixG Kernel:");
		}

		cudaThreadSynchronize();
		if (threadNumber == 0) // Benchmarking
			bench.kernelFinish = gettimeInMilli();

		if (benchConf.verbose == 1) {
			//cudaThreadSynchronize();
			checkCUDAError("Sync:");
			printf("Finished Kernel %d -----------> %d \n", threadNumber, kStep);
		}

		/**
		 * Check if Vm is ok for all neurons
		 */
		if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1) {
			for (int type = startType; type < endType; type++) {

				cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);

				if (benchConf.assertResultsAll == 1) {
					for (int n = 0; n < nNeurons[type]; n++) {
						if ( synData->vmListHost[type][n] < -500 || 500 < synData->vmListHost[type][n] || synData->vmListHost[type][n] == 0.000000000000) {
							printf("********* type=%d neuron=%d %.2f\n", type, n, synData->vmListHost[type][n]);
							assert(false);
						}
					}
				}
			}
		}

		if (benchConf.verbose == 1)
			printf("Getting spikes %d\n", threadNumber);

		/**
		 * Reads information from spike sources
		 * */
		for (int type = startType; type < endType; type++) {

			HinesStruct & hType = hList[type][0];

			cudaMemcpy(synData->spikeTimeListHost[type],    synData->spikeTimeListDevice[type],    
				   sizeof(ftype) * hType.spikeTimeListSize * nNeurons[type], cudaMemcpyDeviceToHost);
			cudaMemcpy(synData->nGeneratedSpikesHost[type], synData->nGeneratedSpikesDevice[type], 
				   sizeof(ucomp) * nNeurons[type], cudaMemcpyDeviceToHost);
			checkCUDAError("Synapses2:");

			cudaThreadSynchronize(); // Used to prevent reads of local memory before the copy

		}

		for (int type = startType; type < endType; type++) {

			HinesStruct & hType = hList[type][0];

			for (int source = 0; source < nNeurons[type]; source++) {

				ucomp nGeneratedSpikes = synData->nGeneratedSpikesHost[type][source];
				if (nGeneratedSpikes > 0) {

					ftype *spikeTimes = synData->spikeTimeListHost[type] + hType.spikeTimeListSize * source;

					// Used to print spike statistics in the end of the simulation
					sharedData->spkStat->addGeneratedSpikes(type, source, spikeTimes, nGeneratedSpikes);

					/**
					 * Adds the generated spikes to the target synaptic channel
					 * Used only for communication processing in the CPU
					 */
					if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM) {
						std::vector<Conn> & connList = sharedData->connection->getConnArray(source + type*CONN_NEURON_TYPE);
						for (int conn=0; conn<connList.size(); conn++) {
							Conn & connStruct = connList[conn];
							SynapticChannels *targetSynapse = matrixList[ connStruct.dest / CONN_NEURON_TYPE ][ connStruct.dest % CONN_NEURON_TYPE ].synapticChannels;
							targetSynapse->addSpikeList(connStruct.synapse, nGeneratedSpikes, spikeTimes, connStruct.delay, connStruct.weigth);
						}
					}

				}
			}
		}

		if (threadNumber == 0 && benchConf.gpuCommMode == CPU_COMM) {
			bench.connRead = gettimeInMilli();
		}
		else if (threadNumber == 0){
			bench.connRead = 0;
			bench.connWrite = 0;

		}


		/**
		 * Synchronize threads before communication
		 */
		pthread_mutex_lock (sharedData->mutex);
		sharedData->nBarrier++;
		if (sharedData->nBarrier < sharedData->nThreadsCpu) {
			pthread_cond_wait(sharedData->cond, sharedData->mutex);
		}
		else {
			sharedData->nBarrier = 0;
			pthread_cond_broadcast(sharedData->cond);
		}
		pthread_mutex_unlock (sharedData->mutex);

		if (threadNumber == 0)
			bench.connWait = gettimeInMilli();


		/**
		 * Prints the Vm at the end of each kernel execution
		 */
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1) {
			for (int type = 0; type < totalTypes; type++) {
				fprintf(vmKernelFile, "dt=%-10.2f\ttype=%d\t", dt * (kStep + kernelSteps), type);
				for (int n = 0; n < nNeurons[type]; n++)
					fprintf(vmKernelFile, "%10.2f\t", sharedData->synData->vmListHost[type][n]);
				fprintf(vmKernelFile, "\n");
			}
		}


		if (benchConf.verbose == 1)
			printf("Updating SpikeList %d\n", threadNumber);

		/**
		 * Updates the generatedSpikeList in GPUs
		 */
		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
			for (int type = 0; type < totalTypes; type++) {

				if (type < startType || endType <= type) {
					cudaMemcpy(synData->spikeTimeListGpusHost[threadDevice][type], synData->spikeTimeListHost[type],    
						   sizeof(ftype) * hList[type][0].spikeTimeListSize * nNeurons[type], cudaMemcpyHostToDevice);
					cudaMemcpy(synData->nGeneratedSpikesGpusHost[threadDevice][type], synData->nGeneratedSpikesHost[type], 
						   sizeof(ucomp) * nNeurons[type], cudaMemcpyHostToDevice);
				}
			}
		}


		for (int type = startType; type < endType; type++) {

			int randomSpikeListSize = sharedData->inputSpikeRate * kernelSteps * dt * nNeurons[type] * 2;
			ftype *randomSpikeTimes = new ftype[ randomSpikeListSize ];
			int *randomSpikeDest = new int[ randomSpikeListSize ];

			// Add some random spikes
			int nRandom = 0;
			for (int neuron = 0; neuron < nNeurons[type]; neuron++) {
				HinesMatrix & m = matrixList[type][neuron];

				if ((kStep + kernelSteps)*m.dt > 9.9999 && sharedData->typeList[type] == PYRAMIDAL_CELL) {
					int32_t spkTime;

					random_r(sharedData->randBuf[threadNumber], &spkTime);
					ftype rate = (sharedData->inputSpikeRate) * (kernelSteps * dt);
					if (spkTime / (float) RAND_MAX < rate ) {
						spkTime = (kStep + kernelSteps)*dt + (ftype)spkTime/RAND_MAX * (kernelSteps * dt);

						if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
							assert(nRandom < randomSpikeListSize);
							randomSpikeTimes[nRandom] = spkTime;
							randomSpikeDest[nRandom] = neuron;
						}
						nRandom++;
						if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM)
							m.synapticChannels->addSpike(0, spkTime, 1);
					}


				}

			}

			int nThreadsComm = 0;
			if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {

				/**
				 * Transfers the list of random spikes to the GPU
				 */
				for (int i=nRandom; i<randomSpikeListSize; i++ ) {
					randomSpikeTimes[i] = -1;
					randomSpikeDest[i] = -1;
				}

				ftype *randomSpikeTimesDev;
				int *randomSpikeDestDev;

				cudaMalloc ( (void **) &randomSpikeTimesDev, sizeof(ftype) * randomSpikeListSize);
				cudaMalloc ( (void **) &randomSpikeDestDev,  sizeof(int) * randomSpikeListSize);

				cudaMemcpy(randomSpikeTimesDev, randomSpikeTimes, sizeof(ftype) * randomSpikeListSize, cudaMemcpyHostToDevice);
				cudaMemcpy(randomSpikeDestDev, randomSpikeDest, sizeof(int) * randomSpikeListSize, cudaMemcpyHostToDevice);

				/**
				 * Performs the Gpu communication
				 */
				nThreadsComm = sharedData->connGpuListHost[type][0].nNeuronsGroup;
				if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {

					uint64 connTmp = 0;
					if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE)
						connTmp = gettimeInMilli();

					performCommunicationsG <<<nGroups[type], nThreadsComm, sharedMemorySize>>> ( nNeurons[type],
						sharedData->connGpuListDevice[type],
						synData->nGeneratedSpikesGpusDev[threadDevice], synData->spikeTimeListGpusDev[threadDevice],
						hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
						synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
						randomSpikeTimesDev, randomSpikeDestDev);

					if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {
						cudaThreadSynchronize();
						bench.connRead += (gettimeInMilli()-connTmp);
					}

				}

				else if (benchConf.gpuCommBenchMode == GPU_COMM_DETAILED) {

					ftype *tmpDevMemory;
					cudaMalloc ( (void **) &tmpDevMemory, sizeof(ftype) * nNeurons[type] * 2 );

					uint64 connTmp = 0;
					if (threadNumber == 0)
						connTmp = gettimeInMilli();
					//int sharedMemComSize = sharedMemorySize/10;
					performCommunicationsG_Step1 <<<nGroups[type], nThreadsComm, sharedMemorySize>>> ( nNeurons[type], sharedData->connGpuListDevice[type],
						synData->nGeneratedSpikesGpusDev[threadDevice], synData->spikeTimeListGpusDev[threadDevice],
						hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
						synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
						randomSpikeTimesDev, randomSpikeDestDev, tmpDevMemory);

					cudaThreadSynchronize();

					if (threadNumber == 0) {
						bench.connRead += (gettimeInMilli()-connTmp);
						connTmp = gettimeInMilli();
					}

					performCommunicationsG_Step2 <<<nGroups[type], nThreadsComm, sharedMemorySize>>> ( nNeurons[type], sharedData->connGpuListDevice[type],
						synData->nGeneratedSpikesGpusDev[threadDevice], synData->spikeTimeListGpusDev[threadDevice],
						hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
						synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
						randomSpikeTimesDev, randomSpikeDestDev, tmpDevMemory);

					cudaThreadSynchronize();

					if (threadNumber == 0)
						bench.connWrite += (gettimeInMilli()-connTmp);

					cudaFree (tmpDevMemory);
				}


				cudaThreadSynchronize();

				cudaFree (randomSpikeTimesDev);
				cudaFree (randomSpikeDestDev);

				cudaThreadSynchronize();
			}


			delete []randomSpikeTimes;
			delete []randomSpikeDest;

			//--------------------------------------------------------------------------------------

			int totalNumberSpikes = 0;
			for (int neuron = 0; neuron < nNeurons[type]; neuron++) {
				HinesMatrix & m = matrixList[type][neuron];

				/**
				 * Updates the spike list when using CPU communications
				 */
				if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM) {
					m.synapticChannels->updateSpikeListGpu(dt*(kStep+kernelSteps),
							synData->spikeListGlobal[type], synData->weightListGlobal[type],
							maxSpikesNeuron, nNeurons[type], neuron, type);
					totalNumberSpikes += m.synapticChannels->spikeListSize;
				}

				// Used to print spike statistics in the end of the simulation
				sharedData->spkStat->addReceivedSpikes(
						type, neuron, m.synapticChannels->getAndResetNumberOfAddedSpikes()-nRandom);

			}

			int synapsePosListTmp = 0;
			int synapseListSize = matrixList[type][0].synapticChannels->synapseListSize;
			int spikeListSizeMax = 0; // The max number of spkikes delievered to a single neuron

			if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == CPU_COMM) {
				for (int n = 0; n < nNeurons[type]; n++) {

					HinesMatrix & m = matrixList[type][n];

					if (m.synapticChannels->spikeListSize > spikeListSizeMax)
						spikeListSizeMax = m.synapticChannels->spikeListSize;

					/**
					 * Stores the number of spikes delivered to each neuron
					 */
					synData->spikeListSizeGlobal[type][n] = m.synapticChannels->spikeListSize;

					/**
					 * Copies the information about the start position of the spikes at each synapse
					 */
					for (int i=0; i<synapseListSize; i++)
						synData->spikeListPosGlobal[type][synapsePosListTmp + i] =
								m.synapticChannels->synSpikeListPos[i];
					synapsePosListTmp += synapseListSize;


					if (spikeListSizeMax > maxSpikesNeuron) {
						printf ("Neuron with %d spikes, more than the max of %d.\n", spikeListSizeMax, maxSpikesNeuron);
						assert(false);
					}
				}
			}


			/**
			 * Tests if the list of spikes, positions and list sizes are the same
			 * for the GPU and CPU communications
			 */
			if (benchConf.gpuCommMode == CHK_COMM ) {
				ftype *spkTmp = (ftype *) malloc (sizeof(ftype)*spikeListSizeMax*nNeurons[type]);
				ftype *weightTmp = (ftype *) malloc (sizeof(ftype)*spikeListSizeMax*nNeurons[type]);
				int *spikeListSizeTmp = (int *) malloc (sizeof(int)*nNeurons[type]);
				int *spikeListPosTmp  = (int *) malloc (sizeof(int)*nNeurons[type]*synapseListSize);

				cudaMemcpy(spkTmp, synData->spikeListDevice[type],
						sizeof(ftype)*spikeListSizeMax*nNeurons[type], cudaMemcpyDeviceToHost);
				cudaMemcpy(weightTmp, synData->weightListDevice[type],
						sizeof(ftype)*spikeListSizeMax*nNeurons[type], cudaMemcpyDeviceToHost);
				cudaMemcpy(spikeListSizeTmp, synData->spikeListSizeDevice[type],
						sizeof(int)*nNeurons[type], cudaMemcpyDeviceToHost);
				cudaMemcpy(spikeListPosTmp, synData->spikeListPosDevice[type],
						sizeof(int)*nNeurons[type]*synapseListSize, cudaMemcpyDeviceToHost);


				for (int neuron=0; neuron<nNeurons[type]; neuron++) {
					if (kStep > 100 && spikeListSizeTmp[neuron] != synData->spikeListSizeGlobal[type][neuron]) {
						printf("SpikeListSIZE time=%f type=%d, neuron=%d, gpu=%d cpu=%d\n",
								dt*(kStep+kernelSteps), type, neuron,
								spikeListSizeTmp[neuron], synData->spikeListSizeGlobal[type][neuron]);
						//assert (false);
					}
				}

				for (int neuronSyn=0; neuronSyn<2*nNeurons[type]; neuronSyn++) {
					if (kStep > 100 && spikeListPosTmp[neuronSyn] != synData->spikeListPosGlobal[type][neuronSyn]) {
						printf("SpikeListPOS time=%f type=%d, neuron=%d, syn=%d, gpu=%d cpu=%d\n",
								dt*(kStep+kernelSteps), type, neuronSyn/2, neuronSyn%2,
								spikeListPosTmp[neuronSyn], synData->spikeListPosGlobal[type][neuronSyn]);
						//assert (false);
					}

				}

				for (int neuronSyn=0; neuronSyn<2*nNeurons[type]; neuronSyn++) {

					int localPos = sharedData->synData->spikeListPosGlobal[type][neuronSyn];

					int nSpikes  = 0;
					if (neuronSyn%2 == 0)
						nSpikes = sharedData->synData->spikeListPosGlobal[type][neuronSyn+1];
					else
						nSpikes = sharedData->synData->spikeListSizeGlobal[type][neuronSyn/2] -
						sharedData->synData->spikeListPosGlobal[type][neuronSyn];

					// threshold = 24ms

					for (int spk = 0; spk < nSpikes; spk++) {

						int globalPos = (localPos + spk) * nNeurons[type] + neuronSyn/2;

						int spk2 = 0;
						for (; spk2 < nSpikes; spk2++) {
							int globalPos2 = (localPos + spk2) * nNeurons[type] + neuronSyn/2;
							if ( (spkTmp[globalPos] == synData->spikeListGlobal[type][globalPos2]) &&
									(weightTmp[globalPos] == synData->weightListGlobal[type][globalPos2]) )
								break;
						}

						if ( spk2 == nSpikes ) {

							printf("time=%f type=%d, neuron=%d, syn=%d, spk=%d, nSpikes=%d, "
									"firstDelievered=%d, nRandom=%d, gpu=%f cpu=%f\n",
									dt*(kStep+kernelSteps), type, neuronSyn/2, neuronSyn%2, spk, nSpikes,
									matrixList[type][neuronSyn/2].synapticChannels->nDelieveredSpikes[neuronSyn%2],
									matrixList[type][neuronSyn/2].synapticChannels->nRandom,
									spkTmp[globalPos], synData->spikeListGlobal[type][globalPos]);

							if (spk > 0) {
								printf("gpuPrevious=%f cpuPrevious=%f\n",
										spkTmp[globalPos-nNeurons[type]], synData->spikeListGlobal[type][globalPos-nNeurons[type]]);

							}
							printf("nNeuronsGroup0=%d, nThreadsConn=%d, nConnections=%d\n",
									sharedData->connGpuListHost[type][0].nNeuronsGroup, nThreadsComm,
									sharedData->connGpuListHost[type][0].nConnectionsTotal);

							printf("GPU\n");
							for (int iSpk = 0; iSpk < nSpikes; iSpk++) {
								int globalPos = (localPos + iSpk) * nNeurons[type] + neuronSyn/2;
								printf("%7.3f ",spkTmp[globalPos]);
							}
							printf("\nCPU\n");
							for (int iSpk = 0; iSpk < nSpikes; iSpk++) {
								int globalPos = (localPos + iSpk) * nNeurons[type] + neuronSyn/2;
								printf("%7.3f ",synData->spikeListGlobal[type][globalPos]);
							}
							printf("\nGPU\n");
							for (int iSpk = 0; iSpk < nSpikes; iSpk++) {
								int globalPos = (localPos + iSpk) * nNeurons[type] + neuronSyn/2;
								printf("%7.3f ",weightTmp[globalPos]);
							}
							printf("\nCPU\n");
							for (int iSpk = 0; iSpk < nSpikes; iSpk++) {
								int globalPos = (localPos + iSpk) * nNeurons[type] + neuronSyn/2;
								printf("%7.3f ",synData->weightListGlobal[type][globalPos]);
							}
							printf("\n");

							//sharedData->connection->getConnArray(type );

							assert(false);
						}
					}
				}

				free(spkTmp);
				free (spikeListSizeTmp);
				free (spikeListPosTmp);
			}

			/**
			 * Copy synaptic and spike info into the GPU
			 */
			if (benchConf.gpuCommMode == CPU_COMM) {

				checkCUDAError("cp2a:");
				cudaMemcpy(synData->spikeListDevice[type], synData->spikeListGlobal[type],
						sizeof(ftype)*spikeListSizeMax*nNeurons[type], cudaMemcpyHostToDevice);
				checkCUDAError("cp2b:");
				cudaMemcpy(synData->weightListDevice[type], synData->weightListGlobal[type],
						sizeof(ftype)*spikeListSizeMax*nNeurons[type], cudaMemcpyHostToDevice);
				checkCUDAError("cp2c:");
				cudaMemcpy(synData->spikeListPosDevice[type], synData->spikeListPosGlobal[type],
						sizeof(int) * (1 + synapseListSize * nNeurons[type]), cudaMemcpyHostToDevice);
				checkCUDAError("cp2d:");
				cudaMemcpy(synData->spikeListSizeDevice[type], synData->spikeListSizeGlobal[type],
						sizeof(int) * (nNeurons[type]+1), cudaMemcpyHostToDevice);
				checkCUDAError("cp2e:");
			}

		}

		if (threadNumber == 0  &&
				(benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE || benchConf.gpuCommMode == CPU_COMM) )
			bench.connWrite = gettimeInMilli();

		if (benchConf.printSampleVms == 1) {
			if (benchConf.verbose == 1)
				printf("Writing Sample Vms thread=%d\n", threadNumber);

			int t1 = 0, n1 = 0;
			if (startType <= t1 && t1 < endType)
				cudaMemcpy(vmTimeSerie[0],  hList[t1][n1].vmTimeSerie, timeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 1;//2291;
			if (startType <= t1 && t1 < endType)
				cudaMemcpy(vmTimeSerie[1],  hList[t1][n1].vmTimeSerie, timeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 2;//135;
			if (startType <= t1 && t1 < endType)
				cudaMemcpy(vmTimeSerie[2],  hList[t1][n1].vmTimeSerie, timeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 3;//1203;
			if (startType <= t1 && t1 < endType)
				cudaMemcpy(vmTimeSerie[3],  hList[t1][n1].vmTimeSerie, timeSerieMemSize, cudaMemcpyDeviceToHost);
			checkCUDAError("Results obtaining:");

			for (int i = kStep; threadNumber == 0 && i < kStep + kernelSteps; i++) {
				fprintf(outFile, "%10.2f\t%10.2f\t%10.2f\t%10.2f\t%10.2f\n", dt * (i+1),
						vmTimeSerie[0][(i-kStep)], vmTimeSerie[1][(i-kStep)], vmTimeSerie[2][(i-kStep)], vmTimeSerie[3][(i-kStep)]);
			}
		}

		if (benchConf.printAllSpikeTimes == 1) {
			if (threadNumber == 0) // Uses only data from SpikeStatistics::addGeneratedSpikes
				sharedData->spkStat->printKernelSpikeStatistics(nSpkfile, lastSpkfile, (kStep+kernelSteps)*dt);
		}

		if (threadNumber == 0 && benchConf.gpuCommMode == CPU_COMM) {
			bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
			bench.totalConnRead	  	+= (bench.connRead 		- bench.kernelFinish)/1000.;
			bench.totalConnWait		+= (bench.connWait 		- bench.connRead)/1000.;
			bench.totalConnWrite	+= (bench.connWrite 	- bench.connWait)/1000.;
		}
		else if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {
			bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
			bench.totalConnWait		+= (bench.connWait 		- bench.kernelFinish)/1000.;
			bench.totalConnRead	  	+=  bench.connRead  / 1000.;
			bench.totalConnWrite	+= (bench.connWrite - bench.connWait - bench.connRead)/1000.;
		}
		else if (threadNumber == 0 && benchConf.gpuCommBenchMode == GPU_COMM_DETAILED) {
			bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
			bench.totalConnWait		+= (bench.connWait 		- bench.kernelFinish)/1000.;
			bench.totalConnRead	  	+=  bench.connRead  / 1000.;
			bench.totalConnWrite	+=  bench.connWrite / 1000.;
		}

	} // (int kStep = 0; kStep < nSteps; kStep += kernelSteps)

	for (int type = startType; type < endType; type++) {
		cudaFree(synData->spikeListDevice[type]);
		cudaFree(synData->weightListDevice[type]);
		free(synData->spikeListGlobal[type]);
		free(synData->weightListGlobal[type]);
	}

	if (threadNumber == 0) {
		bench.execExecution  = gettimeInMilli();
		bench.execExecutionF = (bench.execExecution - bench.execPrepare)/1000.;
	}

	if (threadNumber == 0) {
		printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[0])[nComp0*kernelSteps-1], (vmTimeSerie[0])[kernelSteps-1]);
		printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[1])[nComp0*kernelSteps-1], (vmTimeSerie[1])[kernelSteps-1]);
	}

	// Used to print spike statistics in the end of the simulation
	if (threadNumber == 0)
		sharedData->spkStat->printSpikeStatistics("spikeGpu.dat", totalTime, bench);

	// TODO: Free CUDA Memory
	fclose(outFile);
	free (vmTimeSerie);
	//delete[] hList;
	//delete synData;

	printf("Finished GPU execution.\n" );

	return 0;
}
