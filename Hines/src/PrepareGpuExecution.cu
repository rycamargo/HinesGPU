#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include <cassert>
#include <cstdlib>
#include <pthread.h>

#ifdef MPI_GPU_NN
#include <mpi.h>
#endif

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

extern __global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal);
extern ConnGpu* createGpuConnections( MPIConnectionInfo *connInfoList, int destType, int *nNeurons, int nGroups );
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

void prepareSynapses(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int nThreadsCpu) {

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

		synData->genSpikeTimeListHost     = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->genSpikeTimeListDevice   = (ftype **) malloc (sizeof(ftype *)  * totalTypes);

		synData->nGeneratedSpikesHost     = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);
		synData->nGeneratedSpikesDevice   = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
			synData->genSpikeTimeListGpusDev  = (ftype ***) malloc (sizeof(ftype **) * nThreadsCpu);
			synData->genSpikeTimeListGpusHost = (ftype ***) malloc (sizeof(ftype **) * nThreadsCpu);
			synData->nGeneratedSpikesGpusDev  = (ucomp ***) malloc (sizeof(ucomp **) * nThreadsCpu);
			synData->nGeneratedSpikesGpusHost = (ucomp ***) malloc (sizeof(ucomp **) * nThreadsCpu);
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
		int spikeTimeListSize = GENSPIKETIMELIST_SIZE;

		synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
		cudaMalloc ((void **) &(synData->genSpikeTimeListDevice[type]), sizeof(ftype) * spikeTimeListSize * nNeurons[type]);

		synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->nGeneratedSpikesDevice[type]), sizeof(ucomp) * nNeurons[type]);

		for (int neuron = 0; neuron < nNeurons[type]; neuron++ ) {
			HinesStruct & h = hList[type][neuron];
			h.spikeTimes  = synData->genSpikeTimeListDevice[type] + spikeTimeListSize * neuron;
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
int launchGpuExecution(SharedNeuronGpuData *sharedData, ThreadInfo *tInfo) {

	int *nNeurons = tInfo->nNeurons;
	int startTypeThread   = tInfo->startTypeThread;
	int endTypeThread	  = tInfo->endTypeThread;

	//int totalTypes   = tInfo->totalTypes;
	int threadNumber = tInfo->threadNumber;

	HinesMatrix **matrixList = sharedData->matrixList;
	ftype totalTime = sharedData->totalTime;
	ftype dt = matrixList[startTypeThread][0].dt;
	int nSteps = totalTime / dt;
	int kernelSteps = sharedData->nKernelSteps;

	pthread_mutex_lock (sharedData->mutex);
	if (sharedData->hList == 0) {
		sharedData->hList   = new HinesStruct *[tInfo->totalTypes]; // TODO: To change to totalTypesProcess, need to change all for structures [MPI]
		sharedData->hGpu    = new HinesStruct *[tInfo->totalTypes];
		sharedData->synData = new SynapticData;
		sharedData->synData->spikeListDevice = 0;
		sharedData->synData->totalTypes = tInfo->totalTypes;
	}
	pthread_mutex_unlock (sharedData->mutex);

	HinesStruct **hList	= sharedData->hList;
	HinesStruct **hGpu 	= sharedData->hGpu;


	/*--------------------------------------------------------------
	 * Select the device attributed to each thread
	 *--------------------------------------------------------------*/
	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	cudaSetDevice((threadNumber+tInfo->currProcess+3)%nDevices);

	int deviceNumber=-1;
	cudaGetDevice(&deviceNumber);
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties ( &prop, deviceNumber );
	checkCUDAError("Device selection:");
	//--------------------------------------------------------------

	/*--------------------------------------------------------------
	 * Configure number of threads and shared memory size for each kernel
	 *--------------------------------------------------------------*/
	//int nThreadsProc  	= 64;
	int maxThreadsProc  = 64;
	int maxThreadsComm	= 16;
	int sharedMemSizeProc = 15 * 1024; 	// Valid for capability 1.x (16kB)
	int sharedMemSizeComm = 15 * 1024; 	// Valid for capability 1.x (16kB)

	if (prop.major == 2) {
		//nThreadsProc 	= 256;
		maxThreadsProc 	= 256;
		maxThreadsComm 	= 32; 				// or can be 64
		sharedMemSizeProc = 47 * 1024; 	// Valid for capability 2.x (48kB)
		sharedMemSizeComm = 15 * 1024; 	// Valid for capability 2.x (48kB)
	}
	//--------------------------------------------------------------

	int nBlocksProc[tInfo->totalTypes]; // TODO: To change to totalTypesProcess, need to change all for structures [MPI]
	for (int type = startTypeThread; type < endTypeThread; type++) {
		// Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsProc condition
		nBlocksProc[type] = prop.multiProcessorCount * (nNeurons[type] / maxThreadsProc / prop.multiProcessorCount);
		if (nNeurons[type] % maxThreadsProc != 0  || (nNeurons[type] / maxThreadsProc) % maxThreadsProc != 0)
			nBlocksProc[type] += prop.multiProcessorCount;
	}

	//	if (nComp0 <= 4) nThreads = (sizeof (ftype) == 4) ? 196 : 96;
	//	else if (nComp0 <= 8) nThreads = (sizeof (ftype) == 4) ? 128 : 64;
	//	else if (nComp0 <= 12) nThreads = (sizeof (ftype) == 4) ? 96 : 32;
	//	else if (nComp0 <= 16) nThreads = (sizeof (ftype) == 4) ? 64 : 32;


	FILE *nSpkfile 	  = fopen("nSpikeKernel.dat", "w");
	FILE *lastSpkfile = fopen("lastSpikeKernel.dat", "w");

	/*--------------------------------------------------------------
	 * Initializes the bechmark counters
	 *--------------------------------------------------------------*/
	if (threadNumber == 0) {
		bench.totalHinesKernel = 0;
		bench.totalConnRead    = 0;
		bench.totalConnWait    = 0;
		bench.totalConnWrite   = 0;
	}

	/*--------------------------------------------------------------
	 * Allocates the memory on the GPU for neuron information and transfers the data
	 *--------------------------------------------------------------*/
	for (int type = startTypeThread; type < endTypeThread; type++) {
		prepareExecution(matrixList[type], nNeurons[type], kernelSteps, &(hList[type]), type);
		if (benchConf.verbose)
			printf("GPU allocation with %d neurons, %d comparts on device %d.\n", nNeurons[type], matrixList[type][0].nComp, threadNumber);
	}

	/*--------------------------------------------------------------
	 * Allocates the memory on the GPU for the communications and transfers the data
	 *--------------------------------------------------------------*/
	prepareSynapses(sharedData, nNeurons, startTypeThread, endTypeThread, tInfo->nThreadsCpu);
	SynapticData *synData = sharedData->synData;

	/*--------------------------------------------------------------
	 * Sends the complete data to the GPUs
	 *--------------------------------------------------------------*/
	for (int type = startTypeThread; type < endTypeThread; type++) {
		cudaMalloc((void **)&(hGpu[type]), sizeof(HinesStruct)*nNeurons[type]);
		cudaMemcpy(hGpu[type], hList[type], sizeof(HinesStruct)*nNeurons[type], cudaMemcpyHostToDevice);
		checkCUDAError("Memory Allocation:");
	}

	/*--------------------------------------------------------------
	 * Used to obtain the Vm of neurons and write to files
	 *--------------------------------------------------------------*/
	FILE *outFile;
	outFile = fopen("sampleVm.dat", "w");
	FILE *vmKernelFile;
	vmKernelFile = fopen("vmKernel.dat", "w");

	int nVmTimeSeries = 4;
	ftype **vmTimeSerie  = (ftype **)malloc(sizeof(ftype *) * nVmTimeSeries);
	int nCompVmTimeSerie = matrixList[startTypeThread][0].nComp;
	int vmTimeSerieMemSize = sizeof(ftype) * (nCompVmTimeSerie * kernelSteps);
	for (int k=0; k<nVmTimeSeries; k++)
		vmTimeSerie[k] = (ftype *)malloc(vmTimeSerieMemSize);
	//--------------------------------------------------------------

	/*--------------------------------------------------------------
	 * Prepare the spike list in the format used in the GPU
	 *--------------------------------------------------------------*/
	int maxSpikesNeuron = 5000; //5000;
	for (int type = startTypeThread; type < endTypeThread; type++) {
		int neuronSpikeListSize = maxSpikesNeuron * nNeurons[type];
		synData->spikeListGlobal[type]  = (ftype *) malloc( sizeof(ftype) * neuronSpikeListSize );
		synData->weightListGlobal[type] = (ftype *) malloc( sizeof(ftype) * neuronSpikeListSize );
		cudaMalloc ((void **) &(synData->spikeListDevice[type] ), sizeof(ftype) * neuronSpikeListSize );
		cudaMalloc ((void **) &(synData->weightListDevice[type]), sizeof(ftype) * neuronSpikeListSize );

		if (type == 0) printf ("Spike List size of %.3f MB for each type.\n", sizeof(ftype) * neuronSpikeListSize / 1024. / 1024.);
	}

	/*--------------------------------------------------------------
	 * Creates the connections between the neurons
	 *--------------------------------------------------------------*/
	if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom (sharedData->pyrConnRatio, sharedData->inhConnRatio,
				sharedData->typeList, tInfo->startTypeProcess, tInfo->endTypeProcess, tInfo->totalTypes,
				nNeurons, sharedData, threadNumber);

		if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
			sharedData->connGpuListHost   = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
			sharedData->connGpuListDevice = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
		}

		sharedData->connInfo = sharedData->connection->getMPIConnections();
	}

	/*--------------------------------------------------------------
	 * [MPI] Send the connection list to other processes
	 *--------------------------------------------------------------*/
	if (threadNumber == 0) {
		// Sends the number of connections from each process
		int *countConnProc = new int[tInfo->nProcesses];
		MPI_Allgather( &(sharedData->connInfo->nConnections), 1, MPI_INT, countConnProc, 1, MPI_INT, MPI_COMM_WORLD);

		// Just for testing
		printf("Connections: %d/%d [", tInfo->currProcess, sharedData->connInfo->nConnections);
		for (int p = 0; p < tInfo->nProcesses; p++)
			printf(" %d |", countConnProc[p]);
		printf("]\n");

		// Allocates the buffer to put information about all the connections
		int countTotalAllProc = 0;
		int *displaceConnProc = new int[tInfo->nProcesses];
		displaceConnProc [0] = 0;
		for (int p=0 ; p<tInfo->nProcesses; p++) {
			countTotalAllProc += countConnProc[p];
			if (p > 0)
				displaceConnProc [p] = displaceConnProc [p-1] + countConnProc[p-1];
		}

		MPIConnectionInfo *connInfoAllProc = new MPIConnectionInfo;
		connInfoAllProc->nConnections = countTotalAllProc;
		connInfoAllProc->source  = new int[countTotalAllProc];
		connInfoAllProc->dest 	 = new int[countTotalAllProc];
		connInfoAllProc->synapse = new ucomp[countTotalAllProc];
		connInfoAllProc->weigth  = new ftype[countTotalAllProc];
		connInfoAllProc->delay   = new ftype[countTotalAllProc];

		// Sends the data for each part of the connection list
		// int *source; int *dest; ucomp *synapse; ftype *weigth; ftype *delay;
		MPI_Allgatherv( sharedData->connInfo->source, sharedData->connInfo->nConnections, MPI_INT,
				connInfoAllProc->source, countConnProc, displaceConnProc, MPI_INT, MPI_COMM_WORLD);

		MPI_Allgatherv( sharedData->connInfo->dest, sharedData->connInfo->nConnections, MPI_INT,
				connInfoAllProc->dest, countConnProc, displaceConnProc, MPI_INT, MPI_COMM_WORLD);

		MPI_Allgatherv( sharedData->connInfo->synapse, sharedData->connInfo->nConnections, MPI_UCOMP,
				connInfoAllProc->synapse, countConnProc, displaceConnProc, MPI_UCOMP, MPI_COMM_WORLD);

		MPI_Allgatherv( sharedData->connInfo->weigth, sharedData->connInfo->nConnections, MPI_FTYPE,
				connInfoAllProc->weigth, countConnProc, displaceConnProc, MPI_FTYPE, MPI_COMM_WORLD);

		MPI_Allgatherv( sharedData->connInfo->delay, sharedData->connInfo->nConnections, MPI_FTYPE,
				connInfoAllProc->delay, countConnProc, displaceConnProc, MPI_FTYPE, MPI_COMM_WORLD);


		sharedData->connection->clearMPIConnections( sharedData->connInfo );
		sharedData->connInfo = connInfoAllProc;
	}

	/**
	 * Guarantees that all connections have been setup
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



	/*--------------------------------------------------------------
	 * Creates the connection list for usage in the GPU communication
	 * Should be performed after the connection information from all neurons is available
	 *--------------------------------------------------------------*/
	int nBlocksComm[tInfo->totalTypes]; // TODO: To change to totalTypesProcess, need to change all for structures [MPI]
	if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
		for (int destType = startTypeThread; destType < endTypeThread; destType++) {

			// Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsComm condition
			nBlocksComm[destType] = prop.multiProcessorCount * (nNeurons[destType] / maxThreadsComm / prop.multiProcessorCount);
			if (nNeurons[destType] % maxThreadsComm != 0  || (nNeurons[destType] / maxThreadsComm) % maxThreadsComm != 0)
				nBlocksComm[destType] += prop.multiProcessorCount;

			sharedData->connGpuListHost[destType] = createGpuConnections(
					sharedData->connInfo, destType, nNeurons, nBlocksComm[destType]);
			cudaMalloc( (void **)&(sharedData->connGpuListDevice[destType]), nBlocksComm[destType] * sizeof(ConnGpu) );
			cudaMemcpy(sharedData->connGpuListDevice[destType], sharedData->connGpuListHost[destType], nBlocksComm[destType] * sizeof(ConnGpu), cudaMemcpyHostToDevice);

		}

	}

	//return 0;

	/**
	 * Prepare the lists of generated spikes used for GPU spike delivery
	 * genSpikeTimeListGpusHost[device]
	 */
	if (benchConf.gpuCommMode == CHK_COMM || benchConf.gpuCommMode == GPU_COMM) {
		synData->genSpikeTimeListGpusHost[threadNumber] = (ftype **) malloc( sizeof(ftype *) * tInfo->totalTypes );
		for (int type = 0; type < tInfo->totalTypes; type++) {
			int genSpikeTimeListSize = GENSPIKETIMELIST_SIZE;
			// The device memory for the types of the current thread are already allocated
			if (startTypeThread <= type && type < endTypeThread)
				synData->genSpikeTimeListGpusHost[threadNumber][type] = synData->genSpikeTimeListDevice[type];
			else
				cudaMalloc ((void **) &(synData->genSpikeTimeListGpusHost[threadNumber][type]),
						sizeof(ftype) * genSpikeTimeListSize * nNeurons[type]);
		}

		// Copies the list of pointers to the genSpikeLists of each type
		cudaMalloc( (void **) &synData->genSpikeTimeListGpusDev[threadNumber], sizeof(ftype *) * tInfo->totalTypes);
		cudaMemcpy(synData->genSpikeTimeListGpusDev[threadNumber], synData->genSpikeTimeListGpusHost[threadNumber],
				sizeof(ftype *) * tInfo->totalTypes, cudaMemcpyHostToDevice);

		synData->nGeneratedSpikesGpusHost[threadNumber] = (ucomp **) malloc( sizeof(ucomp *) * tInfo->totalTypes );
		for (int type = 0; type < tInfo->totalTypes; type++) {
			if (startTypeThread <= type && type < endTypeThread)
				synData->nGeneratedSpikesGpusHost[threadNumber][type] = synData->nGeneratedSpikesDevice[type];
			else
				cudaMalloc ((void **) &(synData->nGeneratedSpikesGpusHost[threadNumber][type]), sizeof(ucomp) * nNeurons[type]);
		}

		// Copies the list of pointers to the nGeneratedSpikes of each type
		cudaMalloc( (void **) &synData->nGeneratedSpikesGpusDev[threadNumber], sizeof(ucomp *) * tInfo->totalTypes);
		cudaMemcpy(synData->nGeneratedSpikesGpusDev[threadNumber], synData->nGeneratedSpikesGpusHost[threadNumber],
				sizeof(ucomp *) * tInfo->totalTypes, cudaMemcpyHostToDevice);
	}

	/*--------------------------------------------------------------
	 * [MPI] Prepare the genSpikeListTime to receive the values from other processes
	 *--------------------------------------------------------------*/
	if (threadNumber == 0) {
		for (int type = 0; type < tInfo->totalTypes; type++) {
			if ( type < tInfo->startTypeProcess || tInfo->endTypeProcess <= type ) {
				int spikeTimeListSize = GENSPIKETIMELIST_SIZE;
				synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
				synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
			}
		}
	}

	//return 0;

	/*--------------------------------------------------------------
	 * Synchronize threads before beginning [Used only for Benchmarking]
	 *--------------------------------------------------------------*/
	//pthread_mutex_lock (sharedData->mutex);
	//sharedData->nBarrier++;
	//if (sharedData->nBarrier < sharedData->nThreadsCpu)
	//	pthread_cond_wait(sharedData->cond, sharedData->mutex);
	//else {
	//	sharedData->nBarrier = 0;
	//	pthread_cond_broadcast(sharedData->cond);
	//}
	//pthread_mutex_unlock (sharedData->mutex);


	printf("Launching GPU kernel with %d blocks and %d (+1) threads per block for "
			"types %d-%d for thread %d on device %d [%s|%d.%d|MP=%d|G=%dMB|S=%dkB].\n",
			nBlocksProc[startTypeThread], nNeurons[startTypeThread]/nBlocksProc[startTypeThread],
			startTypeThread, endTypeThread-1, threadNumber, deviceNumber,
			prop.name, prop.major, prop.minor, prop.multiProcessorCount,
			prop.totalGlobalMem/1024/1024, prop.sharedMemPerBlock/1024);

	if (threadNumber == 0) {
		bench.execPrepare  = gettimeInMilli();
		bench.execPrepareF = (bench.execPrepare - bench.matrixSetup)/1000.;
	}

	/*--------------------------------------------------------------
	 * Solves the matrix for n steps
	 *--------------------------------------------------------------*/
	for (int kStep = 0; kStep < nSteps; kStep += kernelSteps) {

		if (threadNumber == 0 && kStep % 100 == 0)
			printf("Starting Kernel %d -----------> %d \n", threadNumber, kStep);

		if (threadNumber == 0) // Benchmarking
			bench.kernelStart  = gettimeInMilli();

		checkCUDAError("Before SolveMatrixG Kernel:");
		for (int type = startTypeThread; type < endTypeThread; type++) {

			int nThreadsProc =  nNeurons[type]/nBlocksProc[type];
			if (nNeurons[type] % nBlocksProc[type]) nThreadsProc++;

			solveMatrixG<<<nBlocksProc[type], nThreadsProc, sharedMemSizeProc>>>(hGpu[type], kernelSteps, nNeurons[type],
					synData->spikeListDevice[type], synData->weightListDevice[type],
					synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
					synData->vmListDevice[type]);
		}

		cudaThreadSynchronize();
		checkCUDAError("Sync:");
		if (threadNumber == 0) // Benchmarking
			bench.kernelFinish = gettimeInMilli();

		if (benchConf.verbose == 1)
			printf("Finished Kernel %d -----------> %d \n", threadNumber, kStep);

		/*--------------------------------------------------------------
		 * - Copy the Vm from GPUs to the CPU memory
		 * - Check if Vm is ok for all neurons
		 *--------------------------------------------------------------*/
		if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1)
			for (int type = startTypeThread; type < endTypeThread; type++)
				cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);
		if (benchConf.assertResultsAll == 1)
			for (int type = startTypeThread; type < endTypeThread; type++)
				for (int n = 0; n < nNeurons[type]; n++)
					if ( synData->vmListHost[type][n] < -500 || 500 < synData->vmListHost[type][n] || synData->vmListHost[type][n] == 0.000000000000) {
						printf("********* type=%d neuron=%d %.2f\n", type, n, synData->vmListHost[type][n]);
						assert(false);
					}
		//--------------------------------------------------------------

		if (benchConf.verbose == 1)
			printf("Getting spikes %d\n", threadNumber);

		/*--------------------------------------------------------------
		 * Reads information from spike sources
		 *--------------------------------------------------------------*/
		for (int type = startTypeThread; type < endTypeThread; type++) {

			HinesStruct & hType = hList[type][0];

			cudaMemcpy(synData->genSpikeTimeListHost[type],    synData->genSpikeTimeListDevice[type],
					sizeof(ftype) * hType.spikeTimeListSize * nNeurons[type], cudaMemcpyDeviceToHost);
			cudaMemcpy(synData->nGeneratedSpikesHost[type], synData->nGeneratedSpikesDevice[type], 
					sizeof(ucomp) * nNeurons[type], cudaMemcpyDeviceToHost);
			checkCUDAError("Synapses2:");
		}
		cudaThreadSynchronize(); // Used to prevent reads of host memory before the copy
		//--------------------------------------------------------------


		for (int type = startTypeThread; type < endTypeThread; type++) {

			HinesStruct & hType = hList[type][0];

			for (int source = 0; source < nNeurons[type]; source++) {

				ucomp nGeneratedSpikes = synData->nGeneratedSpikesHost[type][source];
				if (nGeneratedSpikes > 0) {

					ftype *spikeTimes = synData->genSpikeTimeListHost[type] + hType.spikeTimeListSize * source;

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


		/*--------------------------------------------------------------
		 * Synchronize threads before communication
		 *--------------------------------------------------------------*/
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

		/*--------------------------------------------------------------
		 * [MPI] Send the list of generated spikes to other processes
		 *--------------------------------------------------------------*/
		if (threadNumber == 0) {

			for (int type=0; type < tInfo->totalTypes; type++) {
				int genSpikeListTypeSize = GENSPIKETIMELIST_SIZE * nNeurons[type];
				MPI_Bcast (synData->genSpikeTimeListHost[type], genSpikeListTypeSize, MPI_FTYPE, tInfo->typeProcess[type], MPI_COMM_WORLD);
				MPI_Bcast (synData->nGeneratedSpikesHost[type], nNeurons[type],       MPI_UCOMP, tInfo->typeProcess[type], MPI_COMM_WORLD);
			}
		}

		/*--------------------------------------------------------------
		 * Synchronize threads after communication
		 *--------------------------------------------------------------*/
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
		 * Writes Vm to file at the end of each kernel execution
		 */
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1) {
			for (int type = tInfo->startTypeProcess; type < tInfo->endTypeProcess; type++) {
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
			for (int type = 0; type < tInfo->totalTypes; type++) {

				if (type < startTypeThread || endTypeThread <= type) {
					cudaMemcpy(synData->genSpikeTimeListGpusHost[threadNumber][type], synData->genSpikeTimeListHost[type],
							sizeof(ftype) * GENSPIKETIMELIST_SIZE * nNeurons[type], cudaMemcpyHostToDevice);
					cudaMemcpy(synData->nGeneratedSpikesGpusHost[threadNumber][type], synData->nGeneratedSpikesHost[type],
							sizeof(ucomp) * nNeurons[type], cudaMemcpyHostToDevice);
				}
			}
		}


		/*-------------------------------------------------------
		 *  Generates random spikes for the network
		 *-------------------------------------------------------*/
		for (int type = startTypeThread; type < endTypeThread; type++) {

			int randomSpikeListSize = sharedData->inputSpikeRate * kernelSteps * dt * nNeurons[type] * 3;
			ftype *randomSpikeTimes = new ftype[ randomSpikeListSize ];
			int *randomSpikeDest = new int[ randomSpikeListSize ];

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

					performCommunicationsG <<<nBlocksComm[type], nThreadsComm, sharedMemSizeComm>>> ( nNeurons[type],
							sharedData->connGpuListDevice[type],
							synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
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
					performCommunicationsG_Step1 <<<nBlocksComm[type], nThreadsComm, sharedMemSizeComm>>> ( nNeurons[type], sharedData->connGpuListDevice[type],
							synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
							hGpu[type], synData->spikeListDevice[type], synData->weightListDevice[type],
							synData->spikeListPosDevice[type], synData->spikeListSizeDevice[type],
							randomSpikeTimesDev, randomSpikeDestDev, tmpDevMemory);

					cudaThreadSynchronize();

					if (threadNumber == 0) {
						bench.connRead += (gettimeInMilli()-connTmp);
						connTmp = gettimeInMilli();
					}

					performCommunicationsG_Step2 <<<nBlocksComm[type], nThreadsComm, sharedMemSizeComm>>> ( nNeurons[type], sharedData->connGpuListDevice[type],
							synData->nGeneratedSpikesGpusDev[threadNumber], synData->genSpikeTimeListGpusDev[threadNumber],
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
			if (startTypeThread <= t1 && t1 < endTypeThread)
				cudaMemcpy(vmTimeSerie[0],  hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 1;//2291;
			if (startTypeThread <= t1 && t1 < endTypeThread)
				cudaMemcpy(vmTimeSerie[1],  hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 2;//135;
			if (startTypeThread <= t1 && t1 < endTypeThread)
				cudaMemcpy(vmTimeSerie[2],  hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

			t1 = 0; n1 = 3;//1203;
			if (startTypeThread <= t1 && t1 < endTypeThread)
				cudaMemcpy(vmTimeSerie[3],  hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);
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

	for (int type = startTypeThread; type < endTypeThread; type++) {
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
		printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[0])[nCompVmTimeSerie*kernelSteps-1], (vmTimeSerie[0])[kernelSteps-1]);
		printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[1])[nCompVmTimeSerie*kernelSteps-1], (vmTimeSerie[1])[kernelSteps-1]);
	}

	// Used to print spike statistics in the end of the simulation
	if (threadNumber == 0)
		sharedData->spkStat->printSpikeStatistics("spikeGpu.dat", totalTime, bench, tInfo->startTypeProcess, tInfo->endTypeProcess);

	// TODO: Free CUDA Memory
	fclose(outFile);
	free (vmTimeSerie);
	//delete[] hList;
	//delete synData;

	printf("Finished GPU execution.\n" );

	return 0;
}
