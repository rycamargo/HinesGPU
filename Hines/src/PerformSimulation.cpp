#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "PerformSimulation.hpp"
#include "GpuSimulationControl.hpp"
#include "CpuSimulationControl.hpp"

#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "SpikeStatistics.hpp"

#ifdef MPI_GPU_NN
#include <mpi.h>
#endif

#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration

extern void checkCUDAError(const char *msg);

PerformSimulation::PerformSimulation(ThreadInfo *tInfo) {

	this->tInfo      = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void PerformSimulation::createNeurons( ) {

	SharedNeuronGpuData *sharedData = tInfo->sharedData;


    /**------------------------------------------------------------------------------------
	 * Creates the neurons that will be simulated by the threads
	 *-------------------------------------------------------------------------------------*/
    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){
        int nComp = tInfo->nComp[type];
        int nNeurons = tInfo->nNeurons[type];

        sharedData->matrixList[type] = new HinesMatrix[nNeurons];

        for(int n = 0;n < nNeurons;n++){
            HinesMatrix & m = sharedData->matrixList[type][n];
            if(nComp == 1)
                m.defineNeuronCableSquid();

            else
                m.defineNeuronTreeN(nComp, 1);

            m.createTestMatrix();
        }
    }

}

void PerformSimulation::initializeThreadInformation(){

	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	pthread_mutex_lock (sharedData->mutex);
	tInfo->threadNumber = sharedData->nBarrier;
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);

	char *randstate = new char[256];
	sharedData->randBuf[tInfo->threadNumber] = (struct random_data*)calloc(1, sizeof(struct random_data));
	initstate_r(tInfo->sharedData->globalSeed + tInfo->threadNumber + tInfo->currProcess,
			randstate, 256, tInfo->sharedData->randBuf[tInfo->threadNumber]);

	int nThreadsCpu 	= tInfo->sharedData->nThreadsCpu;
    int nTypesPerThread = (tInfo->totalTypes / (nThreadsCpu * tInfo->nProcesses));
    tInfo->startTypeThread = (tInfo->threadNumber + (tInfo->currProcess * nThreadsCpu)) * nTypesPerThread;
    tInfo->endTypeThread = (tInfo->threadNumber + 1 + (tInfo->currProcess * nThreadsCpu)) * nTypesPerThread;
    tInfo->startTypeProcess = tInfo->currProcess * nThreadsCpu * nTypesPerThread;
    tInfo->endTypeProcess = (tInfo->currProcess + 1) * nThreadsCpu * nTypesPerThread;

    int typeProcessCurr = 0;
    tInfo->typeProcess = new int[tInfo->totalTypes];
    for(int type = 0;type < tInfo->totalTypes;type++){
        if(type / ((typeProcessCurr + 1) * nThreadsCpu * nTypesPerThread) == 1)
            typeProcessCurr++;

        tInfo->typeProcess[type] = typeProcessCurr;
    }
}

void PerformSimulation::updateBenchmark()
{
    if(benchConf.simCommMode == NN_CPU) {

				bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
				bench.totalConnRead	  	+= (bench.connRead 		- bench.kernelFinish)/1000.;
				bench.totalConnWait		+= (bench.connWait 		- bench.connRead)/1000.;
				bench.totalConnWrite	+= (bench.connWrite 	- bench.connWait)/1000.;
			}
			else if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE) {
				bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
				bench.totalConnWait		+= (bench.connWait 		- bench.kernelFinish)/1000.;
				bench.totalConnRead	  	+=  bench.connRead  / 1000.;
				bench.totalConnWrite	+= (bench.connWrite - bench.connWait - bench.connRead)/1000.;
			}
			else if (benchConf.gpuCommBenchMode == GPU_COMM_DETAILED) {
				bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
				bench.totalConnWait		+= (bench.connWait 		- bench.kernelFinish)/1000.;
				bench.totalConnRead	  	+=  bench.connRead  / 1000.;
				bench.totalConnWrite	+=  bench.connWrite / 1000.;
			}
}

void PerformSimulation::syncCpuThreads()
{
    pthread_mutex_lock(sharedData->mutex);
    sharedData->nBarrier++;
    if(sharedData->nBarrier < sharedData->nThreadsCpu)
        pthread_cond_wait(sharedData->cond, sharedData->mutex);

    else{
        sharedData->nBarrier = 0;
        pthread_cond_broadcast(sharedData->cond);
    }
    pthread_mutex_unlock(sharedData->mutex);
}

#ifdef MPI_GPU_NN
void PerformSimulation::broadcastGeneratedSpikesMPISync()
{
    /*--------------------------------------------------------------
		 * [MPI] Send the list of generated spikes to other processes
		 *--------------------------------------------------------------*/
    if (tInfo->threadNumber == 0) {
			for (int type=0; type < tInfo->totalTypes; type++) {
				int genSpikeListTypeSize = GENSPIKETIMELIST_SIZE * tInfo->nNeurons[type];
				MPI_Bcast (sharedData->synData->genSpikeTimeListHost[type], genSpikeListTypeSize, MPI_FTYPE, tInfo->typeProcess[type], MPI_COMM_WORLD);
				MPI_Bcast (sharedData->synData->nGeneratedSpikesHost[type], tInfo->nNeurons[type],       MPI_UCOMP, tInfo->typeProcess[type], MPI_COMM_WORLD);
			}
		}
    // Synchronizes the thread to wait for the communication
    syncCpuThreads();
}
#endif

#ifdef MPI_GPU_NN
void PerformSimulation::mpiAllGatherConnections()
{
    // Sends the number of connections from each process
    int *countConnProc = new int[tInfo->nProcesses];
    MPI_Allgather( &(sharedData->connInfo->nConnections), 1, MPI_INT, countConnProc, 1, MPI_INT, MPI_COMM_WORLD);
    // Just for testing
    printf("Connections: %d/%d [", tInfo->currProcess, sharedData->connInfo->nConnections);
    for(int p = 0;p < tInfo->nProcesses;p++)
        printf(" %d |", countConnProc[p]);

    printf("]\n");
    // Allocates the buffer to put information about all the connections
    int countTotalAllProc = 0;
    int *displaceConnProc = new int[tInfo->nProcesses];
    displaceConnProc[0] = 0;
    for(int p = 0;p < tInfo->nProcesses;p++){
        countTotalAllProc += countConnProc[p];
        if(p > 0)
            displaceConnProc[p] = displaceConnProc[p - 1] + countConnProc[p - 1];

    }
    ConnectionInfo *connInfoAllProc = new ConnectionInfo;
    connInfoAllProc->nConnections = countTotalAllProc;
    connInfoAllProc->source = new int[countTotalAllProc];
    connInfoAllProc->dest = new int[countTotalAllProc];
    connInfoAllProc->synapse = new ucomp[countTotalAllProc];
    connInfoAllProc->weigth = new ftype[countTotalAllProc];
    connInfoAllProc->delay = new ftype[countTotalAllProc];
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
    sharedData->connection->clearMPIConnections(sharedData->connInfo);
    sharedData->connInfo = connInfoAllProc;
}
#endif

#ifdef MPI_GPU_NN
void PerformSimulation::prepareMpiGeneratedSpikeStructures() {

	SynapticData *synData = sharedData->synData;

	if (benchConf.simProcMode == NN_CPU) {

		synData->genSpikeTimeListHost     = (ftype **) malloc (sizeof(ftype *)  * tInfo->totalTypes);
		synData->nGeneratedSpikesHost     = (ucomp **) malloc (sizeof(ucomp *)  * tInfo->totalTypes);

		for (int type = 0; type < tInfo->totalTypes; type++) {
			synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * GENSPIKETIMELIST_SIZE * tInfo->nNeurons[type]);
			synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * tInfo->nNeurons[type]);
		}

		for (int type = tInfo->startTypeProcess; type < tInfo->endTypeProcess; type++) {
			for(int n = 0; n < tInfo->nNeurons[type];n++)
				sharedData->matrixList[type][n].redefineGenSpikeTimeList (
						synData->genSpikeTimeListHost[type] + (GENSPIKETIMELIST_SIZE * n));
		}

	}

	/**
	 * The data structures were prepared in the GpuSimulationControl::prepareSynapses method
	 */
	else {
		for (int type = 0; type < tInfo->totalTypes; type++) {
			if ( type < tInfo->startTypeProcess || tInfo->endTypeProcess <= type ) {
				synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * GENSPIKETIMELIST_SIZE * tInfo->nNeurons[type]);
				synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * tInfo->nNeurons[type]);
			}
		}
	}
}
#endif

void PerformSimulation::updateGenSpkStatistics(int *& nNeurons, SynapticData *& synData)
{
    /*--------------------------------------------------------------
		 * Used to print spike statistics in the end of the simulation
		 *--------------------------------------------------------------*/
    if (benchConf.simProcMode == NN_GPU)
			for (int type=tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
				for (int c=0; c<nNeurons[type]; c++)
					sharedData->spkStat->addGeneratedSpikes(type, c, NULL, synData->nGeneratedSpikesHost[type][c]);
		else
			for (int type=tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
				for (int c=0; c<nNeurons[type]; c++)
					sharedData->spkStat->addGeneratedSpikes(type, c, NULL, sharedData->matrixList[type][c].nGeneratedSpikes);
}

void PerformSimulation::generateRandomSpikes( int type, RandomSpikeInfo & randomSpkInfo )
{
	randomSpkInfo.listSize =
			3 * sharedData->inputSpikeRate * kernelInfo->nKernelSteps *
			sharedData->dt * tInfo->nNeurons[type];
	randomSpkInfo.spikeTimes = new ftype[ randomSpkInfo.listSize ];
	randomSpkInfo.spikeDest = new int[ randomSpkInfo.listSize ];

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


/*======================================================================================================
 * Performs the execution
 *======================================================================================================*/
int PerformSimulation::launchExecution() {

	GpuSimulationControl *gpuSimulation = new GpuSimulationControl(tInfo);
	CpuSimulationControl *cpuSimulation = new CpuSimulationControl(tInfo);

	/**
	 * Initializes thread information
	 */
	initializeThreadInformation( );

	/**------------------------------------------------------------------------------------
	 * Creates the neurons that will be simulated by the threads
	 *-------------------------------------------------------------------------------------*/
    createNeurons();

	printf("process = %d | threadNumber = %d | types [%d|%d] | seed=%d \n", tInfo->currProcess, tInfo->threadNumber, tInfo->startTypeThread, tInfo->endTypeThread, tInfo->sharedData->globalSeed);

    int *nNeurons = tInfo->nNeurons;
    int startTypeThread = tInfo->startTypeThread;
    int endTypeThread = tInfo->endTypeThread;
    int threadNumber = tInfo->threadNumber;

    if(threadNumber == 0)
    	gpuSimulation->updateSharedDataInfo();

    //Synchronize threads before starting
    syncCpuThreads();

    bench.matrixSetup = gettimeInMilli();
    bench.matrixSetupF = (bench.matrixSetup - bench.start) / 1000.;

    /*--------------------------------------------------------------
	 * Configure the Device and GPU kernel information
	 *--------------------------------------------------------------*/
    gpuSimulation->configureGpuKernel();

    /*--------------------------------------------------------------
	 * Initializes the benchmark counters
	 *--------------------------------------------------------------*/
    if(threadNumber == 0){
        bench.totalHinesKernel = 0;
        bench.totalConnRead = 0;
        bench.totalConnWait = 0;
        bench.totalConnWrite = 0;
    }
    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for neuron information and transfers the data
	 *--------------------------------------------------------------*/
    if (benchConf.simProcMode == NN_GPU)
    	for(int type = startTypeThread;type < endTypeThread;type++){
    		printf("GPU allocation with %d neurons, %d comparts on device %d thread %d process %d.\n", nNeurons[type], sharedData->matrixList[type][0].nComp, tInfo->deviceNumber, threadNumber, tInfo->currProcess);
    		gpuSimulation->prepareExecution(type);
    	}

    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for the communications and transfers the data
	 *--------------------------------------------------------------*/
    if (benchConf.simProcMode == NN_GPU)
    	gpuSimulation->prepareSynapses();

    SynapticData *synData = sharedData->synData;
    int nKernelSteps = kernelInfo->nKernelSteps;

    /*--------------------------------------------------------------
	 * Sends the complete data to the GPUs
	 *--------------------------------------------------------------*/
    if (benchConf.simProcMode == NN_GPU) {
    	for(int type = startTypeThread;type < endTypeThread;type++){
    		cudaMalloc((void**)((((&(sharedData->hGpu[type]))))), sizeof (HinesStruct) * nNeurons[type]);
    		cudaMemcpy(sharedData->hGpu[type], sharedData->hList[type], sizeof (HinesStruct) * nNeurons[type], cudaMemcpyHostToDevice);
    		checkCUDAError("Memory Allocation:");
    	}
    }

    /*--------------------------------------------------------------
	 * Prepare the spike list in the format used in the GPU
	 *--------------------------------------------------------------*/
    int maxSpikesNeuron = 5000; //5000;
    if (benchConf.simProcMode == NN_GPU) {
    	for(int type = startTypeThread;type < endTypeThread;type++){
    		int neuronSpikeListSize = maxSpikesNeuron * nNeurons[type];
    		synData->spikeListGlobal[type] = (ftype*)((((malloc(sizeof (ftype) * neuronSpikeListSize)))));
    		synData->weightListGlobal[type] = (ftype*)((((malloc(sizeof (ftype) * neuronSpikeListSize)))));
    		cudaMalloc((void**)((((&(synData->spikeListDevice[type]))))), sizeof (ftype) * neuronSpikeListSize);
    		cudaMalloc((void**)((((&(synData->weightListDevice[type]))))), sizeof (ftype) * neuronSpikeListSize);
    		if(type == 0)
    			printf("Spike List size of %.3f MB for each type.\n", sizeof (ftype) * neuronSpikeListSize / 1024. / 1024.);
    	}
    }

    /*--------------------------------------------------------------
	 * Creates the connections between the neurons
	 *--------------------------------------------------------------*/
    if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom (tInfo );

		if (benchConf.simCommMode == NN_GPU) {
			sharedData->connGpuListHost   = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
			sharedData->connGpuListDevice = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
		}

		sharedData->connInfo = sharedData->connection->getConnectionInfo();
	}

    /*--------------------------------------------------------------
	 * [MPI] Send the connection list to other MPI processes
	 *--------------------------------------------------------------*/
#ifdef MPI_GPU_NN
    if(threadNumber == 0)
        mpiAllGatherConnections();
#endif

    /*--------------------------------------------------------------
	 * Guarantees that all connections have been setup
	 *--------------------------------------------------------------*/
    syncCpuThreads();

    /*--------------------------------------------------------------
	 * Creates the connection list for usage in the GPU communication
	 *--------------------------------------------------------------*/
    //if (benchConf.simCommMode == NN_GPU || benchConf.simProcMode == NN_GPU)
    if (benchConf.simCommMode == NN_GPU)
    	gpuSimulation->createGpuCommunicationStructures();

    /*--------------------------------------------------------------
	 * Prepare the lists of generated spikes used for GPU spike delivery
	 *--------------------------------------------------------------*/
    if (benchConf.simCommMode == NN_GPU || benchConf.simProcMode == NN_GPU)
    	gpuSimulation->prepareGpuSpikeDeliveryStructures();

    /*--------------------------------------------------------------
	 * [MPI] Prepare the genSpikeListTime to receive the values from other processes
	 *--------------------------------------------------------------*/
#ifdef MPI_GPU_NN
	if (threadNumber == 0)
		prepareMpiGeneratedSpikeStructures();
#endif

    /*--------------------------------------------------------------
	 * Synchronize threads before beginning [Used only for Benchmarking]
	 *--------------------------------------------------------------*/
    syncCpuThreads();

    printf("Launching GPU kernel with %d blocks and %d (+1) threads per block for types %d-%d for thread %d "
    		"on device %d [%s|%d.%d|MP=%d|G=%dMB|S=%dkB].\n", kernelInfo->nBlocksProc[startTypeThread],
    		nNeurons[startTypeThread] / kernelInfo->nBlocksProc[startTypeThread], startTypeThread, endTypeThread - 1,
    		threadNumber, tInfo->deviceNumber, tInfo->prop->name, tInfo->prop->major, tInfo->prop->minor,
    		tInfo->prop->multiProcessorCount, (int)((tInfo->prop->totalGlobalMem / 1024 / 1024)),
    		(int)((tInfo->prop->sharedMemPerBlock / 1024)));

    if(threadNumber == 0){
        bench.execPrepare = gettimeInMilli();
        bench.execPrepareF = (bench.execPrepare - bench.matrixSetup) / 1000.;
    }

    /*--------------------------------------------------------------
	 * Solves the matrix for n steps
	 *--------------------------------------------------------------*/
    ftype dt = sharedData->dt;
    int nSteps = sharedData->totalTime / dt;

    for (tInfo->kStep = 0; tInfo->kStep < nSteps; tInfo->kStep += nKernelSteps) {

		// Synchronizes the thread to wait for the communication

		if (threadNumber == 0 && tInfo->kStep % 100 == 0)
			printf("Starting Kernel %d -----------> %d \n", threadNumber, tInfo->kStep);

		if (threadNumber == 0) // Benchmarking
			bench.kernelStart  = gettimeInMilli();

		if (benchConf.simProcMode == NN_CPU)
			cpuSimulation->performCpuNeuronalProcessing();
		else
			gpuSimulation->performGpuNeuronalProcessing();


		cudaThreadSynchronize();

		if (threadNumber == 0) // Benchmarking
			bench.kernelFinish = gettimeInMilli();

		/*--------------------------------------------------------------
		 * Reads information from spike sources fromGPU
		 *--------------------------------------------------------------*/
		if (benchConf.simProcMode == NN_GPU)
			gpuSimulation->readGeneratedSpikesFromGPU();

		/*--------------------------------------------------------------
		 * Synchronize threads before communication
		 *--------------------------------------------------------------*/
		syncCpuThreads();

		if (threadNumber == 0 && benchConf.simCommMode == NN_CPU)
			bench.connRead = gettimeInMilli();
		else if (threadNumber == 0) {
			bench.connRead = 0;
			bench.connWrite = 0;
		}

#ifdef MPI_GPU_NN
		broadcastGeneratedSpikesMPISync();
#endif
		if(threadNumber == 0)
			bench.connWait = gettimeInMilli();

		/*--------------------------------------------------------------
		 * Adds the generated spikes to the target synaptic channel
		 * Used only for communication processing in the CPU
		 *--------------------------------------------------------------*/
		if (benchConf.simCommMode == NN_CPU) {
			cpuSimulation->addReceivedSpikesToTargetChannelCPU();
			syncCpuThreads();
		}

		// Used to print spike statistics in the end of the simulation
		updateGenSpkStatistics(nNeurons, synData);

		/*--------------------------------------------------------------
		 * Copy the Vm from GPUs to the CPU memory
		 *--------------------------------------------------------------*/
		if (benchConf.simProcMode == NN_GPU) {
			if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1)
				for (int type = startTypeThread; type < endTypeThread; type++)
					cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);
		}

		// used only for debugging
		//if(threadNumber == 0 && tInfo->kStep == 0) {
		//	ftype usedSharedMem;
		//	cudaMemcpy((void *)(&usedSharedMem), sharedData->hList[startTypeThread][0].active, sizeof(ftype), cudaMemcpyDeviceToHost);
		//	printf("usedSharedMem=%.1f bytes.\n", usedSharedMem);
		//}

		/*--------------------------------------------------------------
		 * Writes Vm to file at the end of each kernel execution
		 *--------------------------------------------------------------*/
		if (benchConf.assertResultsAll == 1)
			if (benchConf.simProcMode == NN_GPU)
				gpuSimulation->checkVmValues();

		/*--------------------------------------------------------------
		 * Check if Vm is ok for all neurons
		 *--------------------------------------------------------------*/
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1)
			sharedData->neuronInfoWriter->writeVmToFile(tInfo->kStep);

		/*--------------------------------------------------------------
		 * Copy the generatedSpikeList to the GPUs
		 *--------------------------------------------------------------*/
		if (benchConf.simCommMode == NN_GPU)
			gpuSimulation->copyGeneratedSpikeListsToGPU();

		/*-------------------------------------------------------
		 * Perform Communications
		 *-------------------------------------------------------*/
		for (int type = startTypeThread; type < endTypeThread; type++) {

			/*-------------------------------------------------------
			 *  Generates random spikes for the network
			 *-------------------------------------------------------*/
			RandomSpikeInfo randomSpkInfo;
			generateRandomSpikes(type, randomSpkInfo);

			/*-------------------------------------------------------
			 * Perform CPU and GPU Communications
			 *-------------------------------------------------------*/
			if (benchConf.simCommMode == NN_GPU)
				gpuSimulation->performGPUCommunications(type, randomSpkInfo);

			else if (benchConf.simCommMode == NN_CPU) {
				cpuSimulation->performCPUCommunication(type, maxSpikesNeuron, randomSpkInfo.nRandom);

				//Copy synaptic and spike info into the GPU
			    if (benchConf.simProcMode == NN_GPU) {
			        int spikeListSizeMax = gpuSimulation->updateSpikeListSizeGlobal(type, maxSpikesNeuron);
			        gpuSimulation->transferSynapticSpikeInfoToGpu(type, spikeListSizeMax);
			    }
			}

			delete []randomSpkInfo.spikeTimes;
			delete []randomSpkInfo.spikeDest;
		}

		if (threadNumber == 0)
			if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE || benchConf.simCommMode == NN_CPU)
				bench.connWrite = gettimeInMilli();

		if (threadNumber == 0 && benchConf.printSampleVms == 1)
			sharedData->neuronInfoWriter->writeSampleVm(tInfo->kStep);

		if (benchConf.printAllSpikeTimes == 1)
			if (threadNumber == 0) // Uses only data from SpikeStatistics::addGeneratedSpikes
				sharedData->spkStat->printKernelSpikeStatistics((tInfo->kStep+nKernelSteps)*dt);

		if (threadNumber == 0)
			updateBenchmark();


	}
    // --------------- Finished the simulation ------------------------------------

	if (threadNumber == 0) {
		bench.execExecution  = gettimeInMilli();
		bench.execExecutionF = (bench.execExecution - bench.execPrepare)/1000.;
	}

	if (threadNumber == 0) {
		//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[0])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[0])[nKernelSteps-1]);
		//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[1])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[1])[nKernelSteps-1]);
	}

	// Used to print spike statistics in the end of the simulation
	if (threadNumber == 0)
		sharedData->spkStat->printSpikeStatistics((const char *)"spikeGpu.dat", sharedData->totalTime, bench, tInfo->startTypeProcess, tInfo->endTypeProcess);

	// TODO: Free CUDA Memory
	if (benchConf.simProcMode == NN_GPU) {
		for (int type = startTypeThread; type < endTypeThread; type++) {
			cudaFree(synData->spikeListDevice[type]);
			cudaFree(synData->weightListDevice[type]);
			free(synData->spikeListGlobal[type]);
			free(synData->weightListGlobal[type]);
		}

		if (threadNumber == 0) {
			delete[] kernelInfo->nBlocksComm;
			delete[] kernelInfo->nThreadsComm;
		}
	}

	if (benchConf.simProcMode == NN_CPU) {
		for (int type = startTypeThread; type < endTypeThread; type++)
			for (int neuron = 0; neuron < nNeurons[type]; neuron++ )
				sharedData->matrixList[type][neuron].freeMem();
	}

	printf("Finished GPU execution.\n" );

	return 0;
}

