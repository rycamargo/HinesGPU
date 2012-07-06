#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "PerformSimulation.hpp"
#include "GpuSimulationControl.hpp"
#include "CpuSimulationControl.hpp"

#include "SharedNeuronGpuData.hpp"
#include "ThreadInfo.hpp"
#include "KernelInfo.hpp"
#include "SynapticData.hpp"

#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"

//#include "HinesStruct.hpp"
#include "SpikeStatistics.hpp"

#include <cmath>

#ifdef MPI_GPU_NN
#include <mpi.h>
#endif

#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration

extern void checkCUDAError(const char *msg);

PerformSimulation::PerformSimulation(struct ThreadInfo *tInfo) {

	this->tInfo      = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void PerformSimulation::createActivationLists( ) {

	int listSize = sharedData->maxDelay / sharedData->dt;

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
		for (int target = 0; target < tInfo->nNeurons[type]; target++)
			sharedData->matrixList[type][target].synapticChannels->configureSynapticActivationList( sharedData->dt, listSize );
}

void PerformSimulation::createNeurons( ftype dt ) {


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
            m.dt     = dt;
            m.neuron = n;
            m.type   = type;
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
    if(benchConf.checkCommMode(NN_CPU) ) {

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

	if (benchConf.checkProcMode(NN_CPU)) {

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
    if (benchConf.checkProcMode(NN_GPU))
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

	ftype currTime    = sharedData->dt * (tInfo->kStep + kernelInfo->nKernelSteps);
	ftype randWeight  = 1;
	ucomp randSynapse = 0;

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
					int32_t randValue;
					random_r(sharedData->randBuf[tInfo->threadNumber], &randValue);
					ftype rate = (sharedData->inputSpikeRate) * (kernelSteps * dt);
					ftype kPos = (ftype)randValue/RAND_MAX;
					if ( kPos < rate ) {
						ftype spkTime = currTime + (int)( kPos * kernelSteps ) * dt;

						// New implementation
						if (type ==0 && neuron == 0)
						    printf("randomSpike at %.2f.\n", spkTime);

						if (benchConf.checkCommMode(NN_GPU) ) {
							assert(randomSpkInfo.nRandom < randomSpkInfo.listSize);
							randomSpkInfo.spikeTimes[randomSpkInfo.nRandom] = spkTime;
							randomSpkInfo.spikeDest[randomSpkInfo.nRandom] = neuron;
						}
						randomSpkInfo.nRandom++;
						if (benchConf.checkCommMode(NN_CPU) ) {

							if (benchConf.checkProcMode(NN_CPU) )
								m.synapticChannels->addToSynapticActivationList(currTime, sharedData->dt, randSynapse, spkTime, 0, randWeight);

							else if (benchConf.checkProcMode(NN_GPU) )
								GpuSimulationControl::addToInterleavedSynapticActivationList(
										sharedData->synData->activationListGlobal[type],
										sharedData->synData->activationListPosGlobal[type] + neuron * m.synapticChannels->synapseListSize,
										m.synapticChannels->activationListSize,
										neuron, tInfo->nNeurons[type], currTime, sharedData->dt, randSynapse, spkTime, 0, randWeight);

						}

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
    sharedData->dt = 0.1; // 0.1ms
    sharedData->minDelay = 10; // 10ms
    sharedData->maxDelay = 20; // 10ms
	kernelInfo->nKernelSteps = sharedData->minDelay / sharedData->dt;

    createNeurons(sharedData->dt);

	printf("process = %d | threadNumber = %d | types [%d|%d] | seed=%d \n",
			tInfo->currProcess, tInfo->threadNumber, tInfo->startTypeThread, tInfo->endTypeThread, tInfo->sharedData->globalSeed);

    int *nNeurons = tInfo->nNeurons;
    int startTypeThread = tInfo->startTypeThread;
    int endTypeThread = tInfo->endTypeThread;
    int threadNumber = tInfo->threadNumber;

    if(threadNumber == 0)
    	gpuSimulation->updateSharedDataInfo();

    /*--------------------------------------------------------------
	 * Creates the connections between the neurons
	 *--------------------------------------------------------------*/
    if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom (tInfo );

		if (benchConf.checkCommMode(NN_GPU) ) {
			sharedData->connGpuListHost   = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
			sharedData->connGpuListDevice = (ConnGpu **)malloc(tInfo->totalTypes * sizeof(ConnGpu *));
		}

		sharedData->connInfo = sharedData->connection->getConnectionInfo();
	}

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

	//if (benchConf.checkCommMode() == NN_CPU) TODO: should be created only for CPU execution?
	createActivationLists();

    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for neuron information and transfers the data
	 *--------------------------------------------------------------*/
    if (benchConf.checkProcMode(NN_GPU))
    	for(int type = startTypeThread;type < endTypeThread;type++){
    		//printf("GPU allocation with %d neurons, %d comparts on device %d thread %d process %d.\n", nNeurons[type], sharedData->matrixList[type][0].nComp, tInfo->deviceNumber, threadNumber, tInfo->currProcess);
    		gpuSimulation->prepareExecution(type);
    	}

    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for the communications and transfers the data
	 *--------------------------------------------------------------*/
    if (benchConf.checkProcMode(NN_GPU))
    	gpuSimulation->prepareSynapses();

    SynapticData *synData = sharedData->synData;
    int nKernelSteps = kernelInfo->nKernelSteps;

    /*--------------------------------------------------------------
	 * Sends the complete data to the GPUs
	 *--------------------------------------------------------------*/
    if (benchConf.checkProcMode(NN_GPU)) {
    	gpuSimulation->transferHinesStructToGpu();
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
    //if (benchConf.checkCommMode() == NN_GPU || benchConf.checkProcMode() == NN_GPU)
    if (benchConf.checkCommMode(NN_GPU) )
    	gpuSimulation->createGpuCommunicationStructures();

    /*--------------------------------------------------------------
	 * Prepare the lists of generated spikes used for GPU spike delivery
	 *--------------------------------------------------------------*/
    if (benchConf.checkCommMode(NN_GPU)  || benchConf.checkProcMode(NN_GPU) )
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

    printf("Launching GPU kernel with %d blocks and %d (+1) threads per block for types %d-%d for thread %d process %d "
    		"on device %d [%s|%d.%d|MP=%d|G=%dMB|S=%dkB].\n", kernelInfo->nBlocksProc[startTypeThread],
    		nNeurons[startTypeThread] / kernelInfo->nBlocksProc[startTypeThread], startTypeThread, endTypeThread - 1,
    		threadNumber, tInfo->currProcess, tInfo->deviceNumber, tInfo->prop->name, tInfo->prop->major, tInfo->prop->minor,
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

		if (benchConf.checkProcMode(NN_CPU))
			cpuSimulation->performCpuNeuronalProcessing();
		else
			gpuSimulation->performGpuNeuronalProcessing();


		cudaThreadSynchronize();

		if (threadNumber == 0) // Benchmarking
			bench.kernelFinish = gettimeInMilli();

		/*--------------------------------------------------------------
		 * Reads information from spike sources fromGPU
		 *--------------------------------------------------------------*/
		if (benchConf.checkProcMode(NN_GPU))
			gpuSimulation->readGeneratedSpikesFromGPU();

		/*--------------------------------------------------------------
		 * Synchronize threads before communication
		 *--------------------------------------------------------------*/
		syncCpuThreads();

		if (threadNumber == 0 && benchConf.checkCommMode(NN_CPU) )
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
		if (benchConf.checkCommMode(NN_CPU) ) { //  &&  benchConf.checkProcMode(NN_GPU) != NN_TEST

			if ( benchConf.checkProcMode(NN_GPU) ) {
				gpuSimulation->copyActivationListFromGpu();
				syncCpuThreads();
			}

			cpuSimulation->addReceivedSpikesToTargetChannelCPU();
			syncCpuThreads();
		}

		// Used to print spike statistics in the end of the simulation
		updateGenSpkStatistics(nNeurons, synData);

		/*--------------------------------------------------------------
		 * Copy the Vm from GPUs to the CPU memory
		 *--------------------------------------------------------------*/
		if (benchConf.checkProcMode(NN_GPU) ) {
			if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1)
				for (int type = startTypeThread; type < endTypeThread; type++)
					cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);
		}

		/*--------------------------------------------------------------
		 * Writes Vm to file at the end of each kernel execution
		 *--------------------------------------------------------------*/
		if (benchConf.assertResultsAll == 1)
			if (benchConf.checkProcMode(NN_GPU) )
				gpuSimulation->checkVmValues();

		/*--------------------------------------------------------------
		 * Check if Vm is ok for all neurons
		 *--------------------------------------------------------------*/
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1)
			sharedData->neuronInfoWriter->writeVmToFile(tInfo->kStep);

		/*--------------------------------------------------------------
		 * Copy the generatedSpikeList to the GPUs
		 *--------------------------------------------------------------*/
		if (benchConf.checkCommMode(NN_GPU) )
			gpuSimulation->copyGeneratedSpikeListsToGPU();

		/*-------------------------------------------------------
		 * Perform Communications
		 *-------------------------------------------------------*/
		for (int type = startTypeThread; type < endTypeThread; type++) {

			/*-------------------------------------------------------
			 *  Generates random spikes for the network
			 *-------------------------------------------------------*/
			struct RandomSpikeInfo randomSpkInfo;
			generateRandomSpikes(type, randomSpkInfo);

			/*-------------------------------------------------------
			 * Perform CPU and GPU Communications
			 *-------------------------------------------------------*/
			if (benchConf.checkProcMode(NN_GPU) ) {

				if ( benchConf.checkCommMode(NN_GPU) )
					gpuSimulation->performGPUCommunications(type, randomSpkInfo);

				else if ( benchConf.checkCommMode(NN_CPU) == NN_CPU )  { // Hybrid mode
					gpuSimulation->copyActivationListToGpu(type);
				}

				if ( benchConf.checkCommMode(NN_TEST) == NN_TEST && tInfo->kStep > 101)  { // Test mode
					gpuSimulation->testGpuCommunication(type);

				}
			}

			delete []randomSpkInfo.spikeTimes;
			delete []randomSpkInfo.spikeDest;
		}

		if (threadNumber == 0)
			if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE || benchConf.checkCommMode(NN_CPU) )
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
    if (benchConf.checkProcMode(NN_GPU) ) {
    	if (threadNumber == 0) {
    		delete[] kernelInfo->nBlocksComm;
    		delete[] kernelInfo->nThreadsComm;
    	}
    }

    if (benchConf.checkProcMode(NN_CPU) ) {
    	for (int type = startTypeThread; type < endTypeThread; type++)
    		for (int neuron = 0; neuron < nNeurons[type]; neuron++ )
    			sharedData->matrixList[type][neuron].freeMem();
    }

    printf("Finished GPU execution.\n" );

    return 0;
}

