/*
 * launchHines.cpp
 *
 *  Created on: 02/06/2009
 *      Author: Raphael Y. de Camargo
 *      Affiliation: Universidade Federal do ABC (UFABC), Brazil
 */

/**
 * - Create matrix for linear Cable [Ter] OK
 * - Solve matrix with Gaussian elimination [Ter] OK
 * - Print Vm in an output file (input for Gnuplot) [Ter] OK
 * - Add current injection and leakage [Qua] OK
 * - Check constants and compare with GENESIS [Qua] OK
 * - Add soma [Qua] OK
 * - Add branched Tree [Qua] OK
 * - Support for multiples neurons at the same time OK
 * - Allocate all the memory in sequence OK
 * - Simulation in the GPU as a series of steps OK
 * - Larger neurons OK
 *
 * - Active Channels on CPUs [Ter-16/06] OK
 * - Active Channels on GPUs [Qui-18/06] OK
 *
 * - Usage of shared tables for active currents (For GPUs may be not useful) (1) [Seg-23/06]
 * - Optimize performance for larger neurons (GPU shared memory) (2) [Seg-23/06]
 * - Optimize performance for larger neurons (Memory Coalescing) (3) [Ter-24/06]
 * 		- Each thread reads data for multiple neurons
 *
 * - Optimizations (Useful only for large neurons)
 *   - mulList and leftMatrix can be represented as lists (use macros)
 *     - Sharing of matrix values among neurons (CPU {cloneNeuron()} e GPU)
 *     - mulList e tmpVmList can share the same matrix (better to use lists)
 *
 * - Support for communication between neurons [Future Work]
 * - Support for multiple threads for each neuron (from different blocks, y axis) [Future Work]
 */

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <unistd.h>
#include <cmath>

#include "SharedNeuronGpuData.hpp"
#include "KernelInfo.hpp"
#include "ThreadInfo.hpp"
#include "PerformSimulation.hpp"
#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "KernelProfiler.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"
//#include "HinesStruct.hpp"
#include "SpikeStatistics.hpp"
//#include "GpuSimulationControl.hpp"

#ifdef MPI_GPU_NN
#include <mpi.h>
#endif

using namespace std;

// Defined in HinesGpu.cu
//extern "C" {
//extern int launchGpuExecution(SharedNeuronGpuData *sharedData, ThreadInfo *tInfo);
//int launchGpuExecution(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int totalTypes, int threadNumber){return 0;}
//}
//int launchGpuExecution(HinesMatrix *matrixList, int nNeurons){}


//extern int performHostExecution(ThreadInfo *tInfo);


void *launchHostExecution(void *ptr) {

	ThreadInfo *tInfo = (ThreadInfo *)ptr;

	/**
	 * Launches the execution of all threads
	 */
	PerformSimulation *simulation = new PerformSimulation(tInfo);
	simulation->launchExecution();
	//simulation->performHostExecution();

	return 0;
}


void *launchDeviceExecution(void *threadInfo) {

	ThreadInfo *tInfo = (ThreadInfo *)threadInfo;

	/**
	 * Launches the execution of all threads
	 */
	PerformSimulation *simulation = new PerformSimulation(tInfo);
	simulation->launchExecution();

	return 0;
}


ThreadInfo *createInfoArray(int nThreads, ThreadInfo *model){
	ThreadInfo *tInfoArray = new ThreadInfo[nThreads];
	for (int i=0; i<nThreads; i++) {
		tInfoArray[i].sharedData 	= model->sharedData;
		tInfoArray[i].nComp			= model->nComp;
		tInfoArray[i].nNeurons		= model->nNeurons;
		tInfoArray[i].nNeuronsTotalType = model->nNeuronsTotalType;

		tInfoArray[i].nTypes		= model->nTypes;
		tInfoArray[i].totalTypes	= model->totalTypes;

		tInfoArray[i].currProcess	= model->currProcess;

		tInfoArray[i].nProcesses	= model->nProcesses;

		tInfoArray[i].startTypeThread	= model->startTypeThread;
		tInfoArray[i].endTypeThread		= model->endTypeThread;
		tInfoArray[i].threadNumber		= model->threadNumber;

		tInfoArray[i].globalThreadTypes     = model->globalThreadTypes;
		tInfoArray[i].globalThreadTypesSize = model->globalThreadTypesSize;

	}

	return tInfoArray;
}

void configureNeuronTypes(char*& simType, ThreadInfo*& tInfo, int nNeuronsTotal,  char *configFileName) {

	if (simType[0] != 'c') {
		int nComp = 4;

		tInfo->nTypes = 3;
		tInfo->totalTypes = tInfo->nTypes * tInfo->sharedData->nThreadsCpu * tInfo->nProcesses;

		tInfo->nNeurons = new int[tInfo->totalTypes];
		tInfo->nComp = new int[tInfo->totalTypes];
		tInfo->sharedData->typeList = new int[tInfo->totalTypes];
		tInfo->nNeuronsTotalType = new int[tInfo->nTypes];
		for (int type=0; type < tInfo->nTypes; type++)
			tInfo->nNeuronsTotalType[ type ] = 0;

		tInfo->sharedData->matrixList = new HinesMatrix *[tInfo->totalTypes];
		for (int i = 0; i < tInfo->totalTypes; i += tInfo->nTypes) {
			tInfo->nNeurons[i] = nNeuronsTotal / (tInfo->totalTypes);
			tInfo->nComp[i] = nComp;
			tInfo->sharedData->typeList[i] = PYRAMIDAL_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

			tInfo->nNeurons[i + 1] = nNeuronsTotal / (tInfo->totalTypes);
			tInfo->nComp[i + 1] = nComp;
			tInfo->sharedData->typeList[i + 1] = INHIBITORY_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i+1] ] += tInfo->nNeurons[i+1];

			tInfo->nNeurons[i + 2] = nNeuronsTotal / (tInfo->totalTypes);
			tInfo->nComp[i + 2] = nComp;
			tInfo->sharedData->typeList[i + 2] = BASKET_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i+2] ] += tInfo->nNeurons[i+2];

		}
	}
	else {

		int nComp = 4;

		char str[200];
		char *strTmp = str;

		FILE *fp = fopen(configFileName,"r");
		if(!fp) { // exits if file not found
			printf("launchHines.cpp: Network configuration file does not exist.\n");
			exit(-1);
		}



		tInfo->totalTypes = 0;

		int maxPr = 20;
		int maxTh = 10;
		int maxTy = 10;

		int nProcessesGlobal = 0;
		int nThreadsTotal = 0;
		int nThreadsGlobal[maxPr];


		int startTypeProcessGlobal[maxPr];
		int   endTypeProcessGlobal[maxPr];
		int iProcess = -1;

		int startTypeThreadGlobal[maxPr * maxTh];
		int   endTypeThreadGlobal[maxPr * maxTh];
		int iThread = -1;

		int nNeuronsTypeTotal[maxTy];
		int nNeuronsGlobal[maxPr * maxTh];
		int typeListGlobal[maxPr * maxTh];

		int hostFound = 0;

		for (int t=0; t<maxTy; t++)
			nNeuronsTypeTotal[t] = nNeuronsTotal;

		while(fgets(str,sizeof(str),fp) != NULL)
		{
			int len = strlen(str)-1; // strips '\n'
			if (str[len] == '\n') str[len] = 0;
			if (len == 0 || str[0] == '#' ) continue; // skip comment lines

			if (str[0] == 't' ) {
				strTmp = strchr(str,':') + 1; // removes leading identifier
				sscanf(strTmp, "%d", &(tInfo->nTypes));
				assert(tInfo->nTypes <= 10);

				for (int t=0; t < tInfo->nTypes; t++) {
					strTmp = strchr(strTmp,':') + 1;
					ftype rateTmp;
					sscanf(strTmp, "%f", &rateTmp);
					nNeuronsTypeTotal[t] *= rateTmp;
				}
			}

			if (str[0] == 'm' ) {
				iProcess++;	iThread=-1;

				strTmp = strchr(str,':') + 1; // removes leading identifier

				// Get the machine name
				char hostconfig[50];
				sscanf(strTmp, "%s", hostconfig);
				strchr(hostconfig,':')[0] = 0;
			    char hostname[50];
			    gethostname(hostname, 50);

			    printf("%s %s\n", hostconfig, hostname);

				strTmp = strchr(strTmp,':') + 1;
				int nThreadsTmp;
				sscanf(strTmp, "%d", &nThreadsTmp);

				nThreadsGlobal[iProcess] = nThreadsTmp;
				nProcessesGlobal++;
				nThreadsTotal += nThreadsTmp;

				startTypeProcessGlobal[iProcess] = tInfo->totalTypes;
				endTypeProcessGlobal[iProcess]   = tInfo->totalTypes;

			    if (strcmp(hostconfig, hostname) == 0) {
			    	tInfo->sharedData->nThreadsCpu = nThreadsTmp;
			    	hostFound++;
			    }

			}

			if (str[0] == 'n' ) {
				iThread++;
				strTmp = str;

				startTypeThreadGlobal[iProcess * maxTh + iThread] = tInfo->totalTypes;
				ftype rateNeuronsType = 0;
				for (int t=0; t < tInfo->nTypes; t++) {
					strTmp = strchr(strTmp,':') + 1;
					sscanf(strTmp, "%f", &rateNeuronsType);

					if (rateNeuronsType > 0) {
						nNeuronsGlobal[ tInfo->totalTypes ] = roundf(nNeuronsTypeTotal[t] * rateNeuronsType);
						typeListGlobal[ tInfo->totalTypes ] = t;
						tInfo->totalTypes++;
					}
				}
				endTypeThreadGlobal[iProcess * maxTh + iThread] = tInfo->totalTypes;
				endTypeProcessGlobal[iProcess] = tInfo->totalTypes;
			}

		}

		for (int t=0; t<tInfo->nTypes; t++)
			assert (nNeuronsTypeTotal[t] <= nNeuronsTotal);

		assert (tInfo->nProcesses == nProcessesGlobal);

		tInfo->globalThreadTypesSize = 2 * tInfo->nProcesses + 2 * nThreadsTotal;
		tInfo->globalThreadTypes = new int[tInfo->globalThreadTypesSize];

		int offset = 0;
		int t = 0;
		for (t=0; t < tInfo->nProcesses; t++)
			tInfo->globalThreadTypes[offset + t] = startTypeProcessGlobal[t];
		offset += tInfo->nProcesses;
		for (t=0; t < tInfo->nProcesses; t++)
			tInfo->globalThreadTypes[offset + t] = endTypeProcessGlobal[t];
		offset += tInfo->nProcesses;
		for (int p=0, t=0; p < tInfo->nProcesses; p++)
			for (int pt=0; pt < nThreadsGlobal[p]; pt++, t++)
				tInfo->globalThreadTypes[offset + t] = startTypeThreadGlobal[p * maxTh + pt];
		offset += nThreadsTotal;
		for (int p=0, t=0; p < tInfo->nProcesses; p++)
			for (int pt=0; pt < nThreadsGlobal[p]; pt++, t++)
				tInfo->globalThreadTypes[offset + t] = endTypeThreadGlobal[p * maxTh + pt];

		assert(hostFound == 1);
		for (t=0; t < tInfo->nProcesses; t++)
			assert(startTypeProcessGlobal[t] != endTypeProcessGlobal[t]);


		tInfo->sharedData->matrixList = new HinesMatrix *[tInfo->totalTypes];
		tInfo->nNeurons = new int[tInfo->totalTypes];
		tInfo->nComp = new int[tInfo->totalTypes];
		tInfo->sharedData->typeList = new int[tInfo->totalTypes];
		for (int t=0; t < tInfo->totalTypes; t++) {
			tInfo->nNeurons[t] = nNeuronsGlobal[t];
			tInfo->nComp[t] = nComp;
			tInfo->sharedData->typeList[t] = typeListGlobal[t];
		}

		tInfo->nNeuronsTotalType = new int[tInfo->nTypes];
		for (int type=0; type < tInfo->nTypes; type++)
			tInfo->nNeuronsTotalType[ type ] = 0;
		for (int type=0; type < tInfo->totalTypes; type++)
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[type] ] += tInfo->nNeurons[type];

		//*****************************************************
		//printf("nTypes=%d totalTypes=%d nProcesses=%d\n", tInfo->nTypes, tInfo->totalTypes, tInfo->nProcesses);
		//for (int t=0; t < tInfo->totalTypes; t++)
		//	printf( "%d|%d ", nNeuronsGlobal[t], typeListGlobal[t]);
		//printf( "\n");
		//for (int t=0; t < tInfo->globalThreadTypesSize; t++)
		//	printf( "%d ", tInfo->globalThreadTypes[t]);
		//printf( "\n");
		//*****************************************************

		// Check if the total number of neurons is correct
		for (int gType=0; gType < tInfo->nTypes; gType++) {

			// Corrects the distribution of neurons caused by rounding errors
			if (tInfo->nNeuronsTotalType[gType] != nNeuronsTypeTotal[gType]) {

				printf("CORRECTING ERROR: Number of neurons do not match: nNeuronsType=%d|%d\n", tInfo->nNeuronsTotalType[gType], nNeuronsTypeTotal[gType]);

				int nTypeInstances = 0;
				for (int tmp=0; tmp < tInfo->totalTypes; tmp++)
					if (tInfo->sharedData->typeList[tmp] == gType)
						nTypeInstances++;

				int diff = nNeuronsTypeTotal[gType] - tInfo->nNeuronsTotalType[gType];
				for (int tmp=0; tmp < tInfo->totalTypes; tmp++)
					if (tInfo->sharedData->typeList[tmp] == gType)
						tInfo->nNeurons[tmp] += diff / nTypeInstances;


				tInfo->nNeuronsTotalType[ gType ] = 0;
				for (int tmp=0; tmp < tInfo->totalTypes; tmp++)
					if (tInfo->sharedData->typeList[tmp] == gType)
						tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[tmp] ] += tInfo->nNeurons[tmp];

			}

			if (tInfo->nNeuronsTotalType[gType] != nNeuronsTypeTotal[gType]) {
				printf("ERROR2: Number of neurons do not match: nNeuronsType=%d|%d\n", tInfo->nNeuronsTotalType[gType], nNeuronsTypeTotal[gType]);
				assert(tInfo->nNeuronsTotalType[gType] == nNeuronsTypeTotal[gType]);
			}

		}

		fclose(fp);

	}
}

void configureSimulation(char *simType, ThreadInfo *& tInfo, int nNeurons, char mode, char *configFile)
{

	// Configure the types and number of neurons
	configureNeuronTypes(simType, tInfo, nNeurons, configFile);

	// defines some default values
	tInfo->sharedData->inputSpikeRate = 0.1;
	tInfo->sharedData->pyrPyrConnRatio   = 0.1;
	tInfo->sharedData->pyrInhConnRatio   = 0.1;
	tInfo->sharedData->totalTime   = 100; // in ms

	tInfo->sharedData->randWeight = 1;

	tInfo->sharedData->profileKernel = 0;

    if (simType[0] == 'n' || simType[0] == 'd') {
			printf ("Simulation configured as: Running scalability experiments.\n");

			benchConf.printSampleVms = 1;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;
			benchConf.checkGpuComm = 0;

			if (mode=='G')      benchConf.setMode(NN_GPU, NN_GPU);
			else if (mode=='H') benchConf.setMode(NN_GPU, NN_CPU);
			else if (mode=='C') benchConf.setMode(NN_CPU, NN_CPU);
			else if (mode=='T') benchConf.setMode(NN_GPU, NN_TEST);

			if (simType[0] == 'n') benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;
			else if (simType[0] == 'd') benchConf.gpuCommBenchMode = GPU_COMM_DETAILED;

			tInfo->sharedData->totalTime   = 1000;
			tInfo->sharedData->inputSpikeRate = 0.01;
			tInfo->sharedData->connectivityType = CONNECT_RANDOM_1;

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '0') { // 200k: 0.79
				tInfo->sharedData->pyrPyrConnRatio   = 0; // nPyramidal
				tInfo->sharedData->pyrInhConnRatio   = 0; // nPyramidal
				tInfo->sharedData->inputSpikeRate = 0.02; // increases the input
			}
			else if (simType[1] == '1') {
				tInfo->sharedData->pyrPyrConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal //100
				tInfo->sharedData->pyrInhConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal //100

				if (simType[2] == 'l') { // 200k: 0.92
					tInfo->sharedData->excWeight    = 0.030;
					tInfo->sharedData->pyrInhWeight = 0.035;
					tInfo->sharedData->inhPyrWeight = 10;
				}
				if (simType[2] == 'm') { // 200k: 1.93
					tInfo->sharedData->excWeight    = 0.045;
					tInfo->sharedData->pyrInhWeight = 0.020;
					tInfo->sharedData->inhPyrWeight = 4;
				}
				if (simType[2] == 'h') { // 200k: 5.00
					tInfo->sharedData->excWeight    = 0.100;
					tInfo->sharedData->pyrInhWeight = 0.030;
					tInfo->sharedData->inhPyrWeight = 1;
				}
			}
			else if (simType[1] == '2') {
				tInfo->sharedData->pyrPyrConnRatio   = 1000.0 / (nNeurons/tInfo->nTypes); // nPyramidal
				tInfo->sharedData->pyrInhConnRatio   = 1000.0 / (nNeurons/tInfo->nTypes); // nPyramidal

				if (simType[2] == 'l') { // 100k: 1.19
					tInfo->sharedData->excWeight    = 0.004;
					tInfo->sharedData->pyrInhWeight = 0.004;
					tInfo->sharedData->inhPyrWeight = 10;
				}
				if (simType[2] == 'm') { // 10k: 2.04
					tInfo->sharedData->excWeight    = 0.005;
					tInfo->sharedData->pyrInhWeight = 0.003;
					tInfo->sharedData->inhPyrWeight = 4;
				}
				if (simType[2] == 'h') { // 10k: 5.26
					tInfo->sharedData->excWeight    = 0.008;
					tInfo->sharedData->pyrInhWeight = 0.004;
					tInfo->sharedData->inhPyrWeight = 1;
				}

			}
		}


		else if (simType[0] == 'c') {
			printf ("Simulation configured as: Running cluster experiments.\n");

			tInfo->sharedData->profileKernel = 1;
			if (tInfo->sharedData->profileKernel)
				printf ("WARNING: Profiling the kernel execution times!\n");


			benchConf.printSampleVms = 1;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;
			benchConf.checkGpuComm = 0;

			if (mode=='G')      benchConf.setMode(NN_GPU, NN_GPU);
			else if (mode=='H') benchConf.setMode(NN_GPU, NN_CPU);
			else if (mode=='C') benchConf.setMode(NN_CPU, NN_CPU);
			else if (mode=='T') benchConf.setMode(NN_GPU, NN_TEST);

			benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;

			tInfo->sharedData->totalTime   = 10000; // in ms
			tInfo->sharedData->inputSpikeRate = 0.01;
			tInfo->sharedData->connectivityType = CONNECT_RANDOM_2;

			tInfo->sharedData->excWeight = 0.01;
			tInfo->sharedData->pyrInhWeight = 0.1;
			tInfo->sharedData->inhPyrWeight = 1;


			if (simType[1] == '1') {
				tInfo->sharedData->pyrPyrConnRatio   =  90.0 / tInfo->nNeuronsTotalType[PYRAMIDAL_CELL];
				tInfo->sharedData->pyrInhConnRatio   =  10.0 / tInfo->nNeuronsTotalType[INHIBITORY_CELL];
				tInfo->sharedData->inhPyrConnRatio   = 100.0 / tInfo->nNeuronsTotalType[PYRAMIDAL_CELL];

				tInfo->sharedData->randWeight   = 1.000;
				tInfo->sharedData->excWeight    = 0.010;
				tInfo->sharedData->pyrInhWeight = 0.020;
				tInfo->sharedData->inhPyrWeight = 0.100;

				if (simType[2] == 'l')
					tInfo->sharedData->inputSpikeRate = 0.0025;
				if (simType[2] == 'h')
					tInfo->sharedData->inputSpikeRate = 0.0100;

			}
			else if (simType[1] == '2') {
				tInfo->sharedData->pyrPyrConnRatio   =  900.0 / tInfo->nNeuronsTotalType[PYRAMIDAL_CELL];
				tInfo->sharedData->pyrInhConnRatio   =  100.0 / tInfo->nNeuronsTotalType[INHIBITORY_CELL];
				tInfo->sharedData->inhPyrConnRatio   = 1000.0 / tInfo->nNeuronsTotalType[PYRAMIDAL_CELL];

				tInfo->sharedData->randWeight   = 1.000; // 0.004;
				tInfo->sharedData->excWeight    = 0.004; // 0.004;
				tInfo->sharedData->pyrInhWeight = 0.002; // 0.004;
				tInfo->sharedData->inhPyrWeight = 0.010; // 0.010;

				if (simType[2] == 'l')
					tInfo->sharedData->inputSpikeRate = 0.0025;
				if (simType[2] == 'h')
					tInfo->sharedData->inputSpikeRate = 0.0100;
			}
		}
}

// uA, kOhm, mV, cm, uF
int main(int argc, char **argv) {

	int nProcesses = 1;
    int currentProcess=0;
#ifdef MPI_GPU_NN
    int threadLevel;
    //MPI_Init (&argc, &argv);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED , &threadLevel);
    MPI_Comm_rank (MPI_COMM_WORLD, &currentProcess);
    MPI_Comm_size (MPI_COMM_WORLD, &nProcesses);
    if (threadLevel == MPI_THREAD_SINGLE)
    	printf("MPI support enabled with %d processes and MPI_THREAD_SINGLE.\n", nProcesses);
    else if (threadLevel == MPI_THREAD_FUNNELED)
    	printf("MPI support enabled with %d processes and MPI_THREAD_FUNNELED.\n", nProcesses);
    else if (threadLevel == MPI_THREAD_SERIALIZED)
    	printf("MPI support enabled with %d processes and MPI_THREAD_SERIALIZED.\n", nProcesses);
    else if (threadLevel == MPI_THREAD_MULTIPLE)
    	printf("MPI support enabled with %d processes and MPI_THREAD_MULTIPLE.\n", nProcesses);
#endif

	bench.start = gettimeInMilli();

	ThreadInfo *tInfo = new ThreadInfo;
	tInfo->sharedData = new SharedNeuronGpuData;
    tInfo->sharedData->kernelInfo = new KernelInfo;

	tInfo->currProcess = currentProcess;
	tInfo->nProcesses 	= nProcesses;
	tInfo->globalThreadTypes = 0;

	tInfo->sharedData->nBarrier = 0;
	tInfo->sharedData->mutex = new pthread_mutex_t;
	tInfo->sharedData->cond = new pthread_cond_t;
	pthread_cond_init (  tInfo->sharedData->cond, NULL );
	pthread_mutex_init( tInfo->sharedData->mutex, NULL );

	tInfo->sharedData->synData = 0;
	tInfo->sharedData->hGpu = 0;
	tInfo->sharedData->hList = 0;
	tInfo->sharedData->globalSeed = time(NULL);

	benchConf.assertResultsAll = 1; // TODO: was 1
	benchConf.printSampleVms = 0;
	benchConf.printAllVmKernelFinish = 0;
	benchConf.printAllSpikeTimes = 1;
	benchConf.verbose = 0;

	int nNeuronsTotal = 0;

	if ( argc < 4 ) {
		printf("Invalid arguments!\n Usage: %s <mode> <simType> <nNeurons> <nGPUs> [seed]\n", argv[0]);
		printf("Invalid arguments!\n Usage: %s <mode> <simType> <nNeurons> <configFile> [seed]\n", argv[0]);

		exit(-1);
	}

	char mode = argv[1][0];
	assert (mode == 'C' || mode == 'G' || mode == 'H' || mode == 'B' || mode == 'T');

    char *simType = argv[2];

	nNeuronsTotal = atoi(argv[3]);
	assert ( 0 < nNeuronsTotal && nNeuronsTotal < 4096*4096);

    if (simType[0] == 'c') {
    	char *configFile = argv[4];
    	configureSimulation(simType, tInfo, nNeuronsTotal, mode, configFile);
    }
    else {
    	tInfo->sharedData->nThreadsCpu = atoi(argv[4]);
    	configureSimulation(simType, tInfo, nNeuronsTotal, mode, 0);
    }

	if (argc > 5)
		tInfo->sharedData->globalSeed = atoi(argv[4])*123;

	int nThreadsCpu = tInfo->sharedData->nThreadsCpu;

	// Configure the simulationSteps
	tInfo->sharedData->randBuf = new random_data *[nThreadsCpu];
	if(tInfo->sharedData->profileKernel)
		tInfo->sharedData->profiler = new KernelProfiler(2, nThreadsCpu, tInfo->nTypes, tInfo->currProcess);

	pthread_t *thread1 = new pthread_t[nThreadsCpu];
	ThreadInfo *tInfoArray = createInfoArray(nThreadsCpu, tInfo);
	for (int t=0; t<nThreadsCpu; t++) {

			if (mode == 'C' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchHostExecution, &(tInfoArray[t]));

			if (mode == 'G' || mode == 'H' || mode == 'B' || mode == 'T')
				pthread_create ( &thread1[t], NULL, launchDeviceExecution, &(tInfoArray[t]));

			//pthread_detach(thread1[t]);
	}

	for (int t=0; t<nThreadsCpu; t++)
		 pthread_join( thread1[t], NULL);

	bench.finish = gettimeInMilli();
	bench.finishF = (bench.finish - bench.start)/1000.; 

	// TODO: The total number of neurons is wrong
	tInfo->sharedData->neuronInfoWriter->writeResultsToFile(mode, nNeuronsTotal, tInfo->nComp[0], simType, bench);
	//delete tInfo->sharedData->neuronInfoWriter;

	if (tInfo->sharedData->profileKernel) {
		printf("PROFILER: Kernel\n");
		tInfo->sharedData->profiler->printProfile(0);
		printf("PROFILER: Communication\n");
		tInfo->sharedData->profiler->printProfile(1);
		printf("PROFILER: Total per thread\n");
		tInfo->sharedData->profiler->printProfile(-1);

		char *simType = argv[2];
		int nConnPerNeuron = 100;
		if (simType[1]=='2') nConnPerNeuron = 1000;
		int rateLevel = 0;
		if (simType[2]=='h') rateLevel = 1;

		tInfo->sharedData->profiler->printProfileToFile(
				nNeuronsTotal, nProcesses, tInfo->totalTypes, nConnPerNeuron, rateLevel);
	}

	delete[] tInfo->nNeurons;
	delete[] tInfo->nComp;
	delete tInfo;
	delete[] tInfoArray;

	printf ("Finished Simulation!!!\n");

#ifdef MPI_GPU_NN
    MPI_Finalize();
#endif

	return 0;
}

// Used only to check the number of spikes joined in the HashMap
int spkTotal = 0;
int spkEqual = 0;

