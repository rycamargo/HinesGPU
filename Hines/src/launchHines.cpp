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

#include "SharedNeuronGpuData.hpp"
#include "KernelInfo.hpp"
#include "ThreadInfo.hpp"
#include "PerformSimulation.hpp"
#include "Connections.hpp"
#include "HinesMatrix.hpp"
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

void configureNeuronTypes(char*& simType, ThreadInfo*& tInfo, int& nNeurons,  char *configFileName) {

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
			tInfo->nNeurons[i] = nNeurons / (tInfo->totalTypes);
			tInfo->nComp[i] = nComp;
			tInfo->sharedData->typeList[i] = PYRAMIDAL_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

			tInfo->nNeurons[i + 1] = nNeurons / (tInfo->totalTypes);
			tInfo->nComp[i + 1] = nComp;
			tInfo->sharedData->typeList[i + 1] = INHIBITORY_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

			tInfo->nNeurons[i + 2] = nNeurons / (tInfo->totalTypes);
			tInfo->nComp[i + 2] = nComp;
			tInfo->sharedData->typeList[i + 2] = BASKET_CELL;
			tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

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
					sscanf(strTmp, "%d", &nNeuronsTypeTotal[t]);
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
						nNeuronsGlobal[ tInfo->totalTypes ] = nNeuronsTypeTotal[t] * rateNeuronsType;
						typeListGlobal[ tInfo->totalTypes ] = t;
						tInfo->totalTypes++;
					}
				}
				endTypeThreadGlobal[iProcess * maxTh + iThread] = tInfo->totalTypes;
				endTypeProcessGlobal[iProcess] = tInfo->totalTypes;
			}



		}

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
		printf("nTypes=%d totalTypes=%d nProcesses=%d\n", tInfo->nTypes, tInfo->totalTypes, tInfo->nProcesses);
		for (int t=0; t < tInfo->totalTypes; t++)
			printf( "%d|%d ", nNeuronsGlobal[t], typeListGlobal[t]);
		printf( "\n");
		for (int t=0; t < tInfo->globalThreadTypesSize; t++)
			printf( "%d ", tInfo->globalThreadTypes[t]);
		printf( "\n");
		//*****************************************************

		// Check if the total number of neurons is correct
		for (int type=0; type < tInfo->nTypes; type++)
			assert(tInfo->nNeuronsTotalType[type] == nNeuronsTypeTotal[type]);


		// TODO: Implement the profiler

		fclose(fp);

		//exit(0);

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

    if (simType[0] == 'p') {
		  printf ("Simulation configured as: Running performance experiments.\n");
			benchConf.printSampleVms = 1; //1
			benchConf.printAllVmKernelFinish = 1; //1
			benchConf.printAllSpikeTimes = 1; //1

			tInfo->sharedData->totalTime   = 100; // 10s
			tInfo->sharedData->inputSpikeRate = 0.01;
			tInfo->sharedData->pyrPyrConnRatio   = 10.0 / (nNeurons/tInfo->nTypes); // nPyramidal
			tInfo->sharedData->pyrInhConnRatio   = 10.0 / (nNeurons/tInfo->nTypes); // nPyramidal

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '0') { // No
				tInfo->sharedData->pyrPyrConnRatio   = 0;
				tInfo->sharedData->pyrInhConnRatio   = 0;
				tInfo->sharedData->inputSpikeRate = -1;
			}
			if (simType[1] == '1') {
				tInfo->sharedData->pyrPyrConnRatio   = 0;
				tInfo->sharedData->pyrInhConnRatio   = 0;
				tInfo->sharedData->inputSpikeRate = 0.01; // 1 spike each 10 ms
			}


			if (simType[1] == '2') {
				tInfo->sharedData->excWeight    = 0.030;
				tInfo->sharedData->pyrInhWeight = 0.035;
				tInfo->sharedData->inhPyrWeight = 10;
				tInfo->sharedData->inputSpikeRate = 0.01;
			}
		}

		else if (simType[0] == 's') {
			printf ("Simulation configured as: Running nGPU experiments.\n");
			benchConf.printSampleVms = 0;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;

			tInfo->sharedData->totalTime   = 100; // 1s
			tInfo->sharedData->inputSpikeRate = 0.01;

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '1') {
				tInfo->sharedData->pyrPyrConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal
				tInfo->sharedData->pyrInhConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal
				tInfo->sharedData->excWeight    = 0.030;
				tInfo->sharedData->pyrInhWeight = 0.035;
				tInfo->sharedData->inhPyrWeight = 10;
			}
			else if (simType[1] == '0') {
				tInfo->sharedData->pyrPyrConnRatio   = 0; // nPyramidal
				tInfo->sharedData->pyrInhConnRatio   = 0; // nPyramidal
			}
		}

		else if (simType[0] == 'n' || simType[0] == 'd') {
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
			else if (simType[1] == '3' || simType[1] == '4') {
				ftype totalConn = 1000;
				if (simType[1] == '3')
					totalConn = 1 * 1000 * 1000;
				else if (simType[1] == '4')
					totalConn = 10 * 1000 * 1000;
				ftype connPerNeuron = totalConn / nNeurons;
				tInfo->sharedData->pyrPyrConnRatio   = connPerNeuron / (nNeurons/tInfo->nTypes); // nPyramidal
				tInfo->sharedData->pyrInhConnRatio   = connPerNeuron / (nNeurons/tInfo->nTypes); // nPyramidal

				if (simType[2] == 'l') {
					tInfo->sharedData->excWeight    = 4.0/connPerNeuron;
					tInfo->sharedData->pyrInhWeight = 4.0/connPerNeuron;
					tInfo->sharedData->inhPyrWeight = 10;
				}
			}

		}


		else if (simType[0] == 'c') {
			printf ("Simulation configured as: Running cluster experiments.\n");

			benchConf.printSampleVms = 1;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;
			benchConf.checkGpuComm = 0;

			if (mode=='G')      benchConf.setMode(NN_GPU, NN_GPU);
			else if (mode=='H') benchConf.setMode(NN_GPU, NN_CPU);
			else if (mode=='C') benchConf.setMode(NN_CPU, NN_CPU);
			else if (mode=='T') benchConf.setMode(NN_GPU, NN_TEST);

			benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;

			tInfo->sharedData->totalTime   = 1000;
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

	int nThreadsCpu = 0;
	int nNeuronsTotal = 0;

	if ( argc < 4 ) {
		printf("Invalid arguments!\n Usage: %s <mode> <nNeurons> <nThreads> <simConf> [seed]\n", argv[0]);
		printf("Invalid arguments!\n Usage: %s <mode> <simConf> <confFile> [seed]\n", argv[0]);
		exit(-1);
	}

	char mode = argv[1][0];
	assert (mode == 'C' || mode == 'G' || mode == 'H' || mode == 'B' || mode == 'T');

    char *simType = argv[2];
    if (simType[0] == 'c') {
    	char *configFile = argv[3];

    	configureSimulation(simType, tInfo, 0, mode, configFile);
    	nThreadsCpu = tInfo->sharedData->nThreadsCpu; // This value can be changed inside configureSimulation
    	for (int i=0; i<tInfo->totalTypes; i++)
    		nNeuronsTotal += tInfo->nNeurons[i];

    	if (argc > 4)
    		tInfo->sharedData->globalSeed = atoi(argv[4])*123;
    }
    else {

    	nNeuronsTotal = atoi(argv[3]);
    	assert ( 0 < nNeuronsTotal && nNeuronsTotal < 4096*4096);

    	nThreadsCpu = atoi(argv[4]);
    	assert ( 0 < nThreadsCpu && nThreadsCpu < 32);
    	tInfo->sharedData->nThreadsCpu = nThreadsCpu;

    	configureSimulation(simType, tInfo, nNeuronsTotal, mode, 0);

    	if (argc > 5)
    	  tInfo->sharedData->globalSeed = atoi(argv[5])*123;
    }


	// Configure the simulationSteps
	tInfo->sharedData->randBuf = new random_data *[nThreadsCpu];

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
	tInfo->sharedData->neuronInfoWriter->writeResultsToFile(mode, nNeuronsTotal, tInfo->nComp[0], bench);
	//delete tInfo->sharedData->neuronInfoWriter;

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

