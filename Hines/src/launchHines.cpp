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

#include "PerformSimulation.hpp"
#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
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


extern int performHostExecution(ThreadInfo *tInfo);


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
		tInfoArray[i].nNeurons		= model->nNeurons;
		tInfoArray[i].nComp			= model->nComp;

		tInfoArray[i].totalTypes		= model->totalTypes;
		tInfoArray[i].totalTypesProcess	= model->totalTypesProcess;

		tInfoArray[i].currProcess	= model->currProcess;

		tInfoArray[i].nProcesses	= model->nProcesses;

		tInfoArray[i].startTypeThread	= model->startTypeThread;
		tInfoArray[i].endTypeThread		= model->endTypeThread;
		tInfoArray[i].threadNumber		= model->threadNumber;
	}

	return tInfoArray;
}

void configureSimulation(char **argv, ThreadInfo *& tInfo, int & nNeurons, char & mode, int & nKernelSteps)
{
    char *simType = argv[5];
    if (simType[0] == 'p') {
		  printf ("Simulation configured as: Running performance experiments.\n");
			benchConf.printSampleVms = 1; //1
			benchConf.printAllVmKernelFinish = 1; //1
			benchConf.printAllSpikeTimes = 1; //1

			tInfo->sharedData->totalTime   = 1000; // 10s
			tInfo->sharedData->inputSpikeRate = 0.01;
			tInfo->sharedData->pyrConnRatio   = 100.0 / (nNeurons/2); // nPyramidal
			tInfo->sharedData->inhConnRatio   = 100.0 / (nNeurons/2); // nPyramidal

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '0') { // No
				tInfo->sharedData->pyrConnRatio   = 0;
				tInfo->sharedData->inhConnRatio   = 0;
				tInfo->sharedData->inputSpikeRate = -1;
			}
			if (simType[1] == '1') {
				tInfo->sharedData->pyrConnRatio   = 0;
				tInfo->sharedData->inhConnRatio   = 0;
				tInfo->sharedData->inputSpikeRate = 0.01; // 1 spike each 10 ms
			}


			if (simType[1] == '2') {
				tInfo->sharedData->excWeight    = 0.030;
				tInfo->sharedData->pyrInhWeight = 0.035;
				tInfo->sharedData->inhPyrWeight = 10;
				//tInfo->sharedData->excWeight    = 0.045;
				//tInfo->sharedData->pyrInhWeight = 0.020;
				//tInfo->sharedData->inhPyrWeight = 4;
				tInfo->sharedData->inputSpikeRate = 0.01;
			}
// pyr : conn1.weigth = 0.01
			// inh : conn1.weigth = 0.1
		}

		else if (simType[0] == 's') {
			printf ("Simulation configured as: Running nGPU experiments.\n");
			benchConf.printSampleVms = 0;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;

			tInfo->sharedData->totalTime   = 1000; // 1s
			tInfo->sharedData->inputSpikeRate = 0.01;

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '1') {
				tInfo->sharedData->pyrConnRatio   = 100.0 / (nNeurons/2); // nPyramidal
				tInfo->sharedData->inhConnRatio   = 100.0 / (nNeurons/2); // nPyramidal
				tInfo->sharedData->excWeight    = 0.030;
				tInfo->sharedData->pyrInhWeight = 0.035;
				tInfo->sharedData->inhPyrWeight = 10;
			}
			else if (simType[1] == '0') {
				tInfo->sharedData->pyrConnRatio   = 0; // nPyramidal
				tInfo->sharedData->inhConnRatio   = 0; // nPyramidal
			}
		}

		else if (simType[0] == 'n' || simType[0] == 'd') {
			printf ("Simulation configured as: Running scalability experiments.\n");

			benchConf.printSampleVms = 1;
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;
			benchConf.checkGpuComm = 0;
			if (mode=='G') {
				benchConf.simProcMode = NN_GPU;
				benchConf.simCommMode = NN_GPU;
			}
			else if (mode=='H') {
				benchConf.simProcMode = NN_GPU;
				benchConf.simCommMode = NN_CPU;
			}
			else if (mode=='C') {
				benchConf.simProcMode = NN_CPU;
				benchConf.simCommMode = NN_CPU;
			}

			if (simType[0] == 'n') benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;
			else if (simType[0] == 'd') benchConf.gpuCommBenchMode = GPU_COMM_DETAILED;

			tInfo->sharedData->totalTime   = 100;
			tInfo->sharedData->inputSpikeRate = 0.01;

			tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
			tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
			tInfo->sharedData->inhPyrWeight = 1;

			if (simType[1] == '0') { // 200k: 0.79
				tInfo->sharedData->pyrConnRatio   = 0; // nPyramidal
				tInfo->sharedData->inhConnRatio   = 0; // nPyramidal
				tInfo->sharedData->inputSpikeRate = 0.02; // increases the input
			}
			else if (simType[1] == '1') {
				tInfo->sharedData->pyrConnRatio   = 100.0 / (nNeurons/2); // nPyramidal
				tInfo->sharedData->inhConnRatio   = 100.0 / (nNeurons/2); // nPyramidal

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
				tInfo->sharedData->pyrConnRatio   = 1000.0 / (nNeurons/2); // nPyramidal
				tInfo->sharedData->inhConnRatio   = 1000.0 / (nNeurons/2); // nPyramidal

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
				tInfo->sharedData->pyrConnRatio   = connPerNeuron / (nNeurons/2); // nPyramidal
				tInfo->sharedData->inhConnRatio   = connPerNeuron / (nNeurons/2); // nPyramidal

				if (simType[2] == 'l') {
					tInfo->sharedData->excWeight    = 4.0/connPerNeuron;
					tInfo->sharedData->pyrInhWeight = 4.0/connPerNeuron;
					tInfo->sharedData->inhPyrWeight = 10;
				}
			}

			if (simType[3] != 0) {
				int batch[] = { 0, 100, 75, 50, 25, 10, 5, 1};
				char *posChar = new char[1];
				posChar[0] = simType[3];
				int pos = atoi(posChar);
				nKernelSteps = batch[pos];
				delete[] posChar;
			}

		}
}

// uA, kOhm, mV, cm, uF
int main(int argc, char **argv) {

	int nProcesses = 1;
    int currentProcess=0;
#ifdef MPI_GPU_NN
    MPI_Status status;
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

	if (argc < 5 ) {
		printf("Invalid arguments!\n Usage: %s <mode> <nNeurons> <nComp> <nThreads> [inpRat] [pyrRate] [inhRate]\n", argv[0]);
		exit(-1);
	}

	char mode = argv[1][0];
	assert (mode == 'C' || mode == 'G' || mode == 'H' || mode == 'B');
	int nNeurons = atoi(argv[2]);
	assert ( 0 < nNeurons && nNeurons < 4096*4096);
	int nComp = atoi(argv[3]);
	assert ( -4096*4096 < nComp && nComp < 4096*4096);
	int nThreads = atoi(argv[4]);
	assert ( 0 < nThreads && nThreads < 32);

	int nKernelSteps = 100;
	ThreadInfo *tInfo = new ThreadInfo;
	tInfo->sharedData = new SharedNeuronGpuData;
    tInfo->sharedData->kernelInfo = new KernelInfo;
	tInfo->sharedData->kernelInfo->nKernelSteps = nKernelSteps;

	tInfo->nProcesses 	= nProcesses;
	tInfo->totalTypesProcess = 2*nThreads;
	tInfo->totalTypes = tInfo->totalTypesProcess * nProcesses;

	tInfo->currProcess = currentProcess;

	tInfo->nNeurons = new int[tInfo->totalTypes];
	tInfo->nComp    = new int[tInfo->totalTypes];
	tInfo->sharedData->typeList = new int[tInfo->totalTypes];
	for (int i=0; i<tInfo->totalTypes; i += 2) {
		tInfo->nNeurons[i] = nNeurons/(tInfo->totalTypes);
		tInfo->nComp[i]    = nComp;
		tInfo->sharedData->typeList[i] = PYRAMIDAL_CELL;

		tInfo->nNeurons[i+1] = nNeurons/(tInfo->totalTypes);
		tInfo->nComp[i+1]    = tInfo->nComp[i];
		tInfo->sharedData->typeList[i+1] = INHIBITORY_CELL;
	}


	tInfo->sharedData->nBarrier = 0;
	tInfo->sharedData->mutex = new pthread_mutex_t;
	tInfo->sharedData->cond = new pthread_cond_t;
	pthread_cond_init (  tInfo->sharedData->cond, NULL );
	pthread_mutex_init( tInfo->sharedData->mutex, NULL );

	tInfo->sharedData->matrixList = new HinesMatrix *[tInfo->totalTypes];
	tInfo->sharedData->synData = 0;
	tInfo->sharedData->hGpu = 0;
	tInfo->sharedData->hList = 0;
	tInfo->sharedData->nThreadsCpu = nThreads;
	tInfo->sharedData->randBuf = new random_data *[nThreads];

	tInfo->sharedData->inputSpikeRate = 0.1;
	tInfo->sharedData->pyrConnRatio   = 0.1;
	tInfo->sharedData->inhConnRatio   = 0.1;

	tInfo->sharedData->totalTime   = 100; // in ms

	benchConf.assertResultsAll = 1;
	benchConf.printSampleVms = 0;
	benchConf.printAllVmKernelFinish = 0;
	benchConf.printAllSpikeTimes = 1;
	benchConf.verbose = 0;

	tInfo->sharedData->globalSeed = time(NULL);
	if (argc > 6)
	  tInfo->sharedData->globalSeed = atoi(argv[6])*123;

	if (argc > 5) {
		configureSimulation(argv, tInfo, nNeurons, mode, nKernelSteps);
	}

	pthread_t *thread1 = new pthread_t[nThreads];
	ThreadInfo *tInfoArray = createInfoArray(nThreads, tInfo);
	for (int t=0; t<nThreads; t++) {

			if (mode == 'C' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchHostExecution, &(tInfoArray[t]));

			if (mode == 'G' || mode == 'H' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchDeviceExecution, &(tInfoArray[t]));

			//pthread_detach(thread1[t]);
	}

	for (int t=0; t<nThreads; t++)
		 pthread_join( thread1[t], NULL);

	bench.finish = gettimeInMilli();
	bench.finishF = (bench.finish - bench.start)/1000.; 

	printf ("Setup=%-10.3f Prepare=%-10.3f Execution=%-10.3f Total=%-10.3f\n", bench.matrixSetupF, bench.execPrepareF, bench.execExecutionF, bench.finishF);
	printf ("HinesKernel=%-10.3f ConnRead=%-10.3f ConnWait=%-10.3f ConnWrite=%-10.3f\n", bench.totalHinesKernel, bench.totalConnRead, bench.totalConnWait, bench.totalConnWrite);
	printf ("%f %f %f\n", tInfo->sharedData->inputSpikeRate, tInfo->sharedData->pyrConnRatio, tInfo->sharedData->inhConnRatio);

	FILE *outFile;
	outFile = fopen("results.dat", "a");
	fprintf (outFile, "mode=%c neurons=%-6d types=%-2d comp=%-2d threads=%d ftype=%lu \
			meanGenSpikes[T|P|I]=[%-10.5f|%-10.5f|%-10.5f] meanRecSpikes[T|P|I]=[%-10.5f|%-10.5f|%-10.5f] \
			inpRate=%-5.3f pyrRatio=%-5.3f inhRatio=%-5.3f nKernelSteps=%d\n",
			mode, nNeurons, tInfo->totalTypes, nComp, nThreads, sizeof(ftype),
			bench.meanGenSpikes, bench.meanGenPyrSpikes, bench.meanGenInhSpikes,
			bench.meanRecSpikes, bench.meanRecPyrSpikes, bench.meanRecInhSpikes,
			tInfo->sharedData->inputSpikeRate, tInfo->sharedData->pyrConnRatio,
			tInfo->sharedData->inhConnRatio, nKernelSteps);
	fprintf (outFile, "Setup=%-10.3f Prepare=%-10.3f Execution=%-10.3f Total=%-10.3f\n", bench.matrixSetupF, bench.execPrepareF, bench.execExecutionF, bench.finishF);
	fprintf (outFile, "HinesKernel=%-10.3f ConnRead=%-10.3f ConnWait=%-10.3f ConnWrite=%-10.3f\n", bench.totalHinesKernel, bench.totalConnRead, bench.totalConnWait, bench.totalConnWrite);
	fprintf (outFile, "#------------------------------------------------------------------------------\n");

	printf ("nKernelSteps=%d\n", nKernelSteps);
	printf("\n");

	fclose(outFile);

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

