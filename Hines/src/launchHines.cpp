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

#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "SpikeStatistics.hpp"

using namespace std;

struct ThreadInfo {
	SharedNeuronGpuData *sharedData;
	int *nNeurons;
	int *nComp;

	int totalTypes;
	int nThreadsCpu;
};


// Defined in HinesGpu.cu
//extern "C" {
extern int launchGpuExecution(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int totalTypes, int threadNumber);
//int launchGpuExecution(SharedNeuronGpuData *sharedData, int *nNeurons, int startType, int endType, int totalTypes, int threadNumber){return 0;}
//}
//int launchGpuExecution(HinesMatrix *matrixList, int nNeurons){}

void *launchDeviceExecution(void *ptr) {

	ThreadInfo *tInfo = (ThreadInfo *)ptr;
	int totalTypes 	= tInfo->totalTypes;
	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	/**
	 * Define the thread numbers
	 */
	pthread_mutex_lock (sharedData->mutex);
	int threadNumber = sharedData->nBarrier;
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);

	printf ("threadNumber = %d | nBarrier = %d \n", threadNumber, tInfo->sharedData->nBarrier);

	char *randstate = new char[256];
	//tInfo->sharedData->randBuf[threadNumber] = new random_data;
	tInfo->sharedData->randBuf[threadNumber] = (struct random_data*)calloc(1, sizeof(struct random_data));
	initstate_r(tInfo->sharedData->globalSeed + threadNumber,
			randstate, 256, tInfo->sharedData->randBuf[threadNumber]);

	int startType = threadNumber * (totalTypes / tInfo->nThreadsCpu);
	int endType   = (threadNumber+1) * (totalTypes / tInfo->nThreadsCpu);

	for (int type = startType; type < endType; type++) {

		int nComp 	 = tInfo->nComp[type];
		int nNeurons = tInfo->nNeurons[type];

		printf("nComp=%d nNeurons=%d seed=%d\n", nComp, nNeurons, tInfo->sharedData->globalSeed);

		sharedData->matrixList[type] = new HinesMatrix[nNeurons];

		//HinesMatrix *mList = mListPtr[0];

		for (int n = 0; n < nNeurons; n++) {
			HinesMatrix & m = sharedData->matrixList[type][n];

			if (nComp == 1)
				m.defineNeuronCableSquid();
			else
				m.defineNeuronTreeN(nComp, 1);

			m.createTestMatrix();
		}
	}

	bench.matrixSetup  = gettimeInMilli();
	bench.matrixSetupF = (bench.matrixSetup - bench.start)/1000.;

	//printf ("thread %d Sync \n", threadNumber);

	/**
	 * Synchronize threads before starting
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

	//printf ("thread %d OK \n", threadNumber);


	/**
	 * Launches the execution of all threads
	 */
	launchGpuExecution(sharedData, tInfo->nNeurons, startType, endType, totalTypes, threadNumber);

//	for (int type = 0; type < totalTypes; type++)
//		delete[] sharedData->matrixList[type];
//	delete[] sharedData->matrixList;
//	delete sharedData;

	return 0;
}


//void *launchHostExecution(int nNeurons, int nComp) {
void *launchHostExecution(void *ptr) {

	ThreadInfo *tInfo = (ThreadInfo *)ptr;
	int totalTypes 	= tInfo->totalTypes;
	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	FILE *vmKernelFile = fopen("vmKernel.dat", "w");
	FILE *nSpkfile 	   = fopen("nSpikeKernel.dat", "w");
	FILE *lastSpkfile  = fopen("lastSpikeKernel.dat", "w");
	FILE *sampleVmFile = fopen("sampleVm.dat", "w");

	int totalRandom = 0;

	/**
	 * Define the thread numbers
	 */
	pthread_mutex_lock (sharedData->mutex);
	int threadNumber = sharedData->nBarrier;
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);

	//printf("Starting thread %d ############################### \n", threadNumber);

	char *randstate = new char[256];
	tInfo->sharedData->randBuf[threadNumber] = (struct random_data*)calloc(1, sizeof(struct random_data));
	initstate_r(tInfo->sharedData->globalSeed + threadNumber,
			randstate, 256, tInfo->sharedData->randBuf[threadNumber]);

	int startType = threadNumber * (totalTypes / tInfo->nThreadsCpu);
	int endType   = (threadNumber+1) * (totalTypes / tInfo->nThreadsCpu);

	//printf("Creating matrices for %d neurons and %d comparts.\n", nNeurons, nComp);
	//HinesMatrix *mList = new HinesMatrix[nNeurons];

	for (int type = startType; type < endType; type++) {

		int nComp 	 = tInfo->nComp[type];
		int nNeurons = tInfo->nNeurons[type];

		printf("nCom=%d nNeurons=%d seed=%d\n", nComp, nNeurons, tInfo->sharedData->globalSeed);

		sharedData->matrixList[type] = new HinesMatrix[nNeurons];

		for (int n = 0; n < nNeurons; n++) {
			HinesMatrix & m = sharedData->matrixList[type][n];

			if (nComp == 1)
				m.defineNeuronCableSquid();
			else
				m.defineNeuronTreeN(nComp, 1);

			m.createTestMatrix();
		}
	}

	bench.matrixSetup  = gettimeInMilli();
	bench.matrixSetupF = (bench.matrixSetup - bench.start)/1000.;

	/**
	 * Synchronize threads before starting
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

	ftype totalTime = sharedData->totalTime; // 1 second
	int nSteps = totalTime / sharedData->matrixList[0][0].dt;
	int kernelSteps = sharedData->nKernelSteps;

	if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom (sharedData->pyrConnRatio, sharedData->inhConnRatio,
				sharedData->typeList, totalTypes, tInfo->nNeurons, tInfo->sharedData, threadNumber);
	}

	pthread_mutex_lock (sharedData->mutex);
	if (threadNumber == 0) {
		bench.totalHinesKernel = 0;
		bench.totalConnRead    = 0;
		bench.totalConnWait    = 0;
		bench.totalConnWrite   = 0;
	}
	pthread_mutex_unlock (sharedData->mutex);

	printf("Launching Host execution: thread=%d \n", threadNumber);// with %d neurons and %d comparts.\n", nNeurons, nComp);

	/**
	 * Synchronize threads before starting
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

	if (benchConf.printSampleVms == 1 && threadNumber == 0) {
		HinesMatrix & m = sharedData->matrixList[0][0];
		fprintf(sampleVmFile, "%10.2f\t%10.2f\t%10.2f\t%10.2f\t%10.2f\n", 0.0,
			sharedData->matrixList[0][0].vmList[m.nComp-1],
			sharedData->matrixList[0][1].vmList[m.nComp-1],
			sharedData->matrixList[0][2].vmList[m.nComp-1],
			sharedData->matrixList[0][3].vmList[m.nComp-1]);
		 //m.dt * currStep,

		//sharedData->matrixList[0][1].writeVmToFile(sampleVmFile);
		//if (tInfo->nNeurons[0] > 1) sharedData->matrixList[0][1].writeVmToFile(sampleVmFile);
	}

	if (threadNumber == 0) {
		bench.execPrepare  = gettimeInMilli();
		bench.execPrepareF = (bench.execPrepare - bench.matrixSetup)/1000.;
	}

	ftype **vmTimeSeries = new ftype *[4];
	vmTimeSeries[0] = new ftype[kernelSteps];
	vmTimeSeries[1] = new ftype[kernelSteps];
	vmTimeSeries[2] = new ftype[kernelSteps];
	vmTimeSeries[3] = new ftype[kernelSteps];

	for (int kStep = 0; kStep < nSteps; kStep += kernelSteps) {

		if (threadNumber == 0 && kStep % 100 == 0)
			printf("Starting CPU solver -----------> %d\n", kStep);

		if (threadNumber == 0)
			bench.kernelStart  = gettimeInMilli();

		/**
		 * Perform the simulation for kernelSteps for all neurons
		 */
		for (int type = startType; type < endType; type++) {

			//int nComp 	 = tInfo->nComp[type];
			int nNeurons = tInfo->nNeurons[type];

			for (int n = 0; n < nNeurons; n++) {

				HinesMatrix & m = sharedData->matrixList[type][n];

				m.nGeneratedSpikes = 0;
				int s=0;
				ftype inject = (ftype)20e-4;//20e-4;// + n * (ftype)1e-4; // in uA

				/**
				 * Runs the simulation for kernelSteps for a single neuron
				 */
				for (; s < kernelSteps; s++) {
					m.solveMatrix();
					if (benchConf.printSampleVms == 1) {
						if (type == 0 && n < 4)
							vmTimeSeries[n][s] = sharedData->matrixList[0][n].vmList[m.nComp-1];
					}
				}

				/**
				 * Check if Vm is ok for all neurons
				 */
				if (benchConf.assertResultsAll == 1) {
					HinesMatrix & m = sharedData->matrixList[type][n];
					if ( m.vmList[m.nComp-1] < -500 || 500 < m.vmList[m.nComp-1] ) {
						printf("type=%d neuron=%d %.2f\n", type, n, m.vmList[m.nComp-1]);
						assert(false);
					}
				}

			}

		}

		if (benchConf.printSampleVms == 1 && threadNumber == 0) {
			HinesMatrix & m = sharedData->matrixList[0][0];
			for (int s = 0; s < kernelSteps; s++)
				fprintf(sampleVmFile, "%10.2f\t%10.2f\t%10.2f\t%10.2f\t%10.2f\n", m.dt * (kStep+s+1),
						vmTimeSeries[0][s], vmTimeSeries[1][s], vmTimeSeries[2][s], vmTimeSeries[3][s]);
		}


		if (threadNumber == 0)
				bench.kernelFinish = gettimeInMilli();

		/**
		 * Perform the communications
		 */
		for (int type = startType; type < endType; type++) {

			int nNeurons = tInfo->nNeurons[type];
			for (int source = 0; source < nNeurons; source++) {

				HinesMatrix & m = sharedData->matrixList[type][source];

				if (m.nGeneratedSpikes > 0) {

					// Used to print spike statistics in the end of the simulation
					sharedData->spkStat->addGeneratedSpikes(type, source, m.spikeTimes, m.nGeneratedSpikes);

					std::vector<Conn> & connList = sharedData->connection->getConnArray(source + type*CONN_NEURON_TYPE);

					for (int conn=0; conn<connList.size(); conn++) {

						SynapticChannels * synDest = sharedData->matrixList[connList[conn].dest / CONN_NEURON_TYPE ][ connList[conn].dest % CONN_NEURON_TYPE ].synapticChannels;
						synDest->addSpikeList(connList[conn].synapse, m.nGeneratedSpikes, m.spikeTimes, connList[conn].delay, connList[conn].weigth);

						//for (int s=0; s < m.nGeneratedSpikes; s++) {
							//SynapticChannels * synDest = sharedData->matrixList[connList[conn]->dest / CONN_NEURON_TYPE ][ connList[conn]->dest % CONN_NEURON_TYPE ].synapticChannels;
							//synDest->addSpike(connList[conn]->synapse, m.spikeTimes[s] + connList[conn]->delay, connList[conn]->weigth);
						//}
					}
				}
			}
		}

		if (threadNumber == 0)
			bench.connRead = gettimeInMilli();

		/**
		 * Synchronize threads before communication
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

		if (threadNumber == 0)
			bench.connWait = gettimeInMilli();

		/**
		 * Prints the Vm at the end of each kernel execution
		 */
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1) {

			for (int type = 0; type < totalTypes; type++) {
				HinesMatrix *mList = sharedData->matrixList[type];
				fprintf(vmKernelFile, "dt=%-10.2f\ttype=%d\t", mList[0].dt * (kStep + kernelSteps), type);
				for (int n = 0; n < tInfo->nNeurons[type]; n++)
					fprintf(vmKernelFile, "%10.2f\t", mList[n].vmList[mList[n].nComp-1]);
				fprintf(vmKernelFile, "\n");
			}
		}

		for (int type = startType; type < endType; type++) {
			int nNeurons = tInfo->nNeurons[type];
			for (int n = 0; n < nNeurons; n++) {
				HinesMatrix & m = sharedData->matrixList[type][n];


				// Add some random spikes % TODO: starting at 10ms
				int nRandom = 0;
				if ((kStep + kernelSteps)*m.dt > 9.9999 && sharedData->inputSpikeRate > 0 && sharedData->typeList[type] == PYRAMIDAL_CELL) {
					int32_t spkTime;
					for (int t = 0; t < (int)(kernelSteps * m.dt); t++) { // At each ms
						random_r(sharedData->randBuf[threadNumber], &spkTime);
						if (spkTime / (float) RAND_MAX < sharedData->inputSpikeRate) {
							nRandom++;
							m.synapticChannels->addSpike(0,(kStep + kernelSteps)*m.dt + t + (ftype)spkTime/RAND_MAX/sharedData->inputSpikeRate, 1);
						}
					}

					ftype diff = (kernelSteps * m.dt) - (int)(kernelSteps * m.dt);
					if (diff > 0.01 ) {
						random_r(sharedData->randBuf[threadNumber], &spkTime);
						if (spkTime / (float) RAND_MAX < sharedData->inputSpikeRate * diff) {
							nRandom++;
							m.synapticChannels->addSpike(
									0,(kStep + kernelSteps)*m.dt + (ftype)spkTime/RAND_MAX/sharedData->inputSpikeRate, 1);
						}
					}

				}
				// Fires only at some predefines times
				else if (sharedData->inputSpikeRate < 0 && sharedData->typeList[type] == PYRAMIDAL_CELL) {
					if (n%3 == 1) {
						m.synapticChannels->addSpike(0,(kStep + kernelSteps)*m.dt, 1);
					}
					if (n%3 == 2) {
						m.synapticChannels->addSpike(0,(kStep + kernelSteps)*m.dt, 1);
						m.synapticChannels->addSpike(0,(kStep + 1.5*kernelSteps)*m.dt, 1);
					}
				}

				m.synapticChannels->updateSpikeList(m.dt * (kStep + kernelSteps));

				// Used to print spike statistics in the end of the simulation
				sharedData->spkStat->addReceivedSpikes(type, n, m.synapticChannels->getAndResetNumberOfAddedSpikes()-nRandom);
				totalRandom += nRandom;
			}
		}

		// Uses only data from SpikeStatistics::addGeneratedSpikes
		if (benchConf.printAllSpikeTimes == 1 && threadNumber == 0) {
			sharedData->spkStat->printKernelSpikeStatistics(nSpkfile, lastSpkfile,
					(kStep+kernelSteps)*sharedData->matrixList[0][0].dt);
		}

		if (threadNumber == 0)
			bench.connWrite = gettimeInMilli();

		if (threadNumber == 0) {
			bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
			bench.totalConnRead	  	+= (bench.connRead 		- bench.kernelFinish)/1000.;
			bench.totalConnWait		+= (bench.connWait 		- bench.connRead)/1000.;
			bench.totalConnWrite	+= (bench.connWrite 	- bench.connWait)/1000.;
		}
	}

	bench.execExecution  = gettimeInMilli();
	bench.execExecutionF = (bench.execExecution - bench.matrixSetup)/1000.;

	HinesMatrix & m = sharedData->matrixList[0][0];
	printf("%10.2f\t%10.5f\t%10.5f totalRandom=%d\n", m.dt * m.currStep, m.vmList[m.nComp-1], m.vmList[0], totalRandom);

	// Used to print spike statistics in the end of the simulation
	if (threadNumber == 0)
		sharedData->spkStat->printSpikeStatistics((char *)"spikeCpu.dat", totalTime, bench);

	//bench.execPrepare  = gettimeInMilli();
	//bench.execPrepareF = (bench.execPrepare - bench.matrixSetup)/1000.;

	//delete[] mList;
	if (threadNumber == 0)
		delete sharedData->connection;

	printf( "Solving in CPU is OK.\n");

	return 0;
}

// uA, kOhm, mV, cm, uF
int main(int argc, char **argv) {

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

	ThreadInfo *tInfo = new ThreadInfo;
	tInfo->sharedData = new SharedNeuronGpuData;
	tInfo->sharedData->nKernelSteps = 100;

	tInfo->nThreadsCpu = nThreads;

	tInfo->totalTypes = 2*nThreads;//8;

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
	tInfo->sharedData->nThreadsCpu = tInfo->nThreadsCpu;
	tInfo->sharedData->spkStat = new SpikeStatistics(tInfo->nNeurons, tInfo->totalTypes, tInfo->sharedData->typeList);
	tInfo->sharedData->randBuf = new random_data *[tInfo->nThreadsCpu];

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
	  
	// p -> precision with different float types
	
	if (argc > 5) {
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

			benchConf.printSampleVms = 1; // TODO: sould be 0
			benchConf.printAllVmKernelFinish = 0;
			benchConf.printAllSpikeTimes = 0;
			if (mode=='G') benchConf.gpuCommMode = GPU_COMM;
			else if (mode=='H') benchConf.gpuCommMode = CPU_COMM;

			if (simType[0] == 'n') benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;
			else if (simType[0] == 'd') benchConf.gpuCommBenchMode = GPU_COMM_DETAILED;

			tInfo->sharedData->totalTime   = 1000;
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
				tInfo->sharedData->nKernelSteps = batch[pos];
				delete[] posChar;
			}

		}
	}

	pthread_t *thread1 = new pthread_t[nThreads];
	for (int t=0; t<nThreads; t++) {

			if (mode == 'C' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchHostExecution, tInfo);

			if (mode == 'G' || mode == 'H' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchDeviceExecution, tInfo);

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
			tInfo->sharedData->inhConnRatio, tInfo->sharedData->nKernelSteps);
	fprintf (outFile, "Setup=%-10.3f Prepare=%-10.3f Execution=%-10.3f Total=%-10.3f\n", bench.matrixSetupF, bench.execPrepareF, bench.execExecutionF, bench.finishF);
	fprintf (outFile, "HinesKernel=%-10.3f ConnRead=%-10.3f ConnWait=%-10.3f ConnWrite=%-10.3f\n", bench.totalHinesKernel, bench.totalConnRead, bench.totalConnWait, bench.totalConnWrite);
	fprintf (outFile, "#------------------------------------------------------------------------------\n");

	printf ("nKernelSteps=%d\n", tInfo->sharedData->nKernelSteps);
	printf("\n");

	fclose(outFile);

	delete[] tInfo->nNeurons;
	delete[] tInfo->nComp;
	delete tInfo;

	printf ("Finished Simulation!!!\n");

	return 0;
}

// Used only to check the number of spikes joined in the HashMap
int spkTotal = 0;
int spkEqual = 0;

