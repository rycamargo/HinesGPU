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

CpuSimulationControl::CpuSimulationControl(ThreadInfo *tInfo) {

	this->tInfo = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void CpuSimulationControl::performCpuNeuronalProcessing() {

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

		int nNeurons = tInfo->nNeurons[type];

		for (int neuron = 0; neuron < nNeurons; neuron++) {

			HinesMatrix & m = sharedData->matrixList[type][neuron];

			/**
			 * Runs the simulation for kernelSteps for a single neuron
			 */
			m.nGeneratedSpikes = 0;
			for (int s=0; s < kernelInfo->nKernelSteps; s++)
				m.solveMatrix();

#ifdef MPI_GPU_NN
			sharedData->synData->nGeneratedSpikesHost[type][neuron] = m.nGeneratedSpikes;
#endif

			/**
			 * Check if Vm is ok for all neurons
			 */
			if (benchConf.assertResultsAll == 1) {
				HinesMatrix & m = sharedData->matrixList[type][neuron];
				if ( m.vmList[m.nComp-1] < -500 || 500 < m.vmList[m.nComp-1] ) {
					printf("type=%d neuron=%d %.2f\neuron", type, neuron, m.vmList[m.nComp-1]);
					assert(false);
				}
			}

		}
	}
}



// TODO: This function is deprecated
int PerformSimulation::performHostExecution() {

	/**
	 * Initializes thread information
	 */
	initializeThreadInformation( );

	/**------------------------------------------------------------------------------------
	 * Creates the neurons that will be simulated by the threads
	 *-------------------------------------------------------------------------------------*/
    createNeurons( );

	printf("process = %d | threadNumber = %d | types [%d|%d] | seed=%d \n", tInfo->currProcess, tInfo->threadNumber, tInfo->startTypeThread, tInfo->endTypeThread, tInfo->sharedData->globalSeed);

	int totalTypes 	= tInfo->totalTypes;
	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	FILE *vmKernelFile = fopen("vmKernel.dat", "w");
	FILE *sampleVmFile = fopen("sampleVm.dat", "w");

	int totalRandom = 0;

	int threadNumber = tInfo->threadNumber;
	int startType 	 = tInfo->startTypeThread;
	int endType   	 = tInfo->endTypeThread;

	if (threadNumber == 0)
		tInfo->sharedData->spkStat = new SpikeStatistics(tInfo->nNeurons, tInfo->totalTypes, tInfo->sharedData->typeList, startType, endType);

	// Benchmarking
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
	int kernelSteps = sharedData->kernelInfo->nKernelSteps;

	if (threadNumber == 0) {
		sharedData->connection = new Connections();
		sharedData->connection->connectRandom ( tInfo );
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

				/**
				 * Runs the simulation for kernelSteps for a single neuron
				 */
				m.nGeneratedSpikes = 0;
				for (int s=0; s < kernelSteps; s++) {
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
			sharedData->spkStat->printKernelSpikeStatistics((kStep+kernelSteps)*sharedData->matrixList[0][0].dt);
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
		sharedData->spkStat->printSpikeStatistics((char *)"spikeCpu.dat", totalTime, bench, startType, endType);

	//bench.execPrepare  = gettimeInMilli();
	//bench.execPrepareF = (bench.execPrepare - bench.matrixSetup)/1000.;

	//delete[] mList;
	if (threadNumber == 0)
		delete sharedData->connection;

	printf( "Solving in CPU is OK.\n");

	return 0;
}
