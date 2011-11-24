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

void CpuSimulationControl::performCPUCommunication(int type, int maxSpikesNeuron, int nRandom) {

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
}

void CpuSimulationControl::addReceivedSpikesToTargetChannelCPU()
{

	if( benchConf.simProcMode == NN_CPU && tInfo->nProcesses == 1) {

		for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

			for (int source = 0; source < tInfo->nNeurons[type]; source++) {

				ucomp nGeneratedSpikes = sharedData->matrixList[type][source].nGeneratedSpikes;
				if (nGeneratedSpikes > 0) {
					ftype *spikeTimes = sharedData->matrixList[type][source].spikeTimes;

					std::vector<Conn> & connList = sharedData->connection->getConnArray(source + type*CONN_NEURON_TYPE);
					for (unsigned int conn=0; conn<connList.size(); conn++) {
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


