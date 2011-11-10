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

