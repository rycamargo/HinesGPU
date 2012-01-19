#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include "NeuronInfoWriter.hpp"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>

#ifndef GPUSIMULATIONCONTROL_H_
#define GPUSIMULATIONCONTROL_H_

class GpuSimulationControl {

private:
	ThreadInfo * tInfo;
	SharedNeuronGpuData *sharedData;
	KernelInfo *kernelInfo;

public:
	GpuSimulationControl(ThreadInfo *tInfo);
	//int launchGpuExecution();

//private:

    void updateSharedDataInfo();
    void prepareSynapses();
    int  prepareExecution(int type);
    void prepareGpuSpikeDeliveryStructures();
    void createGpuCommunicationStructures();
    void configureGpuKernel();

    void performGPUCommunications(int type, RandomSpikeInfo & randomSpkInfo);
    void performGpuNeuronalProcessing();

    void copyGeneratedSpikeListsToGPU();
    void readGeneratedSpikesFromGPU();

    void checkVmValues();

    static void addToInterleavedSynapticActivationList(
    		ftype *activationList, ucomp *activationListPos, int activationListSize, int neuron, int nNeurons,
    		ftype currTime, ftype dt, ucomp synapse, ftype spikeTime, ftype delay, ftype weight);

};

#endif /* GPUSIMULATIONCONTROL_H_ */

