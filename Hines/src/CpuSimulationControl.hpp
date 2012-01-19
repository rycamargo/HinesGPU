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

#ifndef CPUSIMULATIONCONTROL_H_
#define CPUSIMULATIONCONTROL_H_

class CpuSimulationControl {

private:
	ThreadInfo * tInfo;
	SharedNeuronGpuData *sharedData;
	KernelInfo *kernelInfo;

public:
	CpuSimulationControl(ThreadInfo *tInfo);
	//int launchGpuExecution();

//private:

    void performCpuNeuronalProcessing();
    void addReceivedSpikesToTargetChannelCPU();


//    void updateSharedDataInfo();
//    void prepareSynapses();
//    int  prepareExecution(int type);
//    void prepareGpuSpikeDeliveryStructures();
//    void createGpuCommunicationStructures();
//    void configureGpuKernel();
//
//    int  updateSpikeListSizeGlobal(int type, int maxSpikesNeuron);
//    void transferSynapticSpikeInfoToGpu(int type, int spikeListSizeMax);
//    void generateRandomSpikes(int type, RandomSpikeInfo & randomSpkInfo);
//    void performGPUCommunications(int type, RandomSpikeInfo & randomSpkInfo);
//
//
//    void copyGeneratedSpikeListsToGPU();
//    void readGeneratedSpikesFromGPU();
//
//    void performCPUCommunication(int type, int maxSpikesNeuron, int nRandom);
//    void addReceivedSpikesToTargetChannelCPU();
//
//
//
//    void checkGpuCommunicationsSpikes(int spikeListSizeMax, int type);
//    void checkVmValues();

};

#endif /* CPUSIMULATIONCONTROL_H_ */
