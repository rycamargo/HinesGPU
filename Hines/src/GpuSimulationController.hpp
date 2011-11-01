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

class GpuSimulationController {

private:
	ThreadInfo * tInfo;
	SharedNeuronGpuData *sharedData;
	KernelInfo *kernelInfo;
	int kStep;

public:
	GpuSimulationController(ThreadInfo *tInfo);
    int launchGpuExecution();

private:

    void createNeurons();
    void defineThreadTypes();
    void updateSharedDataInfo();
    void prepareSynapses();
    int  prepareExecution(int type);
    void prepareGpuSpikeDeliveryStructures();
    void createGpuCommunicationStructures();
    void configureGpuKernel();

    int  updateSpikeListSizeGlobal(int type, int maxSpikesNeuron);
    void transferSynapticSpikeInfoToGpu(int type, int spikeListSizeMax);
    void generateRandomSpikes(int type, RandomSpikeInfo & randomSpkInfo);
    void performCPUCommunication(int type, int maxSpikesNeuron, int nRandom);
    void performGPUCommunications(int type, RandomSpikeInfo & randomSpkInfo);
    void addReceivedSpikesToTargetChannelCPU();
    void copyGeneratedSpikeListsToGPU();
    void readGeneratedSpikesFromGPU();

    void mpiAllGatherConnections();
    void broadcastGeneratedSpikesMPISync();

    void checkGpuCommunicationsSpikes(int spikeListSizeMax, int type);
    void checkVmValues();

    void syncCpuThreads();
    void updateBenchmark();
};
