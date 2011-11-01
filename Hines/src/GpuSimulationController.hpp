#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>

class GpuSimulationController {

private:
	ThreadInfo * tInfo;
	SharedNeuronGpuData *sharedData;

public:
	GpuSimulationController(ThreadInfo *tInfo);
    int launchGpuExecution();

private:
    void prepareSynapses();
    int prepareExecution(int type);
    void writeSampleVm(ftype **& vmTimeSerie, int vmTimeSerieMemSize, int & kStep, FILE *& outFile, ftype & dt);
    void checkGpuCommunicationsSpikes(int & spikeListSizeMax, int & type, int & kStep, ftype & dt, int nThreadsComm);
    void updateBenchmark();
    int updateSpikeListSizeGlobal(int type, int maxSpikesNeuron);
    void transferSynapticSpikeInfoToGpu(int type, int spikeListSizeMax);
    void mpiAllGatherConnections();
    void syncCpuThreads();
    int generateRandomSpikes(int type, int kStep, ftype dt, int randomSpikeListSize, ftype *& randomSpikeTimes, int *& randomSpikeDest);
    void performCPUCommunication(int type, ftype & dt, int & kStep, int maxSpikesNeuron, int nRandom, int nThreadsComm);
    void performGPUCommunications(int & nRandom, int randomSpikeListSize, ftype *& randomSpikeTimes, int *& randomSpikeDest, int & nThreadsComm, int & sharedMemSizeComm, int & type, int & threadNumber, int *nBlocksComm, int *& nNeurons, SynapticData *& synData, HinesStruct **hGpu);
    void checkVmValues();
    void writeVmToFile(FILE *vmKernelFile, ftype & dt, int & kStep);
    void prepareGpuSpikeDeliveryStructures();
    void createGpuCommunicationStructures(int *nBlocksComm, struct cudaDeviceProp & prop, int maxThreadsComm);
    void addReceivedSpikesToTargetChannelCPU();
    void copyGeneratedSpikeListsToGPU();
    void readGeneratedSpikesFromGPU();
    void broadcastGeneratedSpikesMPISync();
    void defineThreadTypes();
    void createNeurons();

};
