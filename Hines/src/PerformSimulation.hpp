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

#ifndef PERFORMSIMULATION_H_
#define PERFORMSIMULATION_H_

class PerformSimulation {

private:
	ThreadInfo * tInfo;
	SharedNeuronGpuData *sharedData;
	KernelInfo *kernelInfo;

public:
    PerformSimulation(ThreadInfo *tInfo);
    int launchExecution();
    int performHostExecution();
private:
    void syncCpuThreads();
    void updateBenchmark();
    void createNeurons();
    void initializeThreadInformation();
    void updateGenSpkStatistics(int *& nNeurons, SynapticData *& synData);
    void generateRandomSpikes(int type, RandomSpikeInfo & randomSpkInfo);

#ifdef MPI_GPU_NN
    void prepareMpiGeneratedSpikeStructures();
    void mpiAllGatherConnections();
    void broadcastGeneratedSpikesMPISync();
#endif
};

#endif /* PERFORMSIMULATION_H_ */

