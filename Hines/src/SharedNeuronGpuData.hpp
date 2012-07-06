#ifndef SHAREDNEURONGPUDATA_HPP
#define SHAREDNEURONGPUDATA_HPP

#include "Definitions.hpp"
#include "PlatformFunctions.hpp"
//#include "HinesMatrix.hpp"
//#include "SpikeStatistics.hpp"
//#include "NeuronInfoWriter.hpp"

struct SharedNeuronGpuData {
	class HinesMatrix **matrixList;	// Only exists for the types of the MPI process
	struct HinesStruct **hList;
	struct SynapticData *synData;
	struct HinesStruct **hGpu;

	random_data **randBuf;

	class SpikeStatistics *spkStat;
	struct NeuronInfoWriter *neuronInfoWriter;

	pthread_cond_t *cond;
	pthread_mutex_t *mutex;
	int nBarrier;

	struct KernelInfo *kernelInfo;
	int nThreadsCpu;

	int *typeList;
	class Connections *connection;
	struct ConnectionInfo *connInfo;
	struct ConnGpu **connGpuListHost;
	struct ConnGpu **connGpuListDevice;

	ftype inputSpikeRate;
	ftype pyrConnRatio;
	ftype inhConnRatio;
	ftype excWeight;
	ftype pyrInhWeight;
	ftype inhPyrWeight;

	ftype totalTime;
	ftype dt;
	ftype minDelay; // connectionDelay
	ftype maxDelay; // connectionDelay

	unsigned int globalSeed;


};

#endif
