#ifndef HINESSTRUCT_HPP
#define HINESSTRUCT_HPP

#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "SpikeStatistics.hpp"
//#include "Connections.hpp"
#include <cstdlib>
#include <pthread.h>

typedef struct {

	int type;

	ucomp *ucompMemory;
	ucomp *mulListDest;
	ucomp *mulListComp;
	int mulListSize;

	ftype *memoryS;
	ftype *memoryE;

	/**
	 * Hines matrix and auxiliary data
	 */
	ftype *triangList;

	ftype *leftList;
	ucomp *leftListLine;
	ucomp *leftListColumn;
	ucomp *leftStartPos;
	int leftListSize;

	ftype *rhsM;
	ftype *vmList;
	ftype *vmTmp;
	ftype *mulList;


	/**
	 * Information about the compartments
	 */
	int nComp;
	int nNeurons;
	ftype *Cm;
	ftype *Rm;
	ftype *Ra;
	ftype vRest;
	ftype dx;

	ftype *curr;   // Injected current

	/******************************************************************
	 * Active Channels
	 ******************************************************************/
	ftype *active; // Active current
	int compListSize; // Number of compartments with active channels
	ucomp *compList; // List of compartments with active channels
	ftype *n;
	ftype *h;
	ftype *m;
	ftype *gNaBar;
	ftype *gKBar;
	ftype *gNaChannel;
	ftype *gKChannel;
	ftype ELeak;
	ftype EK;
	ftype ENa;
	int triangAll;


	/******************************************************************
	 * Synaptic Channels
	 ******************************************************************/

	int nChannelTypes;
	int synapseListSize;
	//Compartment where each synapse is located
	ucomp *synapseCompList;
	// The type of each synapse from synapseCompList.
	ucomp *synapseTypeList;


	//The start position in the spikeList for each synapse from synapseList
	//ucomp *synSpikeListPos;
	int *synSpikeListPos;
	// Contains the spike list for each synapticChannel
	ftype *spikeList;
	int spikeListSize;
	// Contains the weight to consider for the synapse of each spike
	ftype *synapseWeightList;

	// Constants of the synaptic channels
	ftype *tau, *gmax, *esyn;

	/******************************************************************
	 * Generated spikes
	 ******************************************************************/

	// Contains the time of the last spike generated on the neuron
	ftype lastSpike;
	// Contains the time of the spikes generated in the current execution block
	ftype *spikeTimes;
	int spikeTimeListSize;
	// Number of spikes generated in the current block (not a vector, just a pointer to a memory location)
	ucomp *nGeneratedSpikes;

	ftype threshold; // in mV
	ftype minSpikeInterval; // in mV

	/**********************************************************************/


	/**
	 * Holds the results that will be copied to the CPU
	 */
	ftype *vmTimeSerie;

	/**
	 * Simulation information
	 */
	int currStep;
	ftype dt;

} HinesStruct;

/**
 *
 */

typedef struct {
	ftype **spikeListDevice;
	ftype **weightListDevice;
	int **spikeListPosDevice;
	int **spikeListSizeDevice;

	ftype **spikeListGlobal;
	ftype **weightListGlobal;
	int **spikeListPosGlobal;
	int **spikeListSizeGlobal;

	/**
	 * Pointers to the number of spikes generated by the neurons of each type
	 */
	ucomp **nGeneratedSpikesHost;   // [type][neuron]    	[MPI]
	ucomp **nGeneratedSpikesDevice; //

	ucomp ***nGeneratedSpikesGpusDev;  //					[MPI]
	ucomp ***nGeneratedSpikesGpusHost; //

	/**
	 * List of generated spikes from the thread types
	 */
	ftype **genSpikeTimeListHost;
	ftype **genSpikeTimeListDevice;

	/**
	 * List of generated spikes from the types of all threads
	 */
	ftype ***genSpikeTimeListGpusHost;
	ftype ***genSpikeTimeListGpusDev;

	ftype **vmListHost;
	ftype **vmListDevice;

	int totalTypes;
} SynapticData;

typedef struct {
	HinesMatrix **matrixList;
	HinesStruct **hList;
	SynapticData *synData;
	HinesStruct **hGpu;

	random_data **randBuf;

	SpikeStatistics *spkStat;

	pthread_cond_t *cond;
	pthread_mutex_t *mutex;
	int nBarrier;

	int nThreadsCpu;

	int nKernelSteps; // Number of integration steps performed on each kernel call

	int *typeList;
	class Connections *connection;
	struct MPIConnectionInfo *connInfo;
	struct ConnGpu **connGpuListHost;
	struct ConnGpu **connGpuListDevice;

	ftype inputSpikeRate;
	ftype pyrConnRatio;
	ftype inhConnRatio;
	ftype excWeight;
	ftype pyrInhWeight;
	ftype inhPyrWeight;

	ftype totalTime;

	unsigned int globalSeed;


} SharedNeuronGpuData;

typedef struct {
	SharedNeuronGpuData *sharedData; 	// Shared among the threads
	int *nNeurons;						// Shared among the threads
	int *nComp;							// Shared among the threads

	int totalTypes;
	int totalTypesProcess;

	int currProcess;

	int nProcesses;
	int nThreadsCpu;

	int startType;
	int endType;
	int threadNumber;
} ThreadInfo;

#endif
