/**
 * SynapticChannels.h
 *
 *  Created on: 07/08/2009
 *      Author: rcamargo
 */

#ifndef SYNAPTICCHANNELS_H_
#define SYNAPTICCHANNELS_H_

#include "Definitions.hpp"
#include <vector>
#include <pthread.h>

#define SYNAPSE_AMPA 0
#define SYNAPSE_GABA 1

// List of spikes generated by a single neuron during the last kernel call
struct SpikeList {
	ftype *spikeTimes;
	ftype delay;
	ftype weight;
	int nGeneratedSpikes;
};

// Set of the spikes generated by each neuron during the last kernel call
typedef std::vector<SpikeList> SpikeSet;
typedef SpikeSet::const_iterator SpikeSetIterator;

// For each synapse in the neuron, maintains a set with the list of received spikes
typedef std::vector<SpikeSet> SynapseSpikeSet;
typedef SynapseSpikeSet::const_iterator SynapseSpikeSetIterator;

class SpikeInfo{

public:
	SpikeInfo(ftype time, ftype weight) {
		this->time = time; this->weight = weight;
	}
	SpikeInfo(ftype time) {
		this->time = time; this->weight = 0;
	}

	ftype time;
	ftype weight;
};

struct spkcomp {
	bool operator()(SpikeInfo s1, SpikeInfo s2) const {
		return (s1.time < s2.time) ? true : false;
	}
};

class SynapticChannels {

	void createChannelsAndSynapses(int nComp);
	ftype getCurrent(ucomp synType, ftype spikeTime, ftype weight, ucomp comp, ftype currTime);

	pthread_mutex_t addSpikeMutex;

	int nAddedSpikes;

	// Keeps the random spikes added on each iteration
	std::vector<ucomp> randomSpikeComp;
	std::vector<ftype> randomSpikeTimes;
	std::vector<ftype> randomSpikeWeights;

public:

	int synapseListSize;
	int *nDelieveredSpikes;
	int nRandom;

	/**
	 * Compartment where each synapse channel is located
	 */
	ucomp *synapseCompList;

	/**
	 * The type of each synaptic channel from synapseCompList. The types are defined
	 * in the header of this file
	 */
	ucomp *synapseTypeList;

	/**
	 * The start position in the spikeList for each synapse from synapseList
	 */
	ucomp *synSpikeListPos;
	ucomp *synSpikeListTmp;
	int spikeListSize;

	/**
	 * Contains the spike list for each synapse in the form:
	 * | spk1 spk2 spk3 ... spkn	| spk1 spk2 spk3 ... spkm	| ... | spk1 spk2 spk3 ... spkh	|
	 * |		synapse1			|		synapse2			| ... | 		synapse_k		|
	 */
	ftype *spikeList;

	/**
	 * Contains the weight for the connection of each spike
	 */
	ftype *synapseWeightList;

	/**
	 * Pointers to data structures from HinesMatrix
	 */
	ftype *synapticCurrent;
	ftype *vmList;

	SynapseSpikeSet synSpikeSet;

	int nNewSpikes;

	/**
	 * Synaptic constants
	 */
	int nChannelTypes;
	ftype *tau, *gmax, *esyn;

	SynapticChannels(ftype *synapticCurrent, ftype *vmList, int nComp);
	virtual ~SynapticChannels();

	void printSpikeTimes(ftype time);

	int getAndResetNumberOfAddedSpikes();

	int getNumberOfAddedSpikes() { return nAddedSpikes;}

	void updateSpikeList(ftype time);
	void updateSpikeListGpu(ftype time, ftype *spikeListGlobal, ftype *weightListGlobal,
			int maxSpikeNeuron, int nNeurons, int neuron, int type);


	void addSpikeList(ucomp synapse, int nGeneratedSpikes, ftype *spikeTimes, ftype delay, ftype weight);
	void addSpike(ucomp synapse, ftype time, ftype weight);

	void evaluateCurrents(ftype currTime);

	void printSpikes(ucomp synapse);
};

#endif /* SYNAPTICCHANNELS_H_ */
