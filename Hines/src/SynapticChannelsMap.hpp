/**
 * SynapticChannels.h
 *
 *  Created on: 07/08/2009
 *      Author: rcamargo
 */

#ifndef SYNAPTICCHANNELS_H_
#define SYNAPTICCHANNELS_H_

#include "Definitions.hpp"
//#define __aligned__ ignored
//#include <tr1/unordered_map>
//#undef __aligned__
#include <vector>
#include <map>
#include <pthread.h>

#define SYNAPSE_AMPA 0
#define SYNAPSE_GABA 1

// commented due to incompatibilities between Ubuntu 10.04 and CUDA
//typedef std::tr1::unordered_map<ftype, ftype> SpikeMap;
typedef std::map<ftype, ftype> SpikeMap;
typedef SpikeMap::const_iterator SpikeMapIterator;

typedef std::vector<SpikeMap *> SynapseSpikeMap;
typedef SynapseSpikeMap::const_iterator SynapseSpikeMapIterator;

// Used only to check the number of spikes joined in the HashMap
extern int spkTotal;
extern int spkEqual;


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

public:

	int synapseListSize;

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

	SynapseSpikeMap synSpikeMap;

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

	void updateSpikeList(ftype time);

	void addSpikeList(ucomp synapse, int nGeneratedSpikes, ftype *spikeTimes, ftype delay, ftype weight);
	void addSpike(ucomp synapse, ftype time, ftype weight);

	void evaluateCurrents(ftype currTime);

	void printSpikes(ucomp synapse);
};

#endif /* SYNAPTICCHANNELS_H_ */
