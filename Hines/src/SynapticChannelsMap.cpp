/*
 * SynapticChannels.cpp
 *
 *  Created on: 07/08/2009
 *      Author: rcamargo
 */

#include "SynapticChannels.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>

SynapticChannels::SynapticChannels(ftype *synapticCurrent, ftype *vmList, int nComp) {
	this->synapticCurrent = synapticCurrent;
	this->vmList = vmList;
	this->nAddedSpikes = 0;
	spikeList = 0;
	spikeListSize = 0;
	synapseWeightList = 0;

	pthread_mutex_init( &addSpikeMutex, NULL );
	createChannelsAndSynapses(nComp);
}

SynapticChannels::~SynapticChannels() {
	delete[] tau;

	delete synSpikeMap[0];
	delete synSpikeMap[1];

	if (synapseCompList != 0) delete[] synapseCompList;
	if (spikeList != 0) delete[] spikeList;
	//if (synapseWeightList != 0) delete[] synapseWeightList;
}

void SynapticChannels::createChannelsAndSynapses(int nComp) {

	nChannelTypes = 2;
	tau  = new ftype[4*nChannelTypes];
	gmax =  tau + 2*nChannelTypes;//new ftype[nChannelTypes];
	esyn = gmax +   nChannelTypes;//new ftype[nChannelTypes];

	/**
	 * Creates AMPA channel
	 */
	tau[2*SYNAPSE_AMPA] = 4;
	tau[2*SYNAPSE_AMPA+1] = 4;
	gmax[SYNAPSE_AMPA] = 20 * 500e-9;
	esyn[SYNAPSE_AMPA] = 70;

	tau[2*SYNAPSE_GABA] = 4;//17;
	tau[2*SYNAPSE_GABA+1] = 4;//100;
	gmax[SYNAPSE_GABA] = 20 * 1250e-9;
	esyn[SYNAPSE_GABA] = -75;

	/**
	 * Creates synapses
	 */
	synapseListSize = 2;

	synapseCompList = new ucomp[3*synapseListSize];
	synapseTypeList = synapseCompList + synapseListSize; //new ucomp[synapseListSize];
	synSpikeListPos = synapseTypeList + synapseListSize; //new ucomp[synapseListSize];

	synapseCompList[0] = 0;
	synapseTypeList[0] = SYNAPSE_AMPA;

	synapseCompList[1] = nComp-1;
	synapseTypeList[1] = SYNAPSE_GABA;

//	synSpikeMap[0] = new SpikeMap();
//	synSpikeMap[1] = new SpikeMap();
	synSpikeMap.push_back( new SpikeMap() );
	synSpikeMap.push_back( new SpikeMap() );

}

ftype SynapticChannels::getCurrent(ucomp synType, ftype spikeTime, ftype weight, ucomp comp, ftype currTime) {

	//printf ("weight=%f\n", weight);
	ftype gsyn = 0;
	ftype current = 0;
	if (synType == SYNAPSE_AMPA) {
		ftype r = (currTime - spikeTime) / tau[2*SYNAPSE_AMPA];
		gsyn = gmax[SYNAPSE_AMPA] * r * exp(1 - r) * weight;
		current = (vmList[comp] - esyn[SYNAPSE_AMPA]) * gsyn;
	}
	else if (synType == SYNAPSE_GABA) {
		ftype r = (currTime - spikeTime) / tau[2*SYNAPSE_GABA];
		gsyn = gmax[SYNAPSE_GABA] * r * exp(1 - r) * weight;
		current = (vmList[comp] - esyn[SYNAPSE_GABA]) * gsyn;
	}
	else
		printf ("ERROR: SynapticChannels::getCurrent -> Defined synapse type not found.\n");

	return current;
}

void SynapticChannels::printSpikeTimes(ftype time) {
	printf("time[%-5.1f]: ", time);

	SpikeMapIterator it = synSpikeMap[0]->begin();

	for (; it != synSpikeMap[0]->end(); it++) {
		printf("%-5.1f ", it->first);
	}
	printf("\n ");
}

void SynapticChannels::updateSpikeList(ftype time) {

	int nSpikes = 0;

	int removeKeysSize = 10000;
	ftype removeKeys[removeKeysSize];
	int removeKeyPos;

	// Remove old spikes and finds number of spikes
	//SynapseSpikeMapIterator it;
	for(int syn = 0; syn < synSpikeMap.size(); syn++) {

		ftype remThresh = 3 * (tau[ 2*synapseTypeList[syn] ] + tau[ 2*synapseTypeList[syn] + 1 ]); // was 2
		SpikeMap * synSpikes = synSpikeMap[syn];
		SpikeMapIterator itlow, itup;

		SpikeMapIterator spkIt = synSpikes->begin();
		for (removeKeyPos = 0; spkIt != synSpikes->end(); spkIt++)
			if ( time - spkIt->first > remThresh )
				removeKeys[removeKeyPos++] = spkIt->first;
		while (removeKeyPos > 0)
			synSpikes->erase(removeKeys[--removeKeyPos]);

		assert(removeKeyPos < removeKeysSize);

		// Counts the number of spikes
		spkIt = synSpikes->begin();
		for (; spkIt != synSpikes->end(); spkIt++)
			nSpikes++;
		//nSpikes += synSpikes->size();
	}

	/**
	 * tests the number of spikes
	 */
//	int testSpike=0;
//	int testSpike2=0;
//	for(int syn = 0; syn < synSpikeMap.size(); syn++) {
//		SpikeMap *synSpikes = synSpikeMap[syn];
//		SpikeMapIterator itSpk = synSpikes->begin();
//		for (; itSpk != synSpikes->end(); itSpk++)
//			testSpike++;
//		testSpike2 += synSpikes->size();
//	}
//	if (testSpike != nSpikes || testSpike2 != nSpikes)
//		printf("Wrong number of spikes: testSpike=%d|%d nSpikes=%d\n", testSpike, testSpike2, nSpikes);
	// Apenas o segundo bate. O primeiro às vezes dá maior ou dá menor

	if (spikeList != 0) delete[] spikeList;

	spikeListSize = nSpikes;
	spikeList   = new ftype[2*spikeListSize]; // ValGrind
	synapseWeightList = spikeList + spikeListSize;

	// Create new list of spikes

	int spike=0;
	for (int i=0; i<synapseListSize; i++)
		synSpikeListPos[i] = 0;

	bool printA = false;
	int synTmp=0;
	for(int syn = 0; syn < synSpikeMap.size(); syn++) {

		synTmp = syn;
		synSpikeListPos[syn] = spike;
		SpikeMap * synSpikes = synSpikeMap[syn];

		SpikeMapIterator itSpk = synSpikes->begin();
		for (; itSpk != synSpikes->end(); itSpk++) {
			if (spike < nSpikes) {
				spikeList[spike] = itSpk->first;
				synapseWeightList[spike] = itSpk->second; // ValGrind
				spike++;
			}
		}
	}

	if (spike < nSpikes)
		printf("Smaller number of spikes: used=%d allocated=%d\n", spike, nSpikes);
	else if (spike > nSpikes)
		printf("Larger number of spikes: used=%d allocated=%d\n", spike, nSpikes);

	for (synTmp = synTmp + 1; synTmp<synapseListSize; synTmp++)
		synSpikeListPos[synTmp] = spike;
}

void SynapticChannels::printSpikes(ucomp synapse) {
//	std::deque<ftype> & synSpikes = newSpikeMap[synapse];
//	for (int i = 0; i < synSpikes.size(); i++)
//		printf ("%f\t",spikeList[i]);
//	printf("A\n");
}

int SynapticChannels::getAndResetNumberOfAddedSpikes() {
	int tmp = nAddedSpikes;
	nAddedSpikes = 0;
	return tmp;
}

// TODO: Optimize with a checking with all the GPUs
void SynapticChannels::addSpike(ucomp synapse, ftype time, ftype weight) {

	pthread_mutex_lock (&addSpikeMutex);
	(*synSpikeMap[synapse])[time] += weight;
	nAddedSpikes++;

	// Used only to check the number of spikes joined in the HashMap
	spkTotal++;
	if ( (*synSpikeMap[synapse])[time] != weight ) spkEqual++;

	pthread_mutex_unlock (&addSpikeMutex);
}

void SynapticChannels::addSpikeList(ucomp synapse, int nGeneratedSpikes, ftype *spikeTimes, ftype delay, ftype weight) {

	pthread_mutex_lock (&addSpikeMutex);
	for (int s=0; s < nGeneratedSpikes; s++) {
		(*synSpikeMap[synapse])[spikeTimes[s]+delay] += weight;

		// Used only to check the number of spikes joined in the HashMap
		spkTotal++;
		if ( (*synSpikeMap[synapse])[spikeTimes[s]+delay] != weight ) spkEqual++;
	}
	pthread_mutex_unlock (&addSpikeMutex);
	nAddedSpikes += nGeneratedSpikes;
}

void SynapticChannels::evaluateCurrents(ftype currTime) {

	if (spikeListSize > 0) {
		for (int syn=0, spike=0; syn < synapseListSize; syn++) {
			int synComp = synapseCompList[syn];
			int lastSpikePos = spikeListSize;
			if ( syn < (synapseListSize-1) )
				lastSpikePos = synSpikeListPos[syn+1];

			for (; spike < lastSpikePos; spike++) {

				if (lastSpikePos > spikeListSize) {
					printf("A spike=%d last=%d listSize=%d syn=%d\n", spike, lastSpikePos, spikeListSize, syn);
					assert (false);
				}

				if (currTime >= spikeList[spike]) {
					ftype current = this->getCurrent(
							synapseTypeList[syn], spikeList[spike], synapseWeightList[spike], synComp, currTime);
					synapticCurrent [synComp] += current;
					//if (synComp == 0) printf ("time=%f current=%f\n", currTime, current);
				}
			}
		}
	}

}
