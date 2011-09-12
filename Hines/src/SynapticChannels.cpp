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

// TODO: check memory free
SynapticChannels::~SynapticChannels() {
	delete[] tau;

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
	nDelieveredSpikes = new int[synapseListSize];

	synapseCompList = new ucomp[4*synapseListSize];
	synapseTypeList = synapseCompList + synapseListSize;
	synSpikeListPos = synapseTypeList + synapseListSize;
	synSpikeListTmp = synSpikeListPos + synapseListSize;

	for (int i=0; i<synapseListSize; i++) {
		synSpikeListPos[i] = 0;
		synSpikeListTmp[i] = 0;
	}

	synapseCompList[0] = 0;
	synapseTypeList[0] = SYNAPSE_AMPA;

	synapseCompList[1] = nComp-1;
	synapseTypeList[1] = SYNAPSE_GABA;

	synSpikeSet.resize(2);
//	synSpikeMap[0] = new SpikeMap();
//	synSpikeMap[1] = new SpikeMap();
//	synSpikeSet.push_back( new SpikeSet() );
//	synSpikeSet.push_back( new SpikeSet() );

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
//	printf("time[%-5.1f]: ", time);
//
//	SpikeSetIterator it = synSpikeSet[0]->begin();
//
//	for (; it != synSpikeSet[0]->end(); it++) {
//		printf("%-5.1f ", it->spikeTimes);
//	}
//	printf("\n ");
}

void SynapticChannels::updateSpikeListGpu(ftype time, ftype *spikeListGlobal,
		ftype *weightListGlobal, int maxSpikeNeuron, int nNeurons, int neuron, int type) {

	pthread_mutex_lock (&addSpikeMutex);

	int nRemovedSpikes = 0;
	int randomPos = 0;
	int nRandom = 0;

	int startPosCurr[synapseListSize+1];
	int startPosNew[synapseListSize];

	nAddedSpikes = 0; // TODO: Should not be necessary to reevaluate again


	/**
	 * Determine the number of spikes
	 */
	for (int syn = 0; syn<synapseListSize; syn++) {
		startPosCurr[syn] = 0;

		ftype remThresh = time - (3 * (tau[2*synapseTypeList[syn]] + tau[2*synapseTypeList[syn]+1]) );

		//printf("%d %f%f\n", syn, time - remThresh);

		int spk = synSpikeListPos[syn];
		int lastSpk = (syn < synapseListSize-1) ? synSpikeListPos[syn+1] : spikeListSize;
		int spkMovePos = -1;
		for (; spk < lastSpk && spkMovePos == -1; spk++) {
			//assert (spikeListGlobal[spk * nNeurons + neuron] < time); // -------
			if (spikeListGlobal[spk * nNeurons + neuron] > remThresh)
				startPosCurr[syn]++;
			else {
				nRemovedSpikes++;
				spkMovePos = spk;
			}
		}
		for (; spk < lastSpk; spk++) {
			//assert (spikeListGlobal[spk * nNeurons + neuron] < time + 10); // -------
			if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
				startPosCurr[syn]++;
				spikeListGlobal[spkMovePos * nNeurons + neuron]  = spikeListGlobal[spk* nNeurons + neuron];
				weightListGlobal[spkMovePos * nNeurons + neuron] = weightListGlobal[spk* nNeurons + neuron];
				spkMovePos++;
			}
			else
				nRemovedSpikes++;
		}

		startPosNew[syn] = 0;
		for (int src=0; src < synSpikeSet[syn].size(); src++)
			startPosNew[syn] += synSpikeSet[syn][src].nGeneratedSpikes;

		for (int i=0; i<randomSpikeComp.size(); i++)
			if (randomSpikeComp[i] == syn) {
				startPosNew[syn]++;
				nRandom++;
			}

		nAddedSpikes += startPosNew[syn];

	}

	startPosCurr[synapseListSize] = 0;
	for (int syn = 0; syn<synapseListSize; syn++)
		startPosCurr[synapseListSize] += (startPosCurr[syn] + startPosNew[syn]);

	for (int syn = synapseListSize-1; syn >= 0; syn--) {
		startPosNew[syn]  = startPosCurr[syn+1] - startPosNew[syn];
		startPosCurr[syn] = startPosNew[syn] - startPosCurr[syn];
	}

	/**
	 * Used only for debugging
	 */
	for (int syn = 0; syn<synapseListSize; syn++)
		nDelieveredSpikes[syn] = startPosNew[syn] - startPosCurr[syn];

	assert (startPosCurr[0] == 0);

	/**
	 * Scans the spike list, copying the new generated spikes and the existing ones.
	 */
	int pos = 0;
	for (int syn=0; syn < synapseListSize ; syn++) {

		//assert (pos == startPosCurr[syn]);
		synSpikeListTmp[syn] = startPosCurr[syn];

		/*
		 * Copy the current spikes, removing the expired ones
		 */
		ftype remThresh = time - (3 * (tau[2*synapseTypeList[syn]] + tau[2*synapseTypeList[syn]+1]) );

		if (startPosCurr[syn] <= synSpikeListPos[syn]) {
			pos = startPosCurr[syn];
			int spk = synSpikeListPos[syn];
			int lastSpk = synSpikeListPos[syn] + startPosNew[syn] - startPosCurr[syn];
			//(syn < synapseListSize-1) ? synSpikeListPos[syn+1] : spikeListSize;

			for (; spk < lastSpk; spk++) {
				// Copy only the spikes not expired
				if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
					spikeListGlobal[pos * nNeurons + neuron]  = spikeListGlobal[spk * nNeurons + neuron];
					weightListGlobal[pos * nNeurons + neuron] = weightListGlobal[spk * nNeurons + neuron];
					pos++;
				}
			}
			//printf("A %d %d %d\n", syn, startPosCurr[syn], synSpikeListPos[syn]);
			assert (pos == startPosNew[syn]);
		}

		else {
			pos = startPosNew[syn]-1;
			int spk = synSpikeListPos[syn] + startPosNew[syn] - startPosCurr[syn] - 1;
			int lastSpk = synSpikeListPos[syn];

			for (; spk >= lastSpk; spk--) {
				// Copy only the spikes not expired
				if (spikeListGlobal[spk * nNeurons + neuron] > remThresh) {
					spikeListGlobal[pos * nNeurons + neuron]  = spikeListGlobal[spk * nNeurons + neuron];
					weightListGlobal[pos * nNeurons + neuron] = weightListGlobal[spk * nNeurons + neuron];
					pos--;
				}
			}
			//printf("B %d %d %d\n", syn, startPosCurr[syn], synSpikeListPos[syn]);
			assert (pos+1 == startPosCurr[syn]);
		}

	}

	for (int syn=0; syn < synapseListSize ; syn++) {

		/*
		 * Include the newly generated spikes
		 */
		pos = startPosNew[syn];
		int nSourceNeurons = synSpikeSet[syn].size();
		for (int src=0; src < nSourceNeurons; src++) {
			SpikeList spkList = synSpikeSet[syn][src];
			for (int spk = 0; spk < spkList.nGeneratedSpikes; spk++) {
				spikeListGlobal[pos * nNeurons + neuron]  = spkList.spikeTimes[spk]+spkList.delay;
				weightListGlobal[pos * nNeurons + neuron] = spkList.weight;
				pos++;
			}
		}
		synSpikeSet[syn].clear();

		/**
		 * Include the random spikes
		 */
		for (; randomPos < randomSpikeComp.size() && randomSpikeComp[randomPos] == syn; randomPos++ ) {
			spikeListGlobal[pos * nNeurons + neuron]  = randomSpikeTimes[randomPos];
			weightListGlobal[pos * nNeurons + neuron] = randomSpikeWeights[randomPos];
			pos++;
		}
	}

	assert (startPosCurr[synapseListSize] == spikeListSize + nAddedSpikes - nRemovedSpikes);
	assert(pos == startPosCurr[synapseListSize]);

	for (int i=0; i<synapseListSize; i++)
		synSpikeListPos[i] = synSpikeListTmp[i];

	spikeListSize += (nAddedSpikes - nRemovedSpikes);

	randomSpikeComp.clear();
	randomSpikeTimes.clear();
	randomSpikeWeights.clear();

	pthread_mutex_unlock (&addSpikeMutex);
}

void SynapticChannels::updateSpikeList(ftype time) {

	// Removes old spikes and finds number of spikes
	int nRemovedSpikes = 0;

	for (int syn=0; syn < synapseListSize; syn++) {
		ftype remThresh = time - (3 * (tau[2*synapseTypeList[syn]] + tau[2*synapseTypeList[syn]+1]) );
		int synEnd = (syn < (synapseListSize-1)) ? synSpikeListPos[syn+1] : spikeListSize;
		for (int i = synSpikeListPos[syn]; i < synEnd; i++) {
			if (spikeList[i] < remThresh) {
				spikeList[i] = -1;
				nRemovedSpikes++;
			}
		}
	}

	int oldSpikeListSize = spikeListSize;
	spikeListSize += (nAddedSpikes - nRemovedSpikes);

	// Creates a new spikeList based in the old one
	ftype *oldSpikeList = 0;
	ftype *oldWeightList = 0;
	if (spikeList != 0) {
		oldSpikeList = spikeList;
		oldWeightList = synapseWeightList;
	}
	spikeList = new ftype[2*spikeListSize];
	synapseWeightList = spikeList + spikeListSize;

	// Updates the new spikeList
	int iNew = 0, r = 0;
	for (int syn=0; syn < synapseListSize; syn++) {

		int synBegin = synSpikeListPos[syn];
		int synEnd   = (syn < (synapseListSize-1)) ? synSpikeListPos[syn+1] : oldSpikeListSize;

		synSpikeListPos[syn] = iNew; // Updates the synSpikeListPos after removing the old spikes

		// Copy the spikes from the previousList
		for (int i = synBegin; i < synEnd; i++) {
			if (oldSpikeList[i] >= 0) {
				spikeList[iNew]         = oldSpikeList[i];
				synapseWeightList[iNew] = oldWeightList[i];
				iNew++;
			}
		}

		// Include the newly generated spikes
		int setSize = synSpikeSet[syn].size();
		for (int i=0; i < setSize; i++) {
			SpikeList spkList = synSpikeSet[syn][i];
			for (int s = 0; s < spkList.nGeneratedSpikes; s++) {
				spikeList[iNew]         = spkList.spikeTimes[s]+spkList.delay;
				synapseWeightList[iNew] = spkList.weight;
				iNew++;
			}
		}
		synSpikeSet[syn].clear();

		// Include the random spikes
		for (; r < randomSpikeComp.size() && randomSpikeComp[r] == syn; r++ ) {
			spikeList[iNew] 	    = randomSpikeTimes[r];
			synapseWeightList[iNew] = randomSpikeWeights[r];
			iNew++;
		}
	}

	if (iNew!=spikeListSize){ // checks if all spikes were added
		printf("total %d %d \n", iNew, spikeListSize);
		assert(false);
	}

	// Remove the old list of spikes from memory
	if (oldSpikeList != 0) delete[] oldSpikeList;

	randomSpikeComp.clear();
	randomSpikeTimes.clear();
	randomSpikeWeights.clear();
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

void SynapticChannels::addSpike(ucomp synapse, ftype time, ftype weight) {

	pthread_mutex_lock (&addSpikeMutex);

	randomSpikeComp.push_back(synapse);
	randomSpikeTimes.push_back(time);
	randomSpikeWeights.push_back(weight);

	nAddedSpikes++;
	pthread_mutex_unlock (&addSpikeMutex);
}

void SynapticChannels::addSpikeList(ucomp synapse, int nGeneratedSpikes, ftype *spikeTimes, ftype delay, ftype weight) {

	pthread_mutex_lock (&addSpikeMutex);

	struct SpikeList spkList;
	spkList.spikeTimes = spikeTimes;
	spkList.delay      = delay;
	spkList.weight     = weight;
	spkList.nGeneratedSpikes = nGeneratedSpikes;
	synSpikeSet[synapse].push_back(spkList);

	nAddedSpikes += nGeneratedSpikes;
	pthread_mutex_unlock (&addSpikeMutex);

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
				}
			}
		}
	}

}
