/*
 * SpikeStatistics.cpp
 *
 *  Created on: 10/09/2009
 *      Author: rcamargo
 */

#include "SpikeStatistics.hpp"
#include "HinesMatrix.hpp"

#include <cstdio>

SpikeStatistics::SpikeStatistics(int *nNeurons, int nTypes, int *typeList, int startTypeProcess, int endTypeProcess) {

    nSpkfile = 0;
    lastSpkfile = 0;

	this->typeList = typeList;
	this->nNeurons = nNeurons;
	this->nTypes   = nTypes;

	totalGeneratedSpikes = new ftype *[nTypes];
	totalReceivedSpikes  = new ftype *[nTypes];
	lastGeneratedSpikeTimes = new ftype *[nTypes];

	for (int type=0; type<nTypes; type++) {
		totalGeneratedSpikes[type] = new ftype[nNeurons[type]];
		totalReceivedSpikes[type]  = new ftype[nNeurons[type]];
		lastGeneratedSpikeTimes[type] = new ftype[nNeurons[type]];

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {
			totalGeneratedSpikes[type][neuron] = 0;
			totalReceivedSpikes[type][neuron]  = 0;
			lastGeneratedSpikeTimes[type][neuron] = 0;
		}
	}

	totalNeurons = 0;
	pyrNeurons = 0;
	inhNeurons = 0;
	for (int type=startTypeProcess; type<endTypeProcess; type++) {

		totalNeurons += nNeurons[type];
		if (typeList[type] == PYRAMIDAL_CELL)
			pyrNeurons += nNeurons[type];
		else if (typeList[type] == INHIBITORY_CELL)
			inhNeurons += nNeurons[type];
	}

}

SpikeStatistics::~SpikeStatistics() {

	if (nSpkfile != 0) fclose(nSpkfile);
	if (lastSpkfile != 0) fclose(lastSpkfile);
}

void SpikeStatistics::addGeneratedSpikes(int type, int neuron, ftype *spikeTimes, int nSpikes) {
	totalGeneratedSpikes[type][neuron] += nSpikes;
	if (spikeTimes != NULL)
		lastGeneratedSpikeTimes[type][neuron] = spikeTimes[nSpikes-1];
	else
		lastGeneratedSpikeTimes[type][neuron] = 0.00;
}

void SpikeStatistics::addReceivedSpikes(int type, int neuron, int nReceivedSpikes) {
	totalReceivedSpikes[type][neuron] += nReceivedSpikes;
}

void SpikeStatistics::printKernelSpikeStatistics( ftype currentTime) {

    if (nSpkfile == 0)
    	nSpkfile = fopen("nSpikeKernel.dat", "w");

    if (lastSpkfile == 0)
    	lastSpkfile = fopen("lastSpikeKernel.dat", "w");

	for (int type=0; type<nTypes; type++) {
		fprintf(nSpkfile,	"%-10.2f\ttype=%d | ", currentTime, type);
		fprintf(lastSpkfile,"%-10.2f\ttype=%d | ", currentTime, type);

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {

			fprintf(nSpkfile, 	 "%10.1f ", totalGeneratedSpikes[type][neuron]);
			fprintf(lastSpkfile, "%10.2f ", lastGeneratedSpikeTimes[type][neuron]);
		}

		fprintf(nSpkfile, 	 "\n");
		fprintf(lastSpkfile, "\n");
	}

	fprintf(nSpkfile, 	 "\n");
	fprintf(lastSpkfile, "\n");
}

void SpikeStatistics::printSpikeStatistics(const char *filename, ftype currentTime, BenchTimes & bench, int startTypeProcess, int endTypeProcess) {

//	ftype genSpikes = 0;
//	ftype recSpikes = 0;

	bench.meanGenSpikes    = 0;
	bench.meanRecSpikes    = 0;
	bench.meanGenPyrSpikes = 0;
	bench.meanRecPyrSpikes = 0;
	bench.meanGenInhSpikes = 0;
	bench.meanRecInhSpikes = 0;

	FILE *outFile = fopen(filename, "w");
	fprintf(outFile, "# totalTime=%f, totalNeurons=%d, nTypes=%d\n", currentTime, totalNeurons, nTypes);

	for (int type=startTypeProcess; type<endTypeProcess; type++) {
		for (int neuron=0; neuron < nNeurons[type]; neuron++) {

			bench.meanGenSpikes += totalGeneratedSpikes[type][neuron];
			bench.meanRecSpikes += totalReceivedSpikes[type][neuron];

			fprintf(outFile, "[%2d][%6d]\t%.1f\t%.1f\t%10.2f\n",
					type, neuron, totalGeneratedSpikes[type][neuron],
					totalReceivedSpikes[type][neuron], lastGeneratedSpikeTimes[type][neuron]);
		}

		if (typeList[type] == PYRAMIDAL_CELL)
			for (int neuron=0; neuron < nNeurons[type]; neuron++) {
				bench.meanGenPyrSpikes += totalGeneratedSpikes[type][neuron];
				bench.meanRecPyrSpikes += totalReceivedSpikes[type][neuron];
			}
		else if (typeList[type] == INHIBITORY_CELL)
			for (int neuron=0; neuron < nNeurons[type]; neuron++) {
				bench.meanGenInhSpikes += totalGeneratedSpikes[type][neuron];
				bench.meanRecInhSpikes += totalReceivedSpikes[type][neuron];
			}

	}

	bench.meanGenSpikes /= totalNeurons;
	bench.meanRecSpikes /= totalNeurons;
	bench.meanGenPyrSpikes /= pyrNeurons;
	bench.meanRecPyrSpikes /= pyrNeurons;
	bench.meanGenInhSpikes /= inhNeurons;
	bench.meanRecInhSpikes /= inhNeurons;


	printf("meanGenSpikes[T|P|I]=[%3.2f|%3.2f|%3.2f]\nmeanRecSpikes[T|P|I]=[%5.2f|%5.2f|%5.2f]\n",
			bench.meanGenSpikes, bench.meanGenPyrSpikes, bench.meanGenInhSpikes,
			bench.meanRecSpikes, bench.meanRecPyrSpikes, bench.meanRecInhSpikes);

	fclose(outFile);
}

