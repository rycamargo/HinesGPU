/*
 * HinesMatrix.hpp
 *
 *  Created on: 05/06/2009
 *      Author: rcamargo
 */

//#include <iostream>
//#include <fstream>
#include <cstdio>
#include "ActiveChannels.hpp"
#include "SynapticChannels.hpp"
#include "Definitions.hpp"
//#include <fcntl.h>
using namespace std;

#ifndef HINESMATRIX_HPP_
#define HINESMATRIX_HPP_

#define PYRAMIDAL_CELL 0
#define INHIBITORY_CELL 1

class HinesMatrix {

public:
	ftype *memory;
	ucomp *ucompMemory;

	ftype *rhsM;
	ftype *vmList;
	ftype *vmTmp;

	ftype *leftList;
	ucomp *leftListLine;
	ucomp *leftListColumn;
	ucomp *leftStartPos;
	int leftListSize;

	/**
	 * Triangularized list
	 */
	ftype *triangList;

	/**
	 *  Used for triangSingle
	 **/
	ftype *mulList;
	ucomp *mulListDest;
	ucomp *mulListComp;
	int mulListSize;

	ftype *curr;

	ftype *Cm;
	ftype *Rm;
	ftype *Ra;

	ftype *active;

	int **juctions;
	ftype *rad;

	ftype RM, CM, RA;

	ftype vRest;

	ftype dx;

	int nComp;

	int currStep;
	ftype dt;

	FILE *outFile;

	ActiveChannels *activeChannels; // Active channels

	SynapticChannels *synapticChannels; // Synaptic channels

	/**
	 * Generated spikes
	 */
	// Contains the time of the last spike generated on each neuron
	ftype lastSpike;
	// Contains the time of the spikes generated in the current execution block
	ftype *spikeTimes;
	int spikeTimeListSize;
	int nGeneratedSpikes;

	ftype threshold; // in mV
	ftype minSpikeInterval; // in mV

	int triangAll;

	HinesMatrix();
	~HinesMatrix();

	void redefineGenSpikeTimeList( ftype *targetSpikeTimeListAddress );

	int getnComp() { return nComp; }
	void setCurrent(int comp, ftype value) {
		curr[comp] = value;
	}

	/**
	 *  [ 1 | 2 | 3 ]
	 */
	void defineNeuronCable();

	void defineNeuronTreeN(int nComp, int active);

	void defineNeuronSingle();

	void defineNeuronCableSquid();

	void initializeFieldsSingle();
	/*
	 * Create a matrix for the neuron
	 */
	void createTestMatrix();

	/**
	 * Performs the upper triangularization of the matrix
	 */
	void upperTriangularizeSingle();

	/***************************************************************************
	 * This part is executed in every integration step
	 ***************************************************************************/

	void upperTriangularizeAll();

	void findActiveCurrents();

	void updateRhs();

	void backSubstitute();

	void solveMatrix();

	/***************************************************************************
	 * This part is executed in every integration step
	 ***************************************************************************/

	void writeVmToFile(FILE *outFile);

	void printMatrix(ftype *list);

	void freeMem();
};


#endif /* HINESMATRIX_HPP_ */
