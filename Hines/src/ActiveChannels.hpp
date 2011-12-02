/*
 * ActiveChannels.h
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#ifndef ACTIVECHANNELS_H_
#define ACTIVECHANNELS_H_

#include "Definitions.hpp"

class ActiveChannels {
	ftype dt;
	int compListSize;
	ucomp *compList;

	public:

	ftype *memory;
	ftype *n;
	ftype *h;
	ftype *m;
	ftype *gNaBar;
	ftype *gKBar;
	ftype *vmList;

	ftype *gNaChannel;
	ftype *gKChannel;

	ftype ELeak;
	ftype EK;
	ftype ENa;

	ActiveChannels(ftype dt, int nActiveComp_, ucomp *activeCompList_, ftype *vmListNeuron);
	virtual ~ActiveChannels();


	int getCompListSize () { return compListSize; }
	ucomp *getCompList () { return compList; }


	void evaluateCurrents( ftype *Rm, ftype *active );
	void evaluateGates();

	/**
	 * New implementation
	 */
	int *channelInfo; // nGates(0) comp(1)
	int nChannels;
	int *gateInfo; // gatePower(0) função alpha (1) função beta (2)
	ftype *gatePar;// parâmetros de alpha (A, B, V0) (0,1,2) e beta (3,4,5)

	ftype *gateState;

	void evaluateCurrentsNew( ftype *Rm, ftype *active );
	void evaluateGatesNew();

	void createChannelList (int nChannels, int *nGates, int *comp, ftype *vBar);

	void setGate (int channel, int gate, int gatePower,
			ucomp alpha, ftype alphaA, ftype alphaB, ftype alphaV0,
			ucomp beta, ftype betaA, ftype betaB, ftype betaV0);
};

#endif /* ACTIVECHANNELS_H_ */
