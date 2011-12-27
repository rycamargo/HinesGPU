/*
 * ActiveChannels.h
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#ifndef ACTIVECHANNELS_H_
#define ACTIVECHANNELS_H_

#define EXPONENTIAL 0   // A exp((v-V0)/B)
#define SIGMOID 1 		// A / (exp((v-V0)/B) + 1)
#define LINOID 2        // A (v-V0) / (exp((v-V0)/B) - 1)

#include "Definitions.hpp"

class ActiveChannels {
	ftype dt;
	int nActiveComp;
	ucomp *activeCompList;

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

	ActiveChannels(ftype dt, int nActiveComp_, ucomp *activeCompList_, ftype *vmListNeuron_, int nComp);
	virtual ~ActiveChannels();


	void setActiveChannels();

	int getCompListSize () { return nActiveComp; }
	ucomp *getCompList () { return activeCompList; }


	void evaluateCurrents( ftype *Rm, ftype *active );
	void evaluateGates();

	/**
	 * New implementation
	 */
	ucomp *ucompMem;
	ftype *ftypeMem;

	ucomp *channelInfo; // nGates(0) comp(1) gatePos(3)
	ftype *channelEk;
	ftype *channelGbar;

	int nChannels;    //

	ucomp *gateInfo;    // gatePower(0): function alpha (1) and function beta (2)
	ftype *gatePar;   // parameters of alpha (A, B, V0) (0,1,2) and beta (3,4,5) functions

	ftype *gateState; // opening of the gates, indexed by gatePos in the channelInfo

	int nComp;
	ftype gSoma; // Contains the soma active conductances. Used when not triangularizing.

	ftype *eLeak; // contains the eLEak of the active compartments

	ftype getSomaCurrents () {return gSoma;}

	void evaluateCurrentsNew( ftype *Rm, ftype *active );
	void evaluateGatesNew();

	void createChannelList (int nChannels, ucomp *nGates, ucomp *comp, ftype *channelEk, ftype *gBar, ftype *eLeak);

	void setGate (int channel, int gate, ftype state, ucomp gatePower,
			ucomp alpha, ftype alphaA, ftype alphaB, ftype alphaV0,
			ucomp beta, ftype betaA, ftype betaB, ftype betaV0);
};

#endif /* ACTIVECHANNELS_H_ */
