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

	void evaluateCurrents( );
	void evaluateGates();

};

#endif /* ACTIVECHANNELS_H_ */
