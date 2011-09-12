/*
 * ActiveChannels.cpp
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#include "ActiveChannels.hpp"
#include <cmath>
#include <cstdio>

ActiveChannels::ActiveChannels(ftype dt_, int nActiveComp_, ucomp *activeCompList_, ftype *vmListNeuron_) {

	dt = dt_;

	this->compListSize = nActiveComp_;
	this->compList = activeCompList_;

	memory = new ftype[compListSize * 7];
	n = memory;
	h = n + compListSize;
	m = h + compListSize;
	gNaBar = m + compListSize;
	gKBar  = gNaBar + compListSize;
	gNaChannel  = gKBar + compListSize;
	gKChannel  = gNaChannel + compListSize;
	vmList = vmListNeuron_;

	/**
	 * TODO: Check if is when curr = 0;
	 */
	for (int i=0; i<compListSize; i++) {
		n[i] = 0.3177;
		h[i] = 0.5960;
		m[i] = 0.0529;
	}
}

ActiveChannels::~ActiveChannels() {
	delete[] memory;
	delete[] compList;
}


void ActiveChannels::evaluateCurrents( ) {

	evaluateGates();

	/**
	 * Update the channel conductances
	 */
	for (int i=0; i<compListSize; i++) {
		gNaChannel[i] = gNaBar[i] * m[i] * m[i] * m[i] * h[i];
		gKChannel[i]  =  gKBar[i] * n[i] * n[i] * n[i] * n[i];
	}
}

/**
 * Find the gate openings in the next time step
 * m(t + dt) = a + b m(t - dt)
 */
void ActiveChannels::evaluateGates(  ) {

	ftype alpha, beta, a, b, V;

	for (int i=0; i<compListSize; i++) {
		V = vmList[ compList[i] ];

		// gate m
		alpha = (V != 25.0) ? (0.1 * (25 - V)) / ( expf( 0.1 * (25-V) ) - 1 ) : 1;
		beta  = 4 * expf ( -V/18 );
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	m[i] = a + b * m[i];

		// gate h
		alpha =  0.07 * expf ( -V/20 );
		beta  = 1 / ( expf( (30-V)/10 ) + 1 );
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	h[i] = a + b * h[i];

	 	// gate n
		alpha = (V != 10.0) ? (0.01 * (10 - V)) / ( expf( 0.1 * (10-V) ) - 1 ) : 0.1;
		beta  = 0.125 * expf ( -V/80 );
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	n[i] = a + b * n[i];
	}
}
