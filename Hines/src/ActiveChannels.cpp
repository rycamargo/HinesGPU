/*
 * ActiveChannels.cpp
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#include "ActiveChannels.hpp"
#include <cmath>
#include <cstdio>

#define PYR_M_ALPHA (V != 25.0) ? (0.1 * (25 - V)) / ( expf( 0.1 * (25-V) ) - 1 ) : 1

#define N_GATE_FIELDS 3

#define GATE_POWER 0
#define ALPHA_FUNCTION 1
#define BETA_FUNCTION 2

#define N_CHANNEL_FIELDS 3

#define CH_NGATES 0
#define CH_COMP 1
#define CH_BAR 2

#define N_GATE_FUNC_PAR 3

#define A  0
#define B  1
#define V0 2

#define EXPONENTIAL 0   // A exp((v-V0)/B)
#define SIGMOID 1 		// A / (exp((v-V0)/B) + 1)
#define LINOID 2        // A (v-V0) / (exp((v-V0)/B) - 1)

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

void ActiveChannels::evaluateCurrents(	ftype *Rm, ftype *active ) {

	evaluateGates();

	/**
	 * Update the channel conductances
	 */
	for (int i=0; i<compListSize; i++) {

		gNaChannel[i] = gNaBar[i] * m[i] * m[i] * m[i] * h[i];
		gKChannel[i]  =  gKBar[i] * n[i] * n[i] * n[i] * n[i];

		unsigned int comp = compList[i];
		active[ comp ] -= gNaChannel[i] * ENa ;
		active[ comp ] -= gKChannel[i] * EK  ;
		active[ comp ] -=  ( 1 / Rm[comp] ) * ( ELeak );
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
		alpha = (V != 10.0) ? (0.01 * (10 - V)) / ( expf( 0.1 * (10-V) ) - 1 ) : 0.1; // LINOID: 0.1; A=-0.01, B=-10, V0=10
		beta  = 0.125 * expf ( -V/80 );
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	n[i] = a + b * n[i];
	}
}



void ActiveChannels::evaluateCurrentsNew( ftype *Rm, ftype *active ) {

	evaluateGates();

	/**
	 * Update the channel conductances
	 */
	int pos = 0;
	for (int ch=0; ch<nChannels; ch++) {

		int nGates     = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		ftype gChannel = channelInfo[ch*N_CHANNEL_FIELDS + CH_BAR];

		for (int gt=0; gt < nGates; gt++, pos++) {
			switch( gateInfo[pos * N_GATE_FIELDS + GATE_POWER] ) {
			case 4:
				gChannel *= (gateState[pos]*gateState[pos]*gateState[pos]*gateState[pos]);
				break;
			case 3:
				gChannel *= (gateState[pos]*gateState[pos]*gateState[pos]);
				break;
			case 2:
				gChannel *= (gateState[pos]*gateState[pos]);
				break;
			case 1:
				gChannel *= gateState[pos];
				break;
			default:
				gChannel *= powf(gateState[pos], gateInfo[pos * N_GATE_FIELDS + GATE_POWER] );
				break;
			}
		}

		int comp = channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP];
		active[ comp ] -= gChannel * ENa ;
	}

	for (int i=0; i<compListSize; i++) {
		gNaChannel[i] = gNaBar[i] * m[i] * m[i] * m[i] * h[i];
		gKChannel[i]  =  gKBar[i] * n[i] * n[i] * n[i] * n[i];
	}
}

/**
 * Find the gate openings in the next time step
 * m(t + dt) = a + b m(t - dt)
 */
void ActiveChannels::evaluateGatesNew(  ) {

	ftype alpha, beta, a, b;

	ftype* gate = gatePar;

	int pos;
	for (int ch=0; ch<nChannels; ch++) {

		int nGates = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		ftype V = vmList[ channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP] ];

		for (int gt=0; gt < nGates; gt++, pos++) {

            // (EXPONENTIAL): alpha(v) = A exp((v-V0)/B)
            // (SIGMOID):     alpha(v) = A / (exp((v-V0)/B) + 1)
            // (LINOID):      alpha(v) = A (v-V0) / (exp((v-V0)/B) - 1)

			// alpha_function
			ftype v0 = gate[V0];
			switch( gateInfo[pos * N_GATE_FIELDS + ALPHA_FUNCTION] ) {
			case EXPONENTIAL:
				alpha = gate[A] * expf((V-v0)/gate[B]);
				break;
			case SIGMOID:
				alpha = gate[A] / ( expf( (V-v0)/gate[B] ) + 1);
				break;
			case LINOID:
				alpha = (V != v0) ? gate[A] * (V-v0) / (expf((V-v0)/gate[B]) - 1) : gate[A] * gate[B];
				break;
			default:
				printf("Active channels parameters are invalid. Exiting...\n");
				exit(-1);
			}
			gate += N_GATE_FUNC_PAR;

			// beta_function
			v0 = gate[V0];
			switch( gateInfo[pos * N_GATE_FIELDS + BETA_FUNCTION] ) {
			case EXPONENTIAL:
				beta = gate[A] * expf((V-v0)/gate[B]);
				break;
			case SIGMOID:
				beta = gate[A] / ( expf( (V-v0)/gate[B] ) + 1);
				break;
			case LINOID:
				beta = (V != v0) ? gate[A] * (V-v0) / (expf((V-v0)/gate[B]) - 1) : gate[A] * gate[B];
				break;
			default:
				printf("Active channels parameters are invalid. Exiting...\n");
				exit(-1);
			}
			gate += N_GATE_FUNC_PAR;

			a = alpha / (1/dt + (alpha + beta)/2);
			b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
		}

		gateState[pos] = a + b * gateState[pos];
	}
}

void ActiveChannels::createChannelList (int nChannels, int *nGates, int *comp, ftype *vBar) {

	int nGatesTotal = 0;
	for (int i=0; i<nChannels; i++)
		nGatesTotal += nGates[i];

	this->nChannels   = nChannels;
	this->channelInfo = new int[nChannels * N_CHANNEL_FIELDS];
	this->gateInfo    = new int[nGatesTotal * N_GATE_FIELDS];
	this->gatePar     = new ftype[nGatesTotal * N_GATE_FUNC_PAR];

	this->gateState   = new ftype[nGatesTotal];
}

//int createChannel (int nGates, int comp, ftype vBar) {}

void ActiveChannels::setGate (int channel, int gate, int gatePower,
		ucomp alpha, ftype alphaA, ftype alphaB, ftype alphaV0,
		ucomp beta, ftype betaA, ftype betaB, ftype betaV0) {

}

