/*
 * ActiveChannels.cpp
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#include "ActiveChannels.hpp"
#include <cmath>
#include <cstdio>
#include <cassert>

#define PYR_M_ALPHA (V != 25.0) ? (0.1 * (25 - V)) / ( expf( 0.1 * (25-V) ) - 1 ) : 1

ActiveChannels::ActiveChannels(ftype dt_, ftype *vmListNeuron_, int nComp) {

	this->dt = dt_;
	this->nComp = nComp;

	this->vmList = vmListNeuron_;

}

void ActiveChannels::setActiveChannels(int nActiveComp_, ucomp *activeCompList_) {


	this->channelInfo = 0;

	this->nActiveComp    = nActiveComp_;
	this->activeCompList = activeCompList_;

	memory = new ftype[nActiveComp * 7];
	n = memory;
	h = n + nActiveComp;
	m = h + nActiveComp;
	gNaBar = m + nActiveComp;
	gKBar  = gNaBar + nActiveComp;
	gNaChannel  = gKBar + nActiveComp;
	gKChannel  = gNaChannel + nActiveComp;

	/**
	 * TODO: Check if is when curr = 0;
	 */
	for (int i=0; i<nActiveComp; i++) {
		n[i] = 0.3177;
		h[i] = 0.5960;
		m[i] = 0.0529;
	}
}

ActiveChannels::~ActiveChannels() {
	delete[] memory;
	//delete[] activeCompList;
}

void ActiveChannels::evaluateCurrents(	ftype *Rm, ftype *active ) {

	if (this->channelInfo != 0)
		return evaluateCurrentsNew(Rm, active); // TODO:

	evaluateGates();

	/**
	 * Update the channel conductances
	 */
	for (int i=0; i<nActiveComp; i++) {

		gNaChannel[i] = gNaBar[i] * m[i] * m[i] * m[i] * h[i];
		gKChannel[i]  =  gKBar[i] * n[i] * n[i] * n[i] * n[i];

		unsigned int comp = activeCompList[i];
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

	for (int i=0; i<nActiveComp; i++) {
		V = vmList[ activeCompList[i] ];

		//#define EXPONENTIAL 0   // A exp((v-V0)/B)
		//#define SIGMOID 1 		// A / (exp((v-V0)/B) + 1)
		//#define LINOID 2        // A (v-V0) / (exp((v-V0)/B) - 1)

		// gate m
		alpha = (V != 25.0) ? (0.1 * (25 - V)) / ( expf( 0.1 * (25-V) ) - 1 ) : 1; // LINOID: 0.1; A=-0.1, B=-10, V0=25
		beta  = 4 * expf ( -V/18 ); // EXPONENTIAL: A = 4, B=-18, V0=0
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	m[i] = a + b * m[i];

		// gate h
		alpha =  0.07 * expf ( -V/20 ); // EXPONENTIAL: A = 0.07, B=-20, V0=0
		beta  = 1 / ( expf( (30-V)/10 ) + 1 ); // SIGMOID: A = 1, B=-10, V0=30
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	h[i] = a + b * h[i];

	 	// gate n
		alpha = (V != 10.0) ? (0.01 * (10 - V)) / ( expf( 0.1 * (10-V) ) - 1 ) : 0.1; // LINOID: 0.1; A=-0.01, B=-10, V0=10
		beta  = 0.125 * expf ( -V/80 );	// EXPONENTIAL: A = 0.125, B=-80, V0=0
		a = alpha / (1/dt + (alpha + beta)/2);
		b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);
	 	n[i] = a + b * n[i];
	}
}



void ActiveChannels::evaluateCurrentsNew( ftype *Rm, ftype *active ) {

	evaluateGatesNew();

	for (int i=0; i<nActiveComp; i++)
		gActive[i] = 0;

	/**
	 * Update the channel conductances
	 */
	int pos = 0;
	int actCompPos = -1;
	int lastActComp = -1;

	for (int ch=0; ch<nChannels; ch++) {

		int nGates     = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		int comp       = channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP];
		ftype gChannel = channelGbar[ch];

		//printf("gChannel1=%f\n", gChannel);

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

			//printf("%f\n", activeChannels->getSomaCurrents());
			//printf("%d|%-15.4f %d|%d %10.4f\n", pos, gateState[pos], ch, gt, vmList[comp]);

		}

		//printf("gChannel2=%f\n", gChannel);

		active[ comp ] -= gChannel * channelEk[ch] ;

		if (comp != lastActComp) {
			actCompPos++;
			lastActComp = comp;
			//assert(activeCompList[actCompPos] == comp);
		}
		gActive[ actCompPos ] += gChannel;

	}

	for (int i=0; i<nActiveComp; i++) {
		unsigned int comp = activeCompList[i];
		active[ comp ] -=  ( 1 / Rm[comp] ) * ( eLeak[i] );
	}
}

/**
 * Find the gate openings in the next time step
 * m(t + dt) = a + b m(t - dt)
 */
void ActiveChannels::evaluateGatesNew(  ) {

	ftype alpha, beta, a, b;

	ftype* gate = gatePar;

	int pos=0;
	for (int ch=0; ch<nChannels; ch++) {

		int nGates = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		ftype V = vmList[ channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP] ];

		for (int gt=0; gt < nGates; gt++, pos++) {

            // (EXPONENTIAL): alpha(v) = A exp((v-V0)/B)
            // (SIGMOID):     alpha(v) = A / (exp((v-V0)/B) + 1)
            // (LINOID):      alpha(v) = A (v-V0) / (exp((v-V0)/B) - 1)

			// alpha_function
			ftype v0 = gate[A_V0];
			switch( gateInfo[pos * N_GATE_FIELDS + ALPHA_FUNCTION] ) {
			case EXPONENTIAL:
				alpha = gate[A_A] * expf((V-v0)/gate[A_B]);
				break;
			case SIGMOID:
				alpha = gate[A_A] / ( expf( (V-v0)/gate[A_B] ) + 1);
				break;
			case LINOID:
				alpha = (V != v0) ? gate[A_A] * (V-v0) / (expf((V-v0)/gate[A_B]) - 1) : gate[A_A] * gate[A_B];
				break;
			default:
				printf("Active channels parameters are invalid. Exiting...\n");
				exit(-1);
			}
			//gate += N_GATE_FUNC_PAR;

			// beta_function
			v0 = gate[B_V0];
			switch( gateInfo[pos * N_GATE_FIELDS + BETA_FUNCTION] ) {
			case EXPONENTIAL:
				beta = gate[B_A] * expf((V-v0)/gate[B_B]);
				break;
			case SIGMOID:
				beta = gate[B_A] / ( expf( (V-v0)/gate[B_B] ) + 1);
				break;
			case LINOID:
				beta = (V != v0) ? gate[B_A] * (V-v0) / (expf((V-v0)/gate[B_B]) - 1) : gate[B_A] * gate[B_B];
				break;
			default:
				printf("Active channels parameters are invalid. Exiting...\n");
				exit(-1);
			}

			gate += N_GATE_FUNC_PAR;

			a = alpha / (1/dt + (alpha + beta)/2);
			b = (1/dt - (alpha + beta)/2) / (1/dt + (alpha + beta)/2);

			gateState[pos] = a + b * gateState[pos];
		}

	}
}

void ActiveChannels::createChannelList (int nChannels_, ucomp *nGates, ucomp *comp, ftype *chEk, ftype *gBar, ftype *eLeak_, int nActiveComp_, ucomp *activeCompList_) {

	if (nChannels_ <= 0) return;

	this->nGatesTotal = 0;
	for (int i=0; i<nChannels_; i++)
		this->nGatesTotal += nGates[i];

	this->nChannels   = nChannels_;
	this->nActiveComp = nActiveComp_;

	ucompMemSize = nChannels * N_CHANNEL_FIELDS + nGatesTotal * N_GATE_FIELDS + nActiveComp;
	this->ucompMem    = new ucomp[ucompMemSize];
	this->activeCompList = this->ucompMem;
	this->channelInfo    = this->activeCompList + nActiveComp;
	this->gateInfo       = this->channelInfo + (nChannels * N_CHANNEL_FIELDS);

	this->channelInfo[CH_NGATES]  = nGates[0];
	this->channelInfo[CH_COMP]    = comp[0];
	this->channelInfo[CH_GATEPOS] = 0;
	for (int ch=1; ch<nChannels_; ch++) {
		this->channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES] = nGates[ch];
		this->channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP] = comp[ch];
		this->channelInfo[ch*N_CHANNEL_FIELDS + CH_GATEPOS] =
				this->channelInfo[(ch-1)*N_CHANNEL_FIELDS + CH_GATEPOS] + nGates[ch-1];
	}

	ftypeMemSize = nChannels_*2 + nGatesTotal * (N_GATE_FUNC_PAR + 1) + nActiveComp*2;
	this->ftypeMem    = new ftype[ftypeMemSize];
	this->channelEk   = this->ftypeMem;
	this->channelGbar = this->channelEk   + nChannels_;
	this->eLeak       = this->channelGbar + nChannels_;
	this->gActive	  = this->eLeak       + nActiveComp;
	this->gateState   = this->gActive     + nActiveComp;
	this->gatePar     = this->gateState   + nGatesTotal;




	for (int ch=0; ch<nChannels_; ch++) {
		this->channelEk[ch]   = chEk[ch];
		this->channelGbar[ch] = gBar[ch];
	}

	for (int i=0; i<nActiveComp; i++) {
		this->eLeak[i] = eLeak_[i];
		this->activeCompList[i] = activeCompList_[i];
		this->gActive[i] = 0;
	}


}

void ActiveChannels::setGate (int channel, int gate, ftype state, ucomp gatePower,
		ucomp alpha, ftype alphaA, ftype alphaB, ftype alphaV0,
		ucomp beta, ftype betaA, ftype betaB, ftype betaV0) {

	int pos = this->channelInfo[channel*N_CHANNEL_FIELDS + CH_GATEPOS] + gate;

	//printf("%d %d %d\n", channel, gate, pos);

	this->gateInfo[pos * N_GATE_FIELDS + GATE_POWER]     = gatePower;
	this->gateInfo[pos * N_GATE_FIELDS + ALPHA_FUNCTION] = alpha;
	this->gateInfo[pos * N_GATE_FIELDS + BETA_FUNCTION]  = beta;

	this->gatePar[pos * N_GATE_FUNC_PAR + A_A] = alphaA;
	this->gatePar[pos * N_GATE_FUNC_PAR + A_B] = alphaB;
	this->gatePar[pos * N_GATE_FUNC_PAR + A_V0] = alphaV0;

	this->gatePar[pos * N_GATE_FUNC_PAR + B_A] = betaA;
	this->gatePar[pos * N_GATE_FUNC_PAR + B_B] = betaB;
	this->gatePar[pos * N_GATE_FUNC_PAR + B_V0] = betaV0;

	this->gateState[pos] = state;

}

