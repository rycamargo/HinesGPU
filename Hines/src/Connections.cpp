/*
 * Connections.cpp
 *
 *  Created on: 10/08/2009
 *      Author: rcamargo
 */

#include "Connections.hpp"
//#include "SynapticChannels.hpp"
#include <cstdlib>
#include <cstdio>
#include <cassert>

Connections::Connections() {
}

Connections::~Connections() {
	// TODO: Remove connections
}

std::vector<Conn> & Connections::getConnArray (int source) {
	return connMap[source];
}

void Connections::clearMPIConnections (ConnectionInfo *connInfo) {

	delete []connInfo->source;
	delete []connInfo->dest;
	delete []connInfo->synapse;
	delete []connInfo->weigth;
	delete []connInfo->delay;

	delete connInfo;
}

ConnectionInfo *Connections::getConnectionInfo () {

	int countTotal = 0;
	ConnectionInfo *connInfo = new ConnectionInfo;

	// Counts the total number of connections
	map< int, std::vector<Conn> >::iterator p;
	for(p = connMap.begin(); p != connMap.end(); p++)
		countTotal += p->second.size();

	connInfo->nConnections = countTotal;
	connInfo->source  = new int[countTotal];
	connInfo->dest 	 = new int[countTotal];
	connInfo->synapse = new ucomp[countTotal];
	connInfo->weigth  = new ftype[countTotal];
	connInfo->delay   = new ftype[countTotal];

	int infoPos = 0;
	for(p = connMap.begin(); p != connMap.end(); p++) {

		int source = p->first;
		std::vector<Conn> & conn = p->second;

		std::vector<Conn>::iterator p;
		for(p = conn.begin(); p != conn.end(); p++) {
			connInfo->source [infoPos] = source;
			connInfo->dest   [infoPos] = (*p).dest;
			connInfo->synapse[infoPos] = (*p).synapse;
			connInfo->weigth [infoPos] = (*p).weigth;
			connInfo->delay  [infoPos] = (*p).delay;
			infoPos++;
		}
	}

	return connInfo;
}

int Connections::createTestConnections () {

	int source;

	Conn conn1;
	source = CONN_NEURON_TYPE*0 + 0;
	conn1.dest   = CONN_NEURON_TYPE*3 + 1;
	conn1.synapse = 0;
	conn1.weigth = 1;
	conn1.delay = 10;
	connMap[source].push_back(conn1);

	Conn conn2;
	source = CONN_NEURON_TYPE*0 + 0;
	conn2.dest   = CONN_NEURON_TYPE*1 + 1;
	conn2.synapse = 0;
	conn2.weigth = 1;
	conn2.delay = 10;
	connMap[source].push_back(conn2);

	Conn conn3;
	source = CONN_NEURON_TYPE*3 + 1;
	conn3.dest   = CONN_NEURON_TYPE*1 + 0;
	conn3.synapse = 1;
	conn3.weigth = 1;
	conn3.delay = 10;
	connMap[source].push_back(conn3);

	return 0;
}

/**
 * type [0][1][2][3] pyramidal
 * type [4][5][6][7] inhibitory
 */
int Connections::connectAssociativeFromFile (char *filename) {
	return 0;
}

void generateRandomList(int nNumbers, int *randomNumberList, random_data *randBuf) {
	for ( int i=0; i<nNumbers; i++ )
		random_r( randBuf, &randomNumberList[i] );
}

int Connections::connectTypeToTypeRandom( ThreadInfo *tInfo,
		int sourceType, int destType, ucomp synapse, ftype connRatio,
		ftype baseW, ftype randW, ftype baseD, ftype randD) {

	int nConnTotal = 0;

	int nDestTypeNeurons = 0;
	for (int type=0; type < tInfo->totalTypes; type++)
		if (tInfo->sharedData->typeList[type] == destType)
			nDestTypeNeurons += tInfo->nNeurons[type];

	int randomListPos = 0;
	int nRandom = nDestTypeNeurons * nDestTypeNeurons * connRatio * 1.2;
	int *randomNumberList = (int *)malloc( sizeof(int) * nRandom);
	random_data *randBuff = tInfo->sharedData->randBuf[tInfo->threadNumber];
	generateRandomList(nRandom, randomNumberList, randBuff);

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	int *randomConnectionList = (int *)malloc( sizeof(int) * (nDestTypeNeurons + 1) );

	for (int type=tInfo->startTypeProcess; type < tInfo->endTypeProcess; type++) {

		if (tInfo->sharedData->typeList[type] != sourceType) continue;

		for (int sNeuron=0; sNeuron < tInfo->nNeurons[type]; sNeuron++) {

			for (int i =0; i < nDestTypeNeurons; i++ )
				randomConnectionList[i] = 0;
			randomConnectionList[nDestTypeNeurons] = 1; // used only for control

			/**
			 * Connects to a random neuron nDestTypeNeurons*connRatio times
			 */
			for (int c=0; c < nDestTypeNeurons * connRatio; c++) {

				Conn conn1;
				conn1.synapse = synapse;

				int32_t wei;
				random_r( randBuff, &wei );
				conn1.weigth = baseW * (randW + ((ftype)wei)/RAND_MAX); // randW = 0.5

				int32_t del;
				random_r( randBuff, &del );
				conn1.delay = baseD + randD*((ftype)del)/RAND_MAX; // baseD = 10, randD = 10

				// Finds a target that did not receive any connection
				conn1.dest = nDestTypeNeurons;
				while(randomConnectionList[conn1.dest] != 0) {
					conn1.dest = randomNumberList[randomListPos++] % nDestTypeNeurons;
					if (randomListPos == nRandom) {
						generateRandomList(nRandom, randomNumberList, randBuff);
						randomListPos = 0;
					}
				}
				randomConnectionList[conn1.dest] = 1;

				assert (0 <= conn1.dest && conn1.dest < nDestTypeNeurons);

				// Transform the connection to the type CONN_NEURON_TYPE*dType + neuron
				int count = 0;
				for (int dType=0; dType < tInfo->totalTypes; dType++) {

					if (tInfo->sharedData->typeList[dType] != destType) continue;

					if (conn1.dest < count + tInfo->nNeurons[dType]) {
						conn1.dest = CONN_NEURON_TYPE*dType + (conn1.dest-count);
						break;
					}
					else
						count += tInfo->nNeurons[dType];
				}

				connMap[CONN_NEURON_TYPE*type + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}
	free (randomConnectionList);
	free(randomNumberList);

	return nConnTotal;
}

int Connections::connectTypeToTypeOneToOne( ThreadInfo *tInfo,
		int sourceType, int destType, ucomp synapse, ftype baseW, ftype baseD) {

	int nConnTotal = 0;

	// Checks the number of inhibitory neurons in the previous processes
	int pyrNeuron = 0;
	for (int sType=0; sType < tInfo->startTypeProcess; sType++)
		if (tInfo->sharedData->typeList[sType] == INHIBITORY_CELL)
			pyrNeuron += tInfo->nNeurons[sType];

	if (tInfo->sharedData->inhConnRatio > 0) {
		for (int sType=tInfo->startTypeProcess; sType < tInfo->endTypeProcess; sType++) {

			if (tInfo->sharedData->typeList[sType] == PYRAMIDAL_CELL) continue;

			for (int sNeuron=0; sNeuron < tInfo->nNeurons[sType]; sNeuron++, pyrNeuron++) {

				Conn conn1;
				conn1.synapse = synapse; // inhibitory
				conn1.weigth = baseW;
				conn1.delay  = baseD;

				conn1.dest = pyrNeuron;
				int count = 0;
				for (int dType=0; dType < tInfo->totalTypes; dType++) {

					if (tInfo->sharedData->typeList[dType] == INHIBITORY_CELL) continue;

					if (conn1.dest < count + tInfo->nNeurons[dType]) {
						conn1.dest = CONN_NEURON_TYPE*dType + (conn1.dest-count);
						break;
					}
					else
						count += tInfo->nNeurons[dType];
				}


				connMap[CONN_NEURON_TYPE*sType + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}

	return nConnTotal;
}


int Connections::connectRandom ( ThreadInfo *tInfo ) {

	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	int nConnTotal = 0;

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, PYRAMIDAL_CELL, 0, sharedData->pyrConnRatio,
			sharedData->excWeight, 0.5, 10, 10);

	/**
	 * Connects the pyramidal-inhibitory cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, INHIBITORY_CELL, 0, sharedData->inhConnRatio/10,
			sharedData->pyrInhWeight*4, 0.5, 10, 10);

//	nConnTotal += connectTypeToTypeRandom(
//			tInfo, PYRAMIDAL_CELL, INHIBITORY_CELL, 0, sharedData->inhConnRatio,
//			sharedData->pyrInhWeight, 0.5, 10, 10);

	/**
	 * Connects the inhibitory-pyramidal cells
	 * Each inhibitory cell connects to a single pyramidal neuron
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, INHIBITORY_CELL, PYRAMIDAL_CELL, 1, sharedData->inhConnRatio/10,
			sharedData->inhPyrWeight/10, 0.5, 10, 10);

//	nConnTotal += connectTypeToTypeOneToOne(
//			tInfo, INHIBITORY_CELL, PYRAMIDAL_CELL, 1,
//			tInfo->sharedData->inhPyrWeight, 10);

	printf("Total number of connections = %d.\n", nConnTotal);

	return 0;
}

