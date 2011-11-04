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

int Connections::connectRandom ( ThreadInfo *tInfo ) {

	SharedNeuronGpuData *sharedData = tInfo->sharedData;
	int *typeList		= sharedData->typeList;
	int *nNeurons 		= tInfo->nNeurons;
	int threadNumber 	= tInfo->threadNumber;

	int nConnTotal = 0;
	int nPyramidal  = 0;
	int nInhibitory = 0;
	for (int type=0; type < tInfo->totalTypes; type++) {
		if (typeList[type] == PYRAMIDAL_CELL)
			nPyramidal += nNeurons[type];
		else if (typeList[type] == INHIBITORY_CELL)
			nInhibitory += nNeurons[type];
	}

	int randomListPos=0;
	int nRandom = 10000; //nPyramidal * nPyramidal * pyrRatio;
	int *randomNumberList = (int *)malloc( sizeof(int) * nRandom);
	generateRandomList(nRandom, randomNumberList, sharedData->randBuf[threadNumber]);

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	int *randomConnectionList = (int *)malloc( sizeof(int) * (nPyramidal + 1) );

	for (int sType=tInfo->startTypeProcess; sType < tInfo->endTypeProcess; sType++) {

		if (typeList[sType] == INHIBITORY_CELL) continue;


		for (int sNeuron=0; sNeuron < nNeurons[sType]; sNeuron++) {

			/**
			 * Connects to a random neuron k times
			 */
			for (int i =0; i < nPyramidal; i++ )
				randomConnectionList[i] = 0;
			randomConnectionList[nPyramidal] = 1; // used only for control

			for (int c=0; c < nPyramidal * sharedData->pyrConnRatio; c++) {

				Conn conn1;
				conn1.synapse = 0;

				int32_t wei;
				random_r( sharedData->randBuf[threadNumber], &wei );
				conn1.weigth = sharedData->excWeight * (0.5 + ((ftype)wei)/RAND_MAX);
				//conn1->weigth = sharedData->excWeight;

				int32_t del;
				random_r( sharedData->randBuf[threadNumber], &del );
				conn1.delay = 10 + 10.*((ftype)del)/RAND_MAX;
				//conn1->delay = 10;

				conn1.dest = nPyramidal;
				while(randomConnectionList[conn1.dest] != 0) {
					conn1.dest = randomNumberList[randomListPos++] % nPyramidal;
					if (randomListPos == nRandom) {
						generateRandomList(nRandom, randomNumberList, sharedData->randBuf[threadNumber]);
						randomListPos = 0;
					}
				}
				randomConnectionList[conn1.dest] = 1;

				assert (0 <= conn1.dest && conn1.dest < nPyramidal);


				int count = 0;
				for (int dType=0; dType < tInfo->totalTypes; dType++) {

					if (typeList[dType] == INHIBITORY_CELL) continue;

					if (conn1.dest < count + nNeurons[dType]) {
						conn1.dest = CONN_NEURON_TYPE*dType + (conn1.dest-count);
						break;
					}
					else
						count += nNeurons[dType];
				}

				connMap[CONN_NEURON_TYPE*sType + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}
	free (randomConnectionList);

	/**
	 * Connects the pyramidal-inhibitory cells
	 */
	randomConnectionList = (int *)malloc( sizeof(int) * (nInhibitory + 1) );
	for (int sType=tInfo->startTypeProcess; sType < tInfo->endTypeProcess; sType++) {

		if (typeList[sType] == INHIBITORY_CELL) continue;

		for (int sNeuron=0; sNeuron < nNeurons[sType]; sNeuron++) {

			/**
			 * Connects to a random neuron k times
			 */
			for (int i =0; i < nInhibitory; i++ )
				randomConnectionList[i] = 0;
			randomConnectionList[nInhibitory] = 1; // used only for control

			for (int c=0; c < nInhibitory * sharedData->inhConnRatio; c++) {

				Conn conn1;
				conn1.synapse = 0;

				int32_t wei;
				random_r( sharedData->randBuf[threadNumber], &wei );
				conn1.weigth = sharedData->pyrInhWeight * (0.5 + ((ftype)wei)/RAND_MAX);
				//conn1->weigth = sharedData->pyrInhWeight;

				int32_t del;
				random_r( sharedData->randBuf[threadNumber], &del );
				conn1.delay = 10 + ((ftype)del)/RAND_MAX;
				//conn1->delay = 10;

				conn1.dest = nInhibitory;
				while(randomConnectionList[conn1.dest] != 0) {
					conn1.dest = randomNumberList[randomListPos++] % nInhibitory;
					if (randomListPos == nRandom) {
						generateRandomList(nRandom, randomNumberList, sharedData->randBuf[threadNumber]);
						randomListPos = 0;
					}
				}
				randomConnectionList[conn1.dest] = 1;

				assert (0 <= conn1.dest && conn1.dest < nInhibitory);

				//------------
//				int32_t pos;
//				random_r( sharedData->randBuf[threadNumber], &pos );
//				conn1.dest = pos % nInhibitory;
				//------------

				int count = 0;
				for (int dType=0; dType < tInfo->totalTypes; dType++) {

					if (typeList[dType] == PYRAMIDAL_CELL) continue;

					if (conn1.dest < count + nNeurons[dType] ) {
						conn1.dest = CONN_NEURON_TYPE*dType + (0.5 + (conn1.dest-count));
						break;
					}
					else
						count += nNeurons[dType];
				}


				connMap[CONN_NEURON_TYPE*sType + sNeuron].push_back(conn1);
				nConnTotal++;
			}

		}
	}
	free (randomConnectionList);

	/**
	 * Connects the inhibitory-pyramidal cells
	 * Each pyramidal cell connects to a single inhibitory neuron
	 */
	// Checks the number of inhibitory neurons in the previous processes
	int inhNeuron = 0;
	for (int sType=0; sType < tInfo->startTypeProcess; sType++)
		if (typeList[sType] == INHIBITORY_CELL)
			inhNeuron += nNeurons[sType];

	if (sharedData->inhConnRatio > 0) {
		for (int sType=tInfo->startTypeProcess; sType < tInfo->endTypeProcess; sType++) {

			if (typeList[sType] == PYRAMIDAL_CELL) continue;

			for (int sNeuron=0; sNeuron < nNeurons[sType]; sNeuron++, inhNeuron++) {

				Conn conn1;
				conn1.synapse = 1; // inhibitory

				//int32_t wei;
				//random_r( sharedData->randBuf[threadNumber], &wei );
				//conn1->weigth = sharedData->inhPyrWeight * ((ftype)wei)/RAND_MAX;
				conn1.weigth = sharedData->inhPyrWeight;

				int32_t del;
				random_r( sharedData->randBuf[threadNumber], &del );
				conn1.delay = 10 + ((ftype)del)/RAND_MAX;
				//conn1->delay = 10;

				conn1.dest = inhNeuron;
				int count = 0;
				for (int dType=0; dType < tInfo->totalTypes; dType++) {

					if (typeList[dType] == INHIBITORY_CELL) continue;

					if (conn1.dest < count + nNeurons[dType]) {
						conn1.dest = CONN_NEURON_TYPE*dType + (conn1.dest-count);
						break;
					}
					else
						count += nNeurons[dType];
				}


				connMap[CONN_NEURON_TYPE*sType + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}

	free(randomNumberList);
	printf("Total number of connections = %d.\n", nConnTotal);

	return 0;
}

