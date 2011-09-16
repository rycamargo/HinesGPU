g++ -c -O3 SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp
echo ==================== [RYC] Finished compilation of cpp files! ====================
sleep 1
nvcc $1 --compiler-options -O3 --ptxas-options=-v -m64 -arch sm_20 -o HinesGpu launchHines.o ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o HinesGpu.cu PrepareGpuExecution.cu SynapticComm.cu

