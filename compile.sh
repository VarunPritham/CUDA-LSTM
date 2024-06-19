g++ -std=c++11 -c FileProcessor.cpp -o FileProcessor.o
g++ -std=c++11 -c DataProcessor.cpp -o DataProcessor.o
nvcc -arch=sm_75 -O3 -std=c++11 -dc LSTM.cu -o LSTM.o
nvcc -arch=sm_75 -O3 -std=c++11 -dc main.cu -o main.o
nvcc -arch=sm_75 -o lstm_exec FileProcessor.o DataProcessor.o LSTM.o main.o