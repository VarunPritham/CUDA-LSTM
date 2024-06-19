#include "DataProcessor.h"

#include <cmath>

DataProcessor::DataProcessor() {
    out_magnitude = 0.0;
}

DataProcessor::DataProcessor(const DataProcessor& orig) { }

DataProcessor::~DataProcessor() { }

std::vector<double> DataProcessor::process(std::vector<double> vec, int vecType) {
    
    double magnitude = 0.0;
    for(std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
        magnitude += std::pow(*it,2);
    }
    magnitude = std::pow(magnitude,0.5);
    
    if (magnitude != 0) {
        for(std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
            *it /= magnitude;
        }
    }
    
    // if target vector
    if (vecType == 1) out_magnitude = magnitude;
    
    return vec;
}

