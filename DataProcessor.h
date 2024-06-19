#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <vector>

class DataProcessor {
public:
    DataProcessor();
    DataProcessor(const DataProcessor& orig);
    virtual ~DataProcessor();
    
    std::vector<double> process(std::vector<double> vec, int vecType);
    double out_magnitude;
    
private:
    

};

#endif

