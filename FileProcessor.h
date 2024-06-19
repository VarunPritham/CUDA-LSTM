#ifndef FILEPROCESSOR_H
#define FILEPROCESSOR_H

#include <iostream>
#include <fstream>
#include <string>

class FileProcessor {
public:
    FileProcessor();
    FileProcessor(const FileProcessor& orig);
    virtual ~FileProcessor();
    
    std::vector<double> read(std::string fileName, int valuesPerLine);
    std::vector<double> * readMultivariate(std::string fileName, int lines, int variables, int * inputCols, int targetValCol);
    int write(std::string fileName);
    int append(std::string line);
    
private:
    std::ofstream out_file;

};

#endif

