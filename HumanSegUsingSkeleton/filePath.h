#pragma once

#include <string>

//////////////////////////////////////////////////////////////////////////
//set the file path
static std::string _filepath = "..\\Dataset\\";
static std::string _dataName = "Data";
//////////////////////////////////////////////////////////////////////////

class filePath{
private:
	filePath(){}
	static filePath instance_;

public:
	static filePath& getInstance();	
	const std::string getColorPath(int fileNumber, int number);
	const std::string getBodyIndexPath(int fileNumber, int number);
	const std::string getMappPath(int fileNumber, int number);
	const std::string getDepthPath(int fileNumber, int number);
	const std::string getHRDepthPath(int fileNumber, int number);
	//const char* getSkeletonPath(int fileNumber);
	const std::string getSkeletonPath(int fileNumber);
	const std::string getResultPath(int fileNumber);
	const std::string getGroundPath(int fileNumber, int frameNumber);
};
