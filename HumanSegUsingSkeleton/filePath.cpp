#include "filePath.h"

filePath filePath::instance_;

filePath& filePath::getInstance(){	
	return instance_;
}

const std::string filePath::getColorPath(int fileNumber, int number){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Color\\") + std::string("KinectScreenshot_RGB") + std::to_string(number) + std::string(".bmp");
}

const std::string filePath::getBodyIndexPath(int fileNumber, int number){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\HR_BodyIndex\\") + std::string("FileHRbodyIndex_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getMappPath(int fileNumber, int number){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Mapp\\") + std::string("FileMapp_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getDepthPath(int fileNumber, int number){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Depth\\") + std::string("Filedepth_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getHRDepthPath(int fileNumber, int number){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\HR_Depth\\") + std::string("FileHRdepth_") + std::to_string(number) + std::string(".bin");
}

//const char* FilePath::getSkeletonPath(int fileNumber){
const std::string filePath::getSkeletonPath(int fileNumber){
	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Body\\Fileskeleton.csv");
}

const std::string filePath::getResultPath(int fileNumber){

	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Result\\");
}

const std::string filePath::getGroundPath(int fileNumber, int frameNumber){
	return _filepath + _dataName + std::to_string(fileNumber) + std::string("\\Color\\") + std::string("Ground") + std::to_string(frameNumber) + std::string(".bmp");
}



/*
const std::string filePath::getColorPath(int number){
	//std::string fileName = "KinectScreenshot_RGB";
	
	//std::string path = filepath + dataName + "\\Color\\" + fileName + std::to_string(number) + ".bmp";
	
	//return path;
	return filepath + dataName + std::string("\\Color\\") + std::string("KinectScreenshot_RGB") + std::to_string(number) + std::string(".bmp");
}

const std::string filePath::getBodyIndexPath(int number){
	//std::string fileName = "FileHRbodyIndex_";

	//std::string path = filepath + dataName + "\\HR_BodyIndex\\" + fileName + std::to_string(number) + ".bin";

	//return path;
	return filepath + dataName + std::string("\\HR_BodyIndex\\") + std::string("FileHRbodyIndex_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getMappPath(int number){
	//std::string fileName = "FileMapp_";

	//std::string path = filepath + dataName + "\\Mapp\\" + fileName + std::to_string(number) + ".bin";

	//return path;
	return filepath + dataName + std::string("\\Mapp\\") + std::string("FileMapp_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getDepthPath(int number){
	//std::string fileName = "Filedepth_";

	//std::string path = filepath + dataName + "\\Depth\\" + fileName + std::to_string(number) + ".bin";

	//return path;
	return filepath + dataName + std::string("\\Depth\\") + std::string("Filedepth_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getHRDepthPath(int number){

	return filepath + dataName + std::string("\\HR_Depth\\") + std::string("FileHRdepth_") + std::to_string(number) + std::string(".bin");
}

const std::string filePath::getSkeletonPath(){
	//std::string fileName = "Fileskeleton";

	//std::string path = filepath + dataName + "\\Body\\" + fileName + ".csv";

	//return _strdup(path.c_str());	

	return filepath + dataName + std::string("\\Body\\") + std::string("Fileskeleton.csv");
}

const std::string filePath::getResultPath(){
	//std::string path = filepath + dataName + "\\Result\\";

	//return path;
	return filepath + dataName + std::string("\\Result\\");
}

const std::string filePath::getGroundPath(){
	return filepath + dataName + std::string("\\Color\\") + std::string("Ground") + std::string(".bmp");
}

*/