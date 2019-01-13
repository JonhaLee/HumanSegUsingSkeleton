#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "BinaryStream.h"

#include "gc.h"
#include "filePath.h"

// For the testing with time check
#include <time.h>
double gseg_total = 0, prop_total = 0;
int gseg_cnt = 0, prop_cnt = 0;


#ifdef _DEBUG
// For the checking memory leak in DEBUG
//#include <vld.h>
#endif

using namespace cv;
using namespace std;

// Const vars
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define PIE 3.1419

const static int kTotal_BodyIndex = 6;
const static int kJointFromKinectV2 = 25;

bool isBody[kTotal_BodyIndex];

// File load methods
void loadBodyIndexFile(BYTE*& bodyIndexData, int fileNumber, int frameNumber);
void loadMappingFile(short*& mappingDats, int fileNumber, int frameNumber);
void loadDepthImageFile(BYTE*& depthData, int fileNumber, int frameNumber);
void loadHRDepthImageFile(cv::Mat& depthData, int fileNumber, int frameNumber);
void loadSkeletonFile(cv::Vec4i**& skeletonDatas_origin, int fileNumber, int frameNumber);


// Input the seed
void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData);
void addSeedBySkeleton(Mat* mask, BYTE** skeletonData);

void createDistanceMaps(Mat*& outputInput, BYTE** skeletonDataMaps);
vector<vector<Point2i>> getSkeletonLinesAround(int jointNumber, cv::Vec4i* skeletonJointPosition);

// For making sekeleton datas
void createSkeletonData(BYTE**& skeletonDatasMap, BYTE**& skeletonDatasMap2, int fileNumber, int frameNumber, cv::Vec4i** skeletonDatas_origin, cv::Vec4i** skeletonDatas_hd);
void createSkeletonWeightMap(double**& weightMaps, int frame, cv::Vec4i** skeletonDatas_hd, Mat* distanceMaps);
void CreateSkeletonLines(cv::Vec4i* jointPoints, int width, int height, int stroke, BYTE* skeletonMap, BYTE* skeletonMap2);


// Utils
cv::Vec4i mappingLowToHigh(cv::Vec4i point, short* mappingData);
void expandPixelBy(int x, int y, int width, int height, int stroke, BYTE* skeletonDatas);
void drawLineBy(cv::Vec4i start, cv::Vec4i end, int width, int height, int stroke, BYTE* skeletonDatas, BYTE* skeletonDatas2);
vector<Point2i> getLineFromTwoPoint(cv::Vec4i point1, cv::Vec4i point2);
vector<Point2i> getLineFromTwoPoint2(cv::Vec4i point1, cv::Vec4i point2);

const char* getfield(char* line, int num);
int getL1DistanceTwoPoint(Point2i left, Point2i right);
double getEuclideanDistanceTwoPoint(Point2i left, Point2i right);

int getCloestJointNumber(Point2i point, cv::Vec4i* skeletonJointPosition);

double inferred_sigma = 80;
double head_sigma = 50;
double foot_sigma = 50;
double sigma = 10;
const double a = 0.00173667 * 2;

int main(){
	
	/**/std::printf("OpenCV Version : %s\n\n", CV_VERSION);

	int fileNumber = 10;	//set the data number, in here 10
	int frameNumber = 0;

	const int kTotal_Joint = 25;
	int isExistFrame = 0;

	for ( ; fileNumber <= 10; fileNumber++){

		printf("%d\n", fileNumber);

		cv::MatND hist_pre[kTotal_BodyIndex][kTotal_Joint];
		bool kalmanStart[6][25];
		for (int i = 0; i < kTotal_BodyIndex; i++){
			for (int j = 0; j < kTotal_Joint; j++)
				kalmanStart[i][j] = false;
		}

		const int n = 6; const int u = 3; const int nw = 6; const int m = 3; const int nv = 3;

		//Kinect v2 is 30FPS. so, dT is 1 / 30sec
		double dT = 1 / (double)30;
		

		for (int frameCount = 0; frameCount <= 20; frameCount++){				
			frameNumber = frameCount;
			
			for (int i = 0; i < kTotal_BodyIndex; i++){
				isBody[i] = false;
			}

			// For time check
			clock_t begin, end;

			// load body index data by providing kinect v2
			BYTE* bodyIndexData;			
			loadBodyIndexFile(bodyIndexData, fileNumber, frameNumber);
			
			// Create Seed mask for GSEG
			Mat* GSEG_Masks = new Mat[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				GSEG_Masks[i] = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(GC_BGD));
			}
			addSeedByBodyIndex(GSEG_Masks, bodyIndexData);

			// Create Seed mask for Prop
			Mat* Prop_Masks = new Mat[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				Prop_Masks[i] = GSEG_Masks[i].clone();
			}

			// Load Skeleton Data
			cv::Vec4i** skeletonDatas_origin = new Vec4i*[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				skeletonDatas_origin[i] = new Vec4i[kJointFromKinectV2];

				for (int j = 0; j < kJointFromKinectV2; j++){
					(skeletonDatas_origin[i][j])[0] = 0;
					(skeletonDatas_origin[i][j])[1] = 0;
					(skeletonDatas_origin[i][j])[2] = 0;
					(skeletonDatas_origin[i][j])[3] = 0;
				}
			}
			loadSkeletonFile(skeletonDatas_origin, fileNumber, frameNumber);

			// Skeleton Data which received from Kinect v2 has low resolution.
			// So, we must increase resolution of skeleton datas. 
			Vec4i** skeletonDatas_hd = new Vec4i*[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				skeletonDatas_hd[i] = new Vec4i[kJointFromKinectV2];
			}
			BYTE** skeletonDataMaps, **skeletonDataMaps2;
			
			// Increase resolution of Skeleton datas from origin to fhd
			createSkeletonData(skeletonDataMaps, skeletonDataMaps2, fileNumber, frameNumber, skeletonDatas_origin, skeletonDatas_hd);

			// Add seed data to Prop Seed mask.
			addSeedBySkeleton(Prop_Masks, skeletonDataMaps);
	
			// Make proposed distance maps.
			Mat* distanceMaps;
			createDistanceMaps(distanceMaps, skeletonDataMaps);

			// Create proposed weight map using skeleton datas.
			double** weightMaps;
			createSkeletonWeightMap(weightMaps, frameNumber, skeletonDatas_hd, distanceMaps);
			
			// Input Image
			Mat con_img = imread(filePath::getInstance().getColorPath(fileNumber, frameNumber));


			//GSEG, Only use Body Index data
#pragma region GSEG Method	
			
			cv::Mat gseg_results[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				gseg_results[i] = Mat(con_img.size(), CV_8UC3, cv::Scalar(255, 0, 0));
			}

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (isBody[i]){					
					/**/printf("%d st frame, %d st BodyIndex, GSEG algorithm start\n", frameNumber, i);
				
					Mat back, fore;
					Rect rect(10, 10, 100, 100);

					begin = clock();
					//Actually, this function perform like graph cut.
					grabCut(con_img,			// Input image
						GSEG_Masks[i],			// GSEG seed mask.
						rect,					// In here, we use GC_INIT_WITH_MASK. so, rect data not use.
						back, fore,				// Result of back/foreground data.
						1,						// Repeat count
						GC_INIT_WITH_MASK);		// MASK mode

					end = clock();


					gseg_total += (end - begin) / (double)CLOCKS_PER_SEC;					
					gseg_cnt++;
					/**/printf("%d st frame, %d st BodyIndex, GSEG algorithm finish. time = %f\n", frameNumber, i, (end - begin) / (double)CLOCKS_PER_SEC);

					Mat fgd_result, pr_fgd_result;					
					Mat final_result(con_img.size(), CV_8UC3, cv::Scalar(255, 0, 0));
					
					compare(GSEG_Masks[i], cv::GC_FGD, fgd_result, cv::CMP_EQ);
					compare(GSEG_Masks[i], GC_PR_FGD, pr_fgd_result, CMP_EQ);

					con_img.copyTo(gseg_results[i], fgd_result);
					con_img.copyTo(gseg_results[i], pr_fgd_result);
				}
			}
			
#pragma endregion

			// save gseg result
			cv::Vec3b blue;
			blue[0] = 255;
			blue[1] = 0;
			blue[2] = 0;
			cv::Mat result_GSEG(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, Scalar(255, 0, 0));

			for (int row = 0; row < IMAGE_HEIGHT; row++){
				for (int col = 0; col < IMAGE_WIDTH; col++){

					for (int i = 0; i < kTotal_BodyIndex; i++){
						if (isBody[i]){
							if (gseg_results[i].at<cv::Vec3b>(row, col) != blue){
								result_GSEG.at<cv::Vec3b>(row, col) = gseg_results[i].at<cv::Vec3b>(row, col);
							}
						}
					}
				}
			}
			cv::imwrite("Gseg_result.jpg", result_GSEG);



			cv::Mat prop_results[kTotal_BodyIndex];
			for (int i = 0; i < kTotal_BodyIndex; i++){
				prop_results[i] = Mat(con_img.size(), CV_8UC3, cv::Scalar(255, 0, 0));
			}

			//Proposed, we use BodyIndex data and Weight map from skeleton data.
#pragma region Proposed Method		

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (isBody[i]){
					/**/printf("%d st frame, %d st BodyIndex, Proposed algorithm start\n", frameNumber, i);
										
					my::GraphCut gc;
					Mat back, fore;
					Rect rect(10, 10, 100, 100);

					begin = clock();
					//Actually, this function perform like graph cut.
					gc.graphCut(con_img,		// Input image
						Prop_Masks[i],			// GSEG seed mask.
						rect,					// In here, we use GC_INIT_WITH_MASK. so, rect data not use.
						back, fore,				// Result of back/foreground data.
						1,						// Repeat count
						GC_INIT_WITH_MASK,		// MASK mode
						weightMaps[i]);			// Proposed weight maps

					end = clock();
					prop_total += (end - begin) / (double)CLOCKS_PER_SEC;
					prop_cnt++;
					/**/printf("%d st frame, %d st BodyIndex, Proposed algorithm finish. time = %f\n", frameNumber, i, (end - begin) / (double)CLOCKS_PER_SEC);

				
					Mat fgd_result, pr_fgd_result;
			
					compare(Prop_Masks[i], cv::GC_FGD, fgd_result, cv::CMP_EQ);					
					compare(Prop_Masks[i], GC_PR_FGD, pr_fgd_result, CMP_EQ);
					
					con_img.copyTo(prop_results[i], fgd_result);
					con_img.copyTo(prop_results[i], pr_fgd_result);

					back.refcount = 0;
					back.release();
					fore.refcount = 0;
					fore.release();
					fgd_result.refcount = 0;
					fgd_result.release();
					pr_fgd_result.refcount = 0;
					pr_fgd_result.release();
				}
			}

#pragma endregion
			// save proposed result
			cv::Mat result(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, Scalar(255, 0, 0));

			for (int row = 0; row < IMAGE_HEIGHT; row++){
				for (int col = 0; col < IMAGE_WIDTH; col++){

					for (int i = 0; i < kTotal_BodyIndex; i++){
						if (prop_results[i].at<cv::Vec3b>(row, col) != blue){
							result.at<cv::Vec3b>(row, col) = prop_results[i].at<cv::Vec3b>(row, col);
						}
					}
				}
			}
			cv::imwrite("prop_result.jpg", result);

		
			// Memory Dealloc
			con_img.refcount = 0;
			con_img.release();

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDataMaps[i] != NULL)
					delete[] skeletonDataMaps[i];
			}
			if (skeletonDataMaps != NULL)
				delete[] skeletonDataMaps;

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDataMaps2[i] != NULL)
					delete[] skeletonDataMaps2[i];
			}
			if (skeletonDataMaps2 != NULL)
				delete[] skeletonDataMaps2;		

			for (int i = 0; i < kTotal_BodyIndex; i++){
				delete[] weightMaps[i];
			}
			delete[] weightMaps;


			for (int i = 0; i < kTotal_BodyIndex; i++){
				GSEG_Masks[i].refcount = 0;
				GSEG_Masks[i].release();
				Prop_Masks[i].refcount = 0;
				Prop_Masks[i].release();
			}
			delete[] GSEG_Masks;
			delete[] Prop_Masks;

			for (int i = 0; i < kTotal_BodyIndex; i++){
				isBody[i] = false;
			}
		}
	}

	printf("gseg total time = %f\n", gseg_total / gseg_cnt);
	printf("prop total time = %f\n", prop_total / prop_cnt);

	return 0;
}

void loadBodyIndexFile(BYTE*& bodyIndexData, int fileNumber, int frameNumber){
	/*
	bodyIndex Data format : (1920 * 2) * 1080
	L1, R1, L2, R2, L3, R3, L4, R4 ... 
	L is read data value including 0 ~ 5 number that bodyIndex number by kinect
	R is accuracy. 0 is calibrationed value. 1 is real data.
	Because, we increased resolution of bodyIndex data from original to fhd
	*/
	BinaryReader br(filePath::getInstance().getBodyIndexPath(fileNumber, frameNumber));
	int cur_pos = 0;
	int file_length = (int)3840 * 1080;
	bodyIndexData = new BYTE[3840 * 1080];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		bodyIndexData[arr_index] = br.ReadBYTE();

		arr_index++;
		cur_pos += sizeof(BYTE);
	}
	br.CloseFile();	
}

void loadMappingFile(short*& mappingDats, int fileNumber, int frameNumber){
	
	BinaryReader br(filePath::getInstance().getMappPath(fileNumber, frameNumber));
	int cur_pos = 0;
	int file_length = (int)1024 * 424;

	mappingDats = new short[1024 * 424];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		mappingDats[arr_index] = br.ReadInt16();
		arr_index++;
		cur_pos++;
	}

	br.CloseFile();
}

void loadDepthImageFile(BYTE*& depthData, int fileNumber, int frameNumber){
	const ushort minDepth = 500;	
	const ushort maxDepth = 65535;	
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(filePath::getInstance().getDepthPath(fileNumber, frameNumber));
	int cur_pos = 0;
	int file_length = (int)512 * 424;

	depthData = new BYTE[512 * 424];

	int arr_index = 0;
	while (cur_pos < file_length)
	{		
		ushort depthValue = br.ReadInt16();		
		depthData[arr_index] = (depthValue >= minDepth && depthValue <= maxDepth ? (depthValue / MapDepthToByte) : 0);
		
		arr_index++;
		cur_pos++;
	}

	br.CloseFile();
}
void loadHRDepthImageFile(cv::Mat& depthData, int fileNumber, int frameNumber){
	
	const ushort minDepth = 500;	
	const ushort maxDepth = 65535;	
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(filePath::getInstance().getHRDepthPath(fileNumber, frameNumber));

	int cur_pos = 0;
	int file_length = (int)3840 * 1080 * 2;

	BYTE* tmp = new BYTE[3840 * 1080];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		ushort depthValue = br.ReadInt16();

		if (arr_index % 2 == 0)
			depthData.at<uchar>(arr_index / 2) = (depthValue >= minDepth && depthValue <= maxDepth ? (depthValue / MapDepthToByte) : 0);

		arr_index++;
		cur_pos += 2 * sizeof(BYTE);
	}

	delete[] tmp;
	br.CloseFile();
}

void loadSkeletonFile(Vec4i**& skeletonDatas_origin, int fileNumber, int frameNumber){
	/*	
	load to skeleton data from .csv file
	resolution 512 x 424 (depth)		
	*/

	FILE* stream;
	fopen_s(&stream, filePath::getInstance().getSkeletonPath(fileNumber).c_str(), "rb");

	char line[1024];
	const int currentPos = 150 * frameNumber;
	int jointsCount = 0;
	int bodyIndex = 0;

	//kinect can detect total 6 people at same time.
	for (int i = 0; i < kTotal_BodyIndex; i++){
		isBody[i] = false;
	}
	
	if (frameNumber != 0)
	{
		while (fgets(line, 1024, stream))
		{
			if (currentPos == jointsCount + 1) break;
			jointsCount++;
		}
	}
	jointsCount = 0;

	while (fgets(line, 1024, stream))
	{		
		if (jointsCount == kJointFromKinectV2){
			jointsCount = 0;
			
			bodyIndex++;
		}
		
		if (bodyIndex == 6) break;
		
		char* tmp_x = _strdup(line);
		int csv_x = atoi(getfield(tmp_x, 1));
		free(tmp_x);
		char* tmp_y = _strdup(line);
		int csv_y = atoi(getfield(tmp_y, 2));
		free(tmp_y);
		char* tmp_z = _strdup(line);
		int csv_z = atoi(getfield(tmp_z, 3));
		free(tmp_z);
		char* tmp_state = _strdup(line);
		int csv_state = atoi(getfield(tmp_state, 4));
		free(tmp_state);
		char* tmp_bodyIndex = _strdup(line);
		int csv_bodyIndex = atoi(getfield(tmp_bodyIndex, 5));
		free(tmp_bodyIndex);

		if (csv_bodyIndex == 9999)
		{
			isBody[bodyIndex] = false;
		}
		if (csv_bodyIndex != 9999)
		{
			isBody[bodyIndex] = true;
		}

		(skeletonDatas_origin[bodyIndex][jointsCount])[0] = csv_x;
		(skeletonDatas_origin[bodyIndex][jointsCount])[1] = csv_y;
		(skeletonDatas_origin[bodyIndex][jointsCount])[2] = csv_z;
		(skeletonDatas_origin[bodyIndex][jointsCount])[3] = csv_state;
		jointsCount++;
	}

	fclose(stream);
}



void addSeedByBodyIndex(Mat* mask, BYTE* bodyIndexData){
	
	Point2i* min_positions = new Point2i[kTotal_BodyIndex];
	Point2i* max_positions = new Point2i[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		min_positions[i] = Point2i(IMAGE_WIDTH, IMAGE_HEIGHT);		
		max_positions[i] = Point2i(0, 0);
		isBody[i] = false;
	}
		

	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH * 2; col += 2){

			int index = bodyIndexData[row * IMAGE_WIDTH * 2 + col];

			if (index <= kTotal_BodyIndex){
				isBody[index] = true;
				if (bodyIndexData[row * IMAGE_WIDTH * 2 + col + 1] == 1){
					mask[index].at<uchar>(row, (col / 2)) = GC_FGD;
				}
				else{				
					mask[index].at<uchar>(row, (col / 2)) = GC_PR_FGD;
				}

			}
			if (col / 2 < min_positions[index].x) min_positions[index].x = col / 2;
			if (row < min_positions[index].y) min_positions[index].y = row;
			if (col / 2 > max_positions[index].x) max_positions[index].x = col / 2;
			if (row > max_positions[index].y) max_positions[index].y = row;
			
		}
	}
	
	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (max_positions[i].x != 0 && max_positions[i].y != 0){		

			int width	= max_positions[i].x - min_positions[i].x;
			int height	= max_positions[i].y - min_positions[i].y;

			int width_expand	= (double)width * 0.2;
			int height_expand	= (double)height * 0.2;

			Point2i start	= Point2i(min_positions[i].x - width_expand, min_positions[i].y - height_expand);
			Point2i end		= Point2i(max_positions[i].x + width_expand, max_positions[i].y + height_expand);

			if (start.x < 0) start.x = 0;
			if (start.y < 0) start.y = 0;
			if (end.x > IMAGE_WIDTH) end.x = IMAGE_WIDTH;
			if (end.y > IMAGE_HEIGHT) end.y = IMAGE_HEIGHT;


			for (int row = start.y; row < end.y; row++){
				for (int col = start.x; col < end.x; col++){					
					if (mask[i].at<uchar>(row, col) == GC_BGD){
						mask[i].at<uchar>(row, col) = GC_PR_BGD;
					}
				}
			}
		}
	}
	

	delete[] min_positions;
	delete[] max_positions;			
}

void addSeedBySkeleton(Mat* mask, BYTE** skeletonDatas){

	for (int row = 0; row < IMAGE_HEIGHT; row++){
		for (int col = 0; col < IMAGE_WIDTH; col++){

			for (int i = 0; i < kTotal_BodyIndex; i++){
				if (skeletonDatas[i][row * IMAGE_WIDTH + col] == 1){
					mask[i].at<uchar>(row, col) = GC_PR_FGD;
				}
			}
		}
	}
}


void createDistanceMaps(Mat*& outputInput, BYTE** skeletonDataMaps){
			
	outputInput = new Mat[kTotal_BodyIndex];

	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (isBody[i]){

			Mat binaryInput = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, Scalar(255));

			for (int row = 0; row < IMAGE_HEIGHT; row++){
				for (int col = 0; col < IMAGE_WIDTH; col++){

					if (skeletonDataMaps[i][row * 1920 + col] == 1){
						binaryInput.at<uchar>(row, col) = 0;
					}

				}
			}
			distanceTransform(binaryInput, outputInput[i], CV_DIST_L1, CV_DIST_MASK_3);
		}
	}		
}


vector<vector<Point2i>> getSkeletonLinesAround(int jointNumber, cv::Vec4i* skeletonJointPosition){
		
	vector<int> connectedJointNumbers;
	/*
	ref. kinect v2 doc
	*/
#pragma region ConnectedJointNumber
	if (jointNumber == 0){
		//SPIN_BASE	
		connectedJointNumbers.push_back(1);
		connectedJointNumbers.push_back(12);
		connectedJointNumbers.push_back(16);		
	}
	else if (jointNumber == 1){
		//SPINE_MID
		connectedJointNumbers.push_back(0);
		connectedJointNumbers.push_back(20);
	}
	else if (jointNumber == 2){
		//NECT
		connectedJointNumbers.push_back(3);
		connectedJointNumbers.push_back(20);
	}
	else if (jointNumber == 3){
		//HEAD
		connectedJointNumbers.push_back(2);
	}
	else if (jointNumber == 4){
		//SHOULDER_LEFT
		connectedJointNumbers.push_back(9);
		connectedJointNumbers.push_back(20);
	}
	else if (jointNumber == 5){
		//ELBOW_LEFT
		connectedJointNumbers.push_back(4); 
		connectedJointNumbers.push_back(6);
	}
	else if (jointNumber == 6){
		//WRIST_LEFT
		connectedJointNumbers.push_back(5);
		connectedJointNumbers.push_back(7);
	}
	else if (jointNumber == 7){
		//HAND_LEFT
		connectedJointNumbers.push_back(6);
		connectedJointNumbers.push_back(21);
		connectedJointNumbers.push_back(22);		
	}
	else if (jointNumber == 8){
		//SHOULDER_RIGHT
		connectedJointNumbers.push_back(5);
		connectedJointNumbers.push_back(9);
	}
	else if (jointNumber == 9){
		//ELBOW_RIGHT
		connectedJointNumbers.push_back(8);
		connectedJointNumbers.push_back(10);
	}
	else if (jointNumber == 10){
		//WRIST_RIGHT
		connectedJointNumbers.push_back(9);
		connectedJointNumbers.push_back(11);
	}
	else if (jointNumber == 11){
		//HAND_RIGHT
		connectedJointNumbers.push_back(10);
		connectedJointNumbers.push_back(23);
		connectedJointNumbers.push_back(24);
	}
	else if (jointNumber == 12){
		//HIP_LEFT
		connectedJointNumbers.push_back(0);
		connectedJointNumbers.push_back(13);
	}
	else if (jointNumber == 13){
		//KNEE_LEFT
		connectedJointNumbers.push_back(12);
		connectedJointNumbers.push_back(14);
	}
	else if (jointNumber == 14){
		//ANKLE_LEFT
		connectedJointNumbers.push_back(13);
		connectedJointNumbers.push_back(15);
	}
	else if (jointNumber == 15){
		//FOOT_LEFT
		connectedJointNumbers.push_back(14);
	}
	else if (jointNumber == 16){
		//HIP_RIGHT
		connectedJointNumbers.push_back(0);
		connectedJointNumbers.push_back(17);
	}
	else if (jointNumber == 17){
		//KNEE_RIGHT
		connectedJointNumbers.push_back(16);
		connectedJointNumbers.push_back(18);
	}
	else if (jointNumber == 18){
		//ANKLE_RIGHT
		connectedJointNumbers.push_back(17);
		connectedJointNumbers.push_back(19);
	}
	else if (jointNumber == 19){
		//FOOT_RIGHT
		connectedJointNumbers.push_back(18);		
	}
	else if (jointNumber == 20){
		//SPINE_SHOULDER
		connectedJointNumbers.push_back(1);
		connectedJointNumbers.push_back(2);
		connectedJointNumbers.push_back(4);
		connectedJointNumbers.push_back(8);
	}
	else if (jointNumber == 21){
		//HAND_TIP_LEFT
		connectedJointNumbers.push_back(7);
	}
	else if (jointNumber == 22){
		//THUMB_LEFT
		connectedJointNumbers.push_back(7);
	}
	else if (jointNumber == 23){
		//HAND_TIP_RIGHT
		connectedJointNumbers.push_back(11);
	}
	else if (jointNumber == 24){
		//THUMB_RIGHT
		connectedJointNumbers.push_back(11);
	}
	else{		
		std::printf("Joint 번호가 잘못됨\n");		
	}
#pragma endregion
	
	vector<vector<Point2i>> table;
	table.assign(connectedJointNumbers.size(), vector<Point2i>());
	
	for (int i = 0; i < connectedJointNumbers.size(); i++){
		table[i] = getLineFromTwoPoint(skeletonJointPosition[jointNumber], skeletonJointPosition[connectedJointNumbers[i]]);
	}			

	return table;
}


void createSkeletonData(BYTE**& skeletonDatasMap, BYTE**& skeletonDatasMap2, int fileNumber, int frameNumber, cv::Vec4i** skeletonDatas_origin, cv::Vec4i** skeletonDatas_hd){
	
	skeletonDatasMap = new BYTE*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonDatasMap[i] = new BYTE[IMAGE_HEIGHT * IMAGE_WIDTH];
	}	
	skeletonDatasMap2 = new BYTE*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		skeletonDatasMap2[i] = new BYTE[IMAGE_HEIGHT * IMAGE_WIDTH];
	}
	for (int row = 0; row < 1080; row++){
		for (int col = 0; col < 1920; col++){
			for (int i = 0; i < kTotal_BodyIndex; i++){
				skeletonDatasMap[i][row * 1920 + col] = 0;
				skeletonDatasMap2[i][row * 1920 + col] = 0;
			}			
		}
	}

	short* mappingDats;
	loadMappingFile(mappingDats, fileNumber, frameNumber);

	for (int i = 0; i < kTotal_BodyIndex; i++){

		if (isBody[i]){
			
			for (int j = 0; j < kJointFromKinectV2; j++){
				skeletonDatas_hd[i][j] = mappingLowToHigh(skeletonDatas_origin[i][j], mappingDats);
			}

			CreateSkeletonLines(skeletonDatas_hd[i], IMAGE_WIDTH, IMAGE_HEIGHT, 3, skeletonDatasMap[i], skeletonDatasMap2[i]);			
		
		}
	}

	delete[] mappingDats;
}

void createSkeletonWeightMap(double**& weightMaps, int frame, cv::Vec4i** skeletonDatas_hd, Mat* distanceMaps){
	
	weightMaps = new double*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){
		weightMaps[i] = new double[IMAGE_HEIGHT * IMAGE_WIDTH];
	}
	
	double** jointsSigma = new double*[kTotal_BodyIndex];
	for (int i = 0; i < kTotal_BodyIndex; i++){

		jointsSigma[i] = new double[kJointFromKinectV2];

		for (int j = 0; j < kJointFromKinectV2; j++){
			jointsSigma[i][j] = sigma;
			if (skeletonDatas_hd[i][j][3] == 1){
				//inferred
				jointsSigma[i][j] = inferred_sigma;
			}
		}

		//HEAD
		jointsSigma[i][3] = head_sigma;
		//FOOT_LEFT
		jointsSigma[i][15] = foot_sigma;
		//FOOT_RIGHT
		jointsSigma[i][19] = foot_sigma;
	}
	
	for (int i = 0; i < kTotal_BodyIndex; i++){
		if (isBody[i]){
			for (int row = 0; row < IMAGE_HEIGHT; row++){
				for (int col = 0; col < IMAGE_WIDTH; col++){
					Point2i pos = Point2i(col, row);
					int jointNumber = getCloestJointNumber(pos, skeletonDatas_hd[i]);

					double jointsigma = jointsSigma[i][jointNumber];
					float distance = distanceMaps[i].at<float>(row, col);
					//const float K = (1.0 / (sqrt(2 * PIE * powf(sigma, 2))));
					const float K = 1;

					double gaussian_weight = K *  exp(-1.0 * (powf(distance, 2) / (2.0 * powf(jointsigma, 2))));

					if (gaussian_weight < 0.00002)
						gaussian_weight = 0.00002;

					if (weightMaps[i][row * IMAGE_WIDTH + col] < gaussian_weight)
						weightMaps[i][row * IMAGE_WIDTH + col] = gaussian_weight;
				}
			}
		}
	}

	for (int i = 0; i < kTotal_BodyIndex; i++){
		delete[] jointsSigma[i];
	}
	delete[] jointsSigma;
}

void CreateSkeletonLines(cv::Vec4i* jointPoints, int width, int height, int stroke, BYTE* skeletonMap, BYTE* skeletonMap2){
	//SPINE_BASE - SPINE_MID
	drawLineBy(jointPoints[0], jointPoints[1], width, height, stroke, skeletonMap, skeletonMap2);
	//SPINE_BASE - HIP_RIGHT
	drawLineBy(jointPoints[0], jointPoints[16], width, height, stroke, skeletonMap, skeletonMap2);
	//SPINE_BASE - HIP_LEFT
	drawLineBy(jointPoints[0], jointPoints[12], width, height, stroke, skeletonMap, skeletonMap2);
	//SPINE_MID - SPINE_SHOULDER
	drawLineBy(jointPoints[1], jointPoints[20], width, height, stroke, skeletonMap, skeletonMap2);
	//SPINE_SHOULDER - NECK
	drawLineBy(jointPoints[20], jointPoints[2], width, height, stroke, skeletonMap, skeletonMap2);
	//SPINE_SHOULDER - SHOULDER_RIGHT
	drawLineBy(jointPoints[20], jointPoints[8], width, height, stroke, skeletonMap, skeletonMap2);
	//SHOULDER_RIGHT - SHOULDER_LEFT
	drawLineBy(jointPoints[20], jointPoints[4], width, height, stroke, skeletonMap, skeletonMap2);
	//NECT - HEAD
	drawLineBy(jointPoints[2], jointPoints[3], width, height, stroke, skeletonMap, skeletonMap2);
	//SHOULDER_LEFT - ELBOW_RIGHT
	drawLineBy(jointPoints[8], jointPoints[9], width, height, stroke, skeletonMap, skeletonMap2);
	//ELBOW_RIGHT - WRIST_RIGHT
	drawLineBy(jointPoints[9], jointPoints[10], width, height, stroke, skeletonMap, skeletonMap2);
	//WRIST_RIGHT - HAND_RIGHT
	drawLineBy(jointPoints[10], jointPoints[11], width, height, stroke, skeletonMap, skeletonMap2);
	//HAND_RIGHT - HAND_TIP_RIGHT
	drawLineBy(jointPoints[11], jointPoints[23], width, height, stroke, skeletonMap, skeletonMap2);
	//HAND_RIGHT - THUMB_RIGHT
	drawLineBy(jointPoints[11], jointPoints[24], width, height, stroke, skeletonMap, skeletonMap2);
	//SHOULDER_LEFT - ELBOW_LEFT
	drawLineBy(jointPoints[4], jointPoints[5], width, height, stroke, skeletonMap, skeletonMap2);
	//ELBOW_LEFT - WRIST_LEFT
	drawLineBy(jointPoints[5], jointPoints[6], width, height, stroke, skeletonMap, skeletonMap2);
	//WRIST_LEFT - HAND_LEFT
	drawLineBy(jointPoints[6], jointPoints[7], width, height, stroke, skeletonMap, skeletonMap2);
	//HAND_LEFT - HAND_TIP_LEFT
	drawLineBy(jointPoints[7], jointPoints[21], width, height, stroke, skeletonMap, skeletonMap2);
	//HAND_LEFT - THUMB_LEFT
	drawLineBy(jointPoints[7], jointPoints[22], width, height, stroke, skeletonMap, skeletonMap2);
	//HIP_RIGHT - KNEE_RIGHT
	drawLineBy(jointPoints[16], jointPoints[17], width, height, stroke, skeletonMap, skeletonMap2);
	//KNEE_RIGHT - ANKLE_RIGHT
	drawLineBy(jointPoints[17], jointPoints[18], width, height, stroke, skeletonMap, skeletonMap2);
	//ANKLE_RIGHT - FOOT_RIGHT
	drawLineBy(jointPoints[18], jointPoints[19], width, height, stroke, skeletonMap, skeletonMap2);
	//HIP_LEFT - KNEE_LEFT
	drawLineBy(jointPoints[12], jointPoints[13], width, height, stroke, skeletonMap, skeletonMap2);
	//KNEE_LEFT - ANKLE_LEFT
	drawLineBy(jointPoints[13], jointPoints[14], width, height, stroke, skeletonMap, skeletonMap2);
	//ANKLE_LEFT - FOOT_LEFT
	drawLineBy(jointPoints[14], jointPoints[15], width, height, stroke, skeletonMap, skeletonMap2);
}

cv::Vec4i mappingLowToHigh(cv::Vec4i point, short* mappingData){
	cv::Vec4i pos;
	if (point[0] == 9999) pos[0] = -1;
	else pos[0] = mappingData[point[1] * 1024 + (2 * point[0]) + 1];

	if (point[1] == 9999) pos[1] = -1;
	else pos[1] = mappingData[point[1] * 1024 + (2 * point[0])];

	pos[2] = point[2];
	pos[3] = point[3];

	if (pos[0] < 0 || pos[0] >= 1920) pos[0] = -1;
	if (pos[1] < 0 || pos[1] >= 1080) pos[1] = -1;


	return pos;
}

void expandPixelBy(int x, int y, int width, int height, int stroke, BYTE* skeletonDatas){
	int offset = stroke / 2;
	if (offset < 0) offset = 0;

	for (int off_y = -offset; off_y <= offset; off_y++){
		for (int off_x = -offset; off_x <= offset; off_x++){

			if (off_y + y >= 0 && off_y + y < height){
				if (off_x + x >= 0 && off_x + x < width){
					skeletonDatas[(y + off_y) * width + (x + off_x)] = 1;
				}
			}
		}
	}
}


cv::vector<cv::Point2i> getLineFromTwoPoint(cv::Vec4i point1, cv::Vec4i point2){
	cv::vector<cv::Point2i> line;

	if (point1[0] == -1 && point1[1] == -1) return line;
	if (point2[0] == -1 && point2[1] == -1) return line;

	if (point1[3] != 2) return line;
	if (point2[3] != 2) return line;

	int start, end;

	if (point1[0] == point2[0]){
		if (point1[1] == point2[1]) return line;

		if (point1[1] > point2[1]){
			start = point2[1];
			end = point1[1];
		}
		else{
			start = point1[1];
			end = point2[1];
		}

		for (int i = 0; i <= end - start; i++){
			cv::Point2i pos = cv::Point2i(point1[0], start + i);
			line.push_back(pos);
		}

		return line;
	}
	
	else if (point1[1] == point2[1]){
		if (point1[0] == point2[0]) return line;

		if (point1[0] > point2[0]){
			start = point2[0];
			end = point1[0];
		}
		else{
			start = point1[0];
			end = point2[0];
		}

		for (int i = 0; i <= end - start; i++){
			cv::Point2i pos = cv::Point2i(start + i, point1[1]);
			line.push_back(pos);
		}

		return line;
	}
	
	else{
		double m = (double)(point2[1] - point1[1]) / (point2[0] - point1[0]);

		if (abs(point2[0] - point1[0]) >= abs(point2[1] - point1[1])){
			if (point1[0] >= point2[0]){
				start = point2[0];
				end = point1[0];
			}
			else{
				start = point1[0];
				end = point2[0];
			}

			for (int i = 0; i <= end - start; i++){
				int y = (int)(m * (start + i - point1[0]) + point1[1] + 0.5);
				cv::Point2i pos = cv::Point2i(start + i, y);
				line.push_back(pos);
			}

			return line;
		}
		else{
			if (point1[1] >= point2[1]){
				start = point2[1];
				end = point1[1];
			}
			else{
				start = point1[1];
				end = point2[1];
			}

			for (int i = 0; i <= end - start; i++){
				int x = (int)((start + i - point1[1]) * (1 / m) + point1[0] + 0.5);
				cv::Point2i pos = cv::Point2i(x, start + i);
				line.push_back(pos);
			}

			return line;
		}
	}
}

// For invalid data
cv::vector<cv::Point2i> getLineFromTwoPoint2(cv::Vec4i point1, cv::Vec4i point2){
	cv::vector<cv::Point2i> line;

	if (point1[0] == -1 && point1[1] == -1) return line;
	if (point2[0] == -1 && point2[1] == -1) return line;
	if (point1[3] == 2 && point2[3] == 2) return line;
	
	int start, end;

	if (point1[0] == point2[0]){
		if (point1[1] == point2[1]) return line;

		if (point1[1] > point2[1]){
			start = point2[1];
			end = point1[1];
		}
		else{
			start = point1[1];
			end = point2[1];
		}

		for (int i = 0; i <= end - start; i++){
			cv::Point2i pos = cv::Point2i(point1[0], start + i);
			line.push_back(pos);
		}

		return line;
	}
	else if (point1[1] == point2[1]){
		if (point1[0] == point2[0]) return line;

		if (point1[0] > point2[0]){
			start = point2[0];
			end = point1[0];
		}
		else{
			start = point1[0];
			end = point2[0];
		}

		for (int i = 0; i <= end - start; i++){
			cv::Point2i pos = cv::Point2i(start + i, point1[1]);
			line.push_back(pos);
		}

		return line;
	}
	else{
		double m = (double)(point2[1] - point1[1]) / (point2[0] - point1[0]);

		if (abs(point2[0] - point1[0]) >= abs(point2[1] - point1[1])){
			if (point1[0] >= point2[0]){
				start = point2[0];
				end = point1[0];
			}
			else{
				start = point1[0];
				end = point2[0];
			}

			for (int i = 0; i <= end - start; i++){
				int y = (int)(m * (start + i - point1[0]) + point1[1] + 0.5);
				cv::Point2i pos = cv::Point2i(start + i, y);
				line.push_back(pos);
			}

			return line;
		}
		else{
			if (point1[1] >= point2[1]){
				start = point2[1];
				end = point1[1];
			}
			else{
				start = point1[1];
				end = point2[1];
			}

			for (int i = 0; i <= end - start; i++){
				int x = (int)((start + i - point1[1]) * (1 / m) + point1[0] + 0.5);
				cv::Point2i pos = cv::Point2i(x, start + i);
				line.push_back(pos);
			}

			return line;
		}
	}
}

void drawLineBy(cv::Vec4i start, cv::Vec4i end, int width, int height, int stroke, BYTE* skeletonDatas, BYTE* skeletonDatas2){
	vector<Point2i> line2 = getLineFromTwoPoint2(start, end);
	for (int i = 0; i < line2.size(); i++)
	{
		expandPixelBy(line2[i].x, line2[i].y, width, height, stroke, skeletonDatas2);
	}

	vector<Point2i> line = getLineFromTwoPoint(start, end);
	for (int i = 0; i < line.size(); i++)
	{
		expandPixelBy(line[i].x, line[i].y, width, height, stroke, skeletonDatas);
	}

	
}

const char* getfield(char* line, int num)
{
	const char* tok;
	for (tok = strtok(line, ",");
		tok && *tok;
		tok = strtok(NULL, ",\n"))
	{
		if (!--num)
			return tok;
	}
	return NULL;
}

int getL1DistanceTwoPoint(Point2i left, Point2i right){
	int distance;
	return distance = abs(left.x - right.x) + abs(left.y - right.y);
}
double getEuclideanDistanceTwoPoint(Point2i left, Point2i right){
	double distance;
	return sqrt(pow(left.x - right.x, 2) + pow(left.y - right.y, 2));
}

int getCloestJointNumber(Point2i point, cv::Vec4i* skeletonJointPosition){
	int number = 0;
	ushort tmp = 65535;
	for (int j = 0; j < kJointFromKinectV2; j++){
		Point2i jointPosition(skeletonJointPosition[j][0], skeletonJointPosition[j][1]);
		ushort distance = getL1DistanceTwoPoint(point, jointPosition);
		if (tmp > distance){
			number = j;
			tmp = distance;
		}
	}

	return number;
}

