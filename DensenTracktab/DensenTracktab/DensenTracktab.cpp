// DensenTracktab.cpp : 定义控制台应用程序的入口点。
//

#include"stdafx.h"

#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>
#include <sstream>
#include<io.h>
#include<iostream>
using namespace std;
using namespace cv;
int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
void  featureExtract(char* video, string outfilename,string posfilename)
{
	FILE *outfile = fopen(outfilename.c_str(), "wb");
	FILE *posoutfile = fopen(posfilename.c_str(), "wb");
	VideoCapture capture;
	 capture.open(video);
	if (!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return;
	}
	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);
	//	bb_file = "C:\\Users\\epiker1\\Desktop\\v_HorseRiding_g01_c01.bb";
	//std::vector<Frame> bb_list;
	//if (bb_file) {
	//	LoadBoundBox(bb_file, bb_list);
	//	assert(bb_list.size() == seqInfo.length);
//	}

	//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	//if (show_track == 1)
	//	namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while (true) {
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if (frame.empty())
			break;

		if (frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		if (frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for (int iScale = 0; iScale < scale_num; iScale++) {
				if (iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale - 1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for (i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
		//	if (bb_file)
		//		InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
		//if (bb_file)
		//	InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if (pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if (countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

		for (int iScale = 0; iScale < scale_num; iScale++) {
			if (iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale - 1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height + 1, width + 1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height + 1, width + 1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width - 1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height - 1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2 * x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2 * x + 1];

				if (point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2 * x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2 * x + 1];

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
			//	if (show_track == 1 && iScale == 0)
				//	DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if (iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length + 1), trajectory1(trackInfo.length + 1);
					for (int i = 0; i <= trackInfo.length; ++i)
					{
						trajectory[i] = iTrack->point[i] * fscales[iScale];
					    trajectory1[i] = iTrack->point[i] * fscales[iScale];
					}
						

					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i] * fscales[iScale];
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if (IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
					
						
						// output the trajectory
						//fprintf(file,"%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
			    		fwrite(&frame_num, sizeof(int), 1, outfile);
						fwrite(&mean_x, sizeof(float), 1, outfile);
						fwrite(&mean_y, sizeof(float), 1, outfile);
						fwrite(&var_x, sizeof(float), 1, outfile);
						fwrite(&var_y, sizeof(float), 1, outfile);
						fwrite(&length, sizeof(float), 1, outfile);
						float cscale = fscales[iScale];
						fwrite(&cscale, sizeof(float), 1, outfile);

						// for spatio-temporal pyramid
						//fprintf(file,"%f\t", std::min<float>(std::max<float>(mean_x / float(seqInfo.width), 0), 0.999));
						//fprintf(file,"%f\t", std::min<float>(std::max<float>(mean_y / float(seqInfo.height), 0), 0.999));
						//fprintf(file,"%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length / 2.0 - start_frame) / float(seqInfo.length), 0), 0.999));
						//fclose(file);
						float temp = std::min<float>(max<float>(mean_x / float(seqInfo.width), 0), 0.999);
						fwrite(&temp, sizeof(float), 1, outfile);
						temp = std::min<float>(max<float>(mean_y / float(seqInfo.height), 0), 0.999);
						fwrite(&temp, sizeof(float), 1, outfile);
						temp = std::min<float>(max<float>((frame_num - trackInfo.length / 2.0 - start_frame) / float(seqInfo.length), 0), 0.999);
						fwrite(&temp, sizeof(float), 1, outfile);
						// output the trajectory
					//	for (int i = 0; i < trackInfo.length; ++i)
					//		fprintf(file,"%f\t%f\t", displacement[i].x, displacement[i].y);
						for (int i = 0; i < trackInfo.length; ++i){
							temp = displacement[i].x;
							fwrite(&temp, sizeof(float), 1, outfile);
							temp = displacement[i].y;
							fwrite(&temp, sizeof(float), 1, outfile);
						}
						PrintDesc(outfile,iTrack->hog, hogInfo, trackInfo);
						PrintDesc(outfile,iTrack->hof, hofInfo, trackInfo);
						PrintDesc(outfile,iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(outfile,iTrack->mbhY, mbhInfo, trackInfo);
						float temp1;
						for (int i = 0; i < trackInfo.length; ++i){
							temp1 = trajectory1[i].x;
							fwrite(&temp1, sizeof(float), 1, posoutfile);
							temp1 = trajectory1[i].y;
							fwrite(&temp1, sizeof(float), 1, posoutfile);
						}
					}
					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if (init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for (i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for (i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

		if (show_track == 1) {
			imshow("DenseTrackStab", image);
			c = cvWaitKey(3);
			if ((char)c == 27) break;
		}
	}

	if (show_track == 1)
		destroyWindow("DenseTrackStab");
	fclose(posoutfile);
	fclose(outfile);
}
bool file_exists(const char * filename)
{
	if (FILE * file = fopen(filename, "r"))
	{
		fclose(file);
		return true;
	}
	return false;
}
void work(string start, string end,string dest2)
{
	if (file_exists(end.c_str()))
	{
		printf("%s Exists!\n",end.c_str());
		return;
	}	
	int len = start.length();
	char *inputfile = (char *)malloc((len + 1)*sizeof(char));
	strcpy(inputfile, start.c_str());
	featureExtract(inputfile,end,dest2);
	free(inputfile);
}
int cnt = 0;
void getFiles(string path, string dest,string dest2)
{
	//cout << path <<endl<< dest << endl;
	long   hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{

			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					cnt++;
					//if (cnt>=35) continue;
					//if (cnt <35||cnt>= 70) continue;
					//if (cnt <70) continue;
					cout << p.assign(path).append("\\").append(fileinfo.name) << endl;
					string cmd = "md " + dest + "\\" + fileinfo.name;
					system(cmd.c_str());
					cmd = "md " + dest2+ "\\" + fileinfo.name;
					system(cmd.c_str());
					getFiles(p.assign(path).append("\\").append(fileinfo.name), dest + "\\" + fileinfo.name, dest2 + "\\" + fileinfo.name);
					//printf("%s\n%s\n",p.assign(path).append("\\").append(fileinfo.name).c_str(), dest.c_str());
					//	system("pause");
				}
			}
			else
			{
               
				string str = fileinfo.name; 
				
				clock_t start = clock();
				work(p.assign(path).append("\\").append(fileinfo.name), dest + "\\" + str.substr(0, str.length() - 4), dest2 + "\\" + str.substr(0, str.length() - 4));
				clock_t end = clock();
				printf("%s time:%lf s\n", str.c_str(), (double)(end - start) / CLOCKS_PER_SEC);
				//system("pause");
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
int main(int argc, char** argv)
{

	//char* video = "C:\\Users\\epiker\\Documents\\UCF-101\\CricketBowling\\v_CricketBowling_g03_c01.avi";// argv[1];
	//featureExtract(video,"tmp");
	//work(string(video), "C:\\trajectory");
	//runPCA("bb.txt","pca.txt",128);
	//PCA
	//char *sourcepath = "C:\\Users\\epiker1\\Documents\\UCF-101";
	//char *destpath = "C:\\trajectory";
	//getFiles(sourcepath,destpath);
	/*char *source = "C:\\Users\\epiker\\Documents\\UCF-101";
	char *dest = "X:\\UCF101_idt";
	char *dest2 = "X:\\UCF101_pos";*/
	char *source = "C:\\Users\\epiker\\Documents\\Olympic Sports";
	char *dest = "X:\\Olympic_Sports_idt";
	char *dest2 = "X:\\Olympic_Sports_pos";
	getFiles(string(source),string(dest),string(dest2));
	
	//work(source,dest);
	//system("pause");
	return 0;
}
