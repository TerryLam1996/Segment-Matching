#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <io.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#define scale_choice_num 5
#define scale_MIN 0.1
#define scale_MAX 10
#define PAD_NUM 1
#define POINT_STEP 2
#define lambda 100
#define ANGLE_NUM 60
#define ACC_THRESHOLD 0.8
#define TOP_CHOICE 100
float scale_choice[scale_choice_num] = { 1.0, 0.9, 1.1, 0.95, 1.05 };
int check_num = 1;

std::string load_temp, load_src;
cv::Mat img_template, img_src, img_template_gray, img_src_gray, img_src_padded, draw_lines_template, draw_lines_src;
std::vector<cv::Vec4f> lines_template, lines_src;
std::string imgPath;
std::ofstream fout;
bool is_there;
int src_size_avg, avg_cnt;
double detection_time, matching_time;

void line_detection(cv::Mat img_template_gray, cv::Mat img_src_gray, cv::Mat& draw_lines_template, cv::Mat& draw_lines_src, std::vector<cv::Vec4f>& lines_template, std::vector<cv::Vec4f>& lines_src)
{
#if 0
	cv::Canny(img_src, img_src, 50, 200, 3); // Apply canny edge
#endif

/** Create and LSD detector with standard or no refinement. */
/*
#if 1
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.8, 0.6, 2.0, 15.0);
#else
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
#endif
*/

/** using LSD or FLD */
#if 0
	// Create LSD detector
	// Param				Default value	Description
	// refine				1			  -	The way found lines will be refined, see #LineSegmentDetectorModes
	// scale				0.8			  -	The scale of the image that will be used to find the lines.Range(0..1].
	// sigma_scale			0.6			  -	Sigma for Gaussian filter.It is computed as sigma = _sigma_scale / _scale.
	// quant				2.0			  -	Bound to the quantization error on the gradient norm.
	// ang_th				22.5		  -	Gradient angle tolerance in degrees.
	// log_eps				0			  -	Detection threshold : -log10(NFA) \> log_eps.Used only when advance refinement is chosen.
	// density_th			0.7			  -	Minimal density of aligned region points in the enclosing rectangle.
	// n_bins				1024		  -	Number of bins in pseudo - ordering of gradient modulus.
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.8, 0.6, 2.0, 15.0, 0, 0.7, 1024);
#else
	// Create FLD detector
	// Param				Default value	Description
	// length_threshold		10			  -	Segments shorter than this will be discarded
	// distance_threshold	1.41421356	  -	A point placed from a hypothesis line segment farther than this will be regarded as an outlier
	// canny_th1			50			  -	First threshold for hysteresis procedure in Canny()
	// canny_th2			50			  -	Second threshold for hysteresis procedure in Canny()
	// canny_aperture_size	3			  -	Aperturesize for the sobel operator in Canny()
	// do_merge				false		  -	If true, incremental merging of segments will be perfomred
	cv::Ptr<cv::ximgproc::FastLineDetector> ls = cv::ximgproc::createFastLineDetector(20, 1.41421356f, 50.0, 50.0, 3, false);
#endif

	double start = double(cv::getTickCount());
	/** Detect the lines */
	ls->detect(img_template_gray, lines_template);
	ls->detect(img_src_gray, lines_src);
	double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
	std::cout << "It took " << duration_ms << " ms for line detection." << std::endl;
	fout << "It took " << duration_ms << " ms for line detection." << std::endl;
	detection_time += duration_ms;
	/** Show found lines */
	draw_lines_template = cv::Mat::zeros(img_template_gray.size(), CV_8UC1);
	ls->drawSegments(draw_lines_template, lines_template);
	draw_lines_src = cv::Mat::zeros(img_src_gray.size(), CV_8UC1);
	ls->drawSegments(draw_lines_src, lines_src);
	std::cout << "template size: " << lines_template.size() << "\nsrc size: " << lines_src.size() << "\n";
	fout << "template size: " << lines_template.size() << "\nsrc size: " << lines_src.size() << "\n";
	src_size_avg += lines_src.size();
	avg_cnt++;
#if 0
	cv::Mat draw_lines_src_bg;
	img_src_gray.copyTo(draw_lines_src_bg);
	ls->drawSegments(draw_lines_src_bg, lines_src);
	cv::imshow("Template", draw_lines_template);
	cv::imshow("Source Image", draw_lines_src);
	cv::imshow("Source Image With Background", draw_lines_src_bg);
	//cv::waitKey(0);
	cv::waitKey(1);
	cv::destroyAllWindows();
#endif
}

bool is_in(float val_x, float val_y, cv::Point2f top_left, cv::Point2f bottom_right)
{
	return (val_x > top_left.x && val_y > top_left.y && val_x < bottom_right.x && val_y < bottom_right.y);
}

bool is_in(cv::Point2f val, cv::Point2f top_left, cv::Point2f bottom_right)
{
	return (val.x > top_left.x && val.y > top_left.y && val.x < bottom_right.x && val.y < bottom_right.y);
}

void get_parameter(float temp_lx, float temp_ly, float temp_rx, float temp_ry, float src_lx, float src_ly, float src_rx, float src_ry, cv::Point2f &trans, float &theta, float &scaled, cv::Point2f &ori)
{
	trans.x = src_lx - temp_lx;
	trans.y = src_ly - temp_ly;
	ori.x = temp_lx;
	ori.y = temp_ly;
	ori += trans;
	float temp_x = temp_rx - temp_lx, temp_y = temp_ry - temp_ly, src_x = src_rx - src_lx, src_y = src_ry - src_ly;
	theta = atan2(src_y, src_x) - atan2(temp_y, temp_x);
	scaled = sqrt(src_x * src_x + src_y * src_y) / sqrt(temp_x * temp_x + temp_y * temp_y);
}

void get_parameter(float temp_lx, float temp_ly, float temp_rx, float temp_ry, float src_lx, float src_ly, float src_rx, float src_ry, float &theta, float &scaled)
{
	float temp_x = temp_rx - temp_lx, temp_y = temp_ry - temp_ly, src_x = src_rx - src_lx, src_y = src_ry - src_ly;
	theta = atan2(src_y, src_x) - atan2(temp_y, temp_x);
	scaled = sqrt(src_x * src_x + src_y * src_y) / sqrt(temp_x * temp_x + temp_y * temp_y);
}

cv::Mat init_DT(cv::Mat draw_lines_src)
{
	cv::Mat draw_lines_src_gray, edges, dt;
	cv::cvtColor(draw_lines_src, draw_lines_src_gray, CV_BGR2GRAY);
	cv::threshold(draw_lines_src_gray, edges, 32, 255, cv::THRESH_BINARY_INV);
	cv::distanceTransform(edges, dt, cv::DIST_L2, cv::DIST_MASK_5);
	return dt;
}

void visualization_DT(cv::Mat dt)
{
	cv::Mat dt_vis;
	cv::normalize(dt, dt_vis, 1.0, 0, cv::NORM_MINMAX);
	cv::Mat dt_temp = dt_vis * 255;
	dt_temp.convertTo(dt_vis, CV_8UC1);
	cv::imshow("dt", dt_vis);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

cv::Mat dt_channel[ANGLE_NUM], dt3[ANGLE_NUM];
void init_DT3(cv::Mat img_src_gray, std::vector<cv::Vec4f> lines_src)
{
	double start = double(cv::getTickCount());
	for (int i = 0; i < ANGLE_NUM; i++)
		dt_channel[i] = cv::Mat::zeros(img_src_gray.size(), CV_8UC3);
	for (int i = 0; i < lines_src.size(); i++)
	{
		float theta = atan2(lines_src[i][3] - lines_src[i][1], lines_src[i][2] - lines_src[i][0]);
		int channel_num = (int)std::round((theta + M_PI) / M_PI * ANGLE_NUM) % ANGLE_NUM;
		cv::Point2f x((int)std::round(lines_src[i][0]), (int)std::round(lines_src[i][1])), y((int)std::round(lines_src[i][2]), (int)std::round(lines_src[i][3]));
		cv::line(dt_channel[channel_num], x, y, cv::Scalar(255, 255, 255));
	}
	for (int i = 0; i < ANGLE_NUM; i++)
		dt3[i] = init_DT(dt_channel[i]);

	for (int y = 0; y < dt3[0].rows; y++)
		for (int x = 0; x < dt3[0].cols; x++)
		{
			/** first iteration */
			int forward = ANGLE_NUM, backward = ANGLE_NUM;
			/** forward */
			while (dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x))
			{
				dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x) = dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
				forward--;
			}
			forward--;
			/** backward */
			while (dt3[backward % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x))
			{
				dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x) = dt3[backward % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
				backward++;
			}
			backward++;

			int left = forward, right = backward;

			while (right - left < ANGLE_NUM)
			{
				/** left */
				forward = left, backward = left;
				/** forward */
				while (dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x))
				{
					dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x) = dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
					forward--;
				}
				forward--;
				left = forward;
				/** backward */
				while (dt3[backward % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x))
				{
					dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x) = dt3[backward % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
					backward++;
				}
				backward++;
				right = std::max(right, backward);

				/** right */
				forward = right, backward = right;
				/** forward */
				while (dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x))
				{
					dt3[(forward + ANGLE_NUM - 1) % ANGLE_NUM].at<float>(y, x) = dt3[(forward + ANGLE_NUM) % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
					forward--;
				}
				forward--;
				left = std::min(left, forward);
				/** backward */
				while (dt3[backward % ANGLE_NUM].at<float>(y, x) + lambda * M_PI / ANGLE_NUM < dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x))
				{
					dt3[(backward + 1) % ANGLE_NUM].at<float>(y, x) = dt3[backward % ANGLE_NUM].at<float>(y, x) + (float)(lambda * M_PI / ANGLE_NUM);
					backward++;
				}
				backward++;
				right = backward;
			}

		}
	double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
	std::cout << "It took " << duration_ms << " ms for initializing DT maps.\n";
	fout << "It took " << duration_ms << " ms for initializing DT maps.\n";
}

float sin_[ANGLE_NUM * 2], cos_[ANGLE_NUM * 2];
void init_sin_cos()
{
	for (int i = 0; i < ANGLE_NUM * 2; i++)
	{
		sin_[i] = sin(M_PI * i / ANGLE_NUM);
		cos_[i] = cos(M_PI * i / ANGLE_NUM);
	}
}

std::vector<cv::Vec4f> lines_rotated[ANGLE_NUM * 2];
cv::Point2f lines_bbox[ANGLE_NUM * 2];
std::vector<cv::Point2f> points_template[ANGLE_NUM * 2];
void template_rotation()
{
	for (int i = 0; i < ANGLE_NUM * 2; i++)
	{
		/** top left and bottom right */
		cv::Point2f tl(img_template_gray.cols, img_template_gray.rows), br(-img_template_gray.cols, -img_template_gray.rows);
		cv::Point2f temp_points;
		cv::Vec4f temp_segs;
		for (int j = 0; j < lines_template.size(); j++)
		{
			temp_points.x = lines_template[j][0];
			temp_points.y = lines_template[j][1];
			temp_segs[0] = cos_[i] * temp_points.x - sin_[i] * temp_points.y;
			temp_segs[1] = sin_[i] * temp_points.x + cos_[i] * temp_points.y;
			tl.x = std::min(tl.x, temp_segs[0]);
			tl.y = std::min(tl.y, temp_segs[1]);
			br.x = std::max(br.x, temp_segs[0]);
			br.y = std::max(br.y, temp_segs[1]);
			temp_points.x = lines_template[j][2];
			temp_points.y = lines_template[j][3];
			temp_segs[2] = cos_[i] * temp_points.x - sin_[i] * temp_points.y;
			temp_segs[3] = sin_[i] * temp_points.x + cos_[i] * temp_points.y;
			tl.x = std::min(tl.x, temp_segs[2]);
			tl.y = std::min(tl.y, temp_segs[3]);
			br.x = std::max(br.x, temp_segs[2]);
			br.y = std::max(br.y, temp_segs[3]);
			lines_rotated[i].push_back(temp_segs);
		}
		/** tl -> (0, 0), br -> bbox */
		for (int j = 0; j < lines_template.size(); j++)
		{
			lines_rotated[i][j][0] -= tl.x;
			lines_rotated[i][j][1] -= tl.y;
			lines_rotated[i][j][2] -= tl.x;
			lines_rotated[i][j][3] -= tl.y;
			lines_bbox[i] = br - tl;
		}
	}
}

std::vector<cv::Point2f> template_segments_to_points(std::vector<cv::Vec4f> lines_template)
{
	std::vector<cv::Point2f> points_template;
	points_template.clear();
	cv::Point2f temp_template, line_l, line_r;
	for (int v = 0; v < lines_template.size(); v++)
	{
		//int cnt = 0;
		int x_t, y_t;
		line_l.x = lines_template[v][0];
		line_l.y = lines_template[v][1];
		line_r.x = lines_template[v][2];
		line_r.y = lines_template[v][3];
		x_t = line_l.x <= line_r.x ? POINT_STEP : -POINT_STEP;
		y_t = line_l.y <= line_r.y ? POINT_STEP : -POINT_STEP;
		int begin_x = (int)std::roundf(line_l.x), end_x = (int)std::roundf(line_r.x), begin_y = (int)std::roundf(line_l.y), end_y = (int)std::roundf(line_r.y);
		if (line_l.y == line_r.y)
		{
			int i = begin_x, j = begin_y;
			while ((i < end_x && x_t > 0) || (i > end_x && x_t < 0))
			{
				//cnt++;
				temp_template.x = i;
				temp_template.y = j;
				points_template.push_back(temp_template);
				i += x_t;
			}
		}
		else if (fabs(line_r.x - line_l.x) < fabs(line_r.y - line_l.y))
		{
			int i = begin_x, j = begin_y;
			float k1 = (line_r.x - line_l.x) / (line_r.y - line_l.y);
			while ((j < end_y && y_t > 0) || (j > end_y && y_t < 0))
			{
				//cnt++;
				temp_template.x = i;
				temp_template.y = j;
				points_template.push_back(temp_template);
				j += y_t;
				i = (int)std::roundf(line_l.x + (j - line_l.y) * k1);
			}
		}
		else
		{
			int i = begin_x, j = begin_y;
			float k = (line_r.y - line_l.y) / (line_r.x - line_l.x);
			while ((i < end_x && x_t > 0) || (i > end_x && x_t < 0))
			{
				//cnt++;
				temp_template.x = i;
				temp_template.y = j;
				points_template.push_back(temp_template);
				i += x_t;
				j = (int)std::roundf(line_l.y + (i - line_l.x) * k);
			}
		}
		//cnt++;
		temp_template.x = end_x;
		temp_template.y = end_y;
		points_template.push_back(temp_template);
	}
	return points_template;
}

cv::Point2f get_centroids(std::vector<cv::Vec4f> lines_temp)
{
	cv::Point2f sum(0.0, 0.0);
	cv::Point2f lines_l, lines_r, lines_m;
	double cnt = 0.0, lines_length;
	for (int i = 0; i < lines_temp.size(); i++)
	{
		lines_l.x = lines_temp[i][0];
		lines_l.y = lines_temp[i][1];
		lines_r.x = lines_temp[i][2];
		lines_r.y = lines_temp[i][3];
		lines_m.x = (lines_r.x + lines_l.x) / 2;
		lines_m.y = (lines_r.y + lines_l.y) / 2;
		lines_length = sqrt((lines_r.x - lines_l.x) * (lines_r.x - lines_l.x) + (lines_r.y - lines_l.y) * (lines_r.y - lines_l.y));
		cnt += lines_length;
		sum += (lines_m * lines_length);
	}
	sum /= cnt;
	return sum;
}

std::vector<cv::Point3f> get_centroids_map(std::vector<cv::Vec4f> lines_temp)
{
	cv::Point2f centroid = get_centroids(lines_temp);
	std::vector<cv::Point3f> ans;
	ans.clear();
	cv::Point2f lines_l, lines_r, lines_m;
	cv::Point3f points_temp;
	float lines_length, dist_centroid, lines_theta, centroid_theta;
	for (int i = 0; i < lines_temp.size(); i++)
	{
		lines_l.x = lines_temp[i][0];
		lines_l.y = lines_temp[i][1];
		lines_r.x = lines_temp[i][2];
		lines_r.y = lines_temp[i][3];
		lines_m.x = (lines_r.x + lines_l.x) / 2;
		lines_m.y = (lines_r.y + lines_l.y) / 2;
		lines_length = sqrt((lines_r.x - lines_l.x) * (lines_r.x - lines_l.x) + (lines_r.y - lines_l.y) * (lines_r.y - lines_l.y));
		dist_centroid = sqrt((centroid.x - lines_m.x) * (centroid.x - lines_m.x) + (centroid.y - lines_m.y) * (centroid.y - lines_m.y));
		lines_theta = atan2(lines_r.y - lines_l.y, lines_r.x - lines_l.x);
		centroid_theta = atan2(centroid.y - lines_m.y, centroid.x - lines_m.x);
		centroid_theta -= lines_theta;
		dist_centroid = dist_centroid / lines_length;
		points_temp.x = dist_centroid * cos(centroid_theta);
		points_temp.y = dist_centroid * sin(centroid_theta);
		/** points_temp.z: weights of points */
		points_temp.z = lines_length;
		ans.push_back(points_temp);
	}
	return ans;
}

std::vector<cv::Point3f> centroids_map[ANGLE_NUM * 2];
void centroids_rotation(std::vector<cv::Vec4f> lines_temp)
{
	centroids_map[0] = get_centroids_map(lines_temp);
	for (int i = 1; i < ANGLE_NUM * 2; i++)
	{
		cv::Point3f temp;
		for (int j = 0; j < centroids_map[0].size(); j++)
		{
			temp.x = cos_[i] * centroids_map[0][j].x - sin_[i] * centroids_map[0][j].y;
			temp.y = sin_[i] * centroids_map[0][j].x + cos_[i] * centroids_map[0][j].y;
			temp.z = centroids_map[0][j].z;
			centroids_map[i].push_back(temp);
		}
	}
}

struct tops
{
	cv::Point2f trans_;
	int theta_;
	float scaled_;
	cv::Point2f ori_;
	double dist_total_;
	double dist_;
	cv::Point2f bbox_tl;
	cv::Point2f bbox_br;
};

bool cmp_tops(tops x, tops y)
{
	return x.dist_ < y.dist_;
}

void draw_choice(tops top_c, std::vector<cv::Vec4f> *lines_rotated, std::vector<cv::Vec4f> lines_src, cv::Mat img_src, cv::Point2f tl, cv::Point2f br, cv::Scalar sca, int ind)
{
	std::vector<cv::Vec4f> lines_template_trans;
	cv::Point2f temp_l, temp_r;
	cv::Point2f trans = top_c.trans_, ori = top_c.ori_;
	float scaled_s = top_c.scaled_;
	int theta_deg = top_c.theta_;
	double dist_ans = top_c.dist_;
	for (int k = 0; k < lines_rotated[theta_deg].size(); k++)
	{
		temp_l.x = lines_rotated[theta_deg][k][0];
		temp_l.y = lines_rotated[theta_deg][k][1];
		temp_l = (temp_l - ori) * scaled_s + trans + ori;

		temp_r.x = lines_rotated[theta_deg][k][2];
		temp_r.y = lines_rotated[theta_deg][k][3];
		temp_r = (temp_r - ori) * scaled_s + trans + ori;

		cv::Vec4f push_in(temp_l.x, temp_l.y, temp_r.x, temp_r.y);
		lines_template_trans.push_back(push_in);
	}
	std::cout << "Distance: " << dist_ans << "\n";
	fout << "Distance: " << dist_ans << "\n";
	cv::Mat draw_lines_results(img_src.rows * PAD_NUM, img_src.cols * PAD_NUM, img_src.type(), cv::Scalar(0, 0));
	img_src.copyTo(cv::Mat(draw_lines_results, cv::Rect(img_src.cols * (PAD_NUM - 1) / 2, img_src.rows* (PAD_NUM - 1) / 2, img_src.cols, img_src.rows)));
	for (int j = 0; j < lines_src.size(); j++)
	{
		cv::Point2f x(lines_src[j][0], lines_src[j][1]), y(lines_src[j][2], lines_src[j][3]);
		cv::line(draw_lines_results, x, y, cv::Scalar(0, 0, 255));
	}
	for (int k = 0; k < lines_template_trans.size(); k++)
	{
		cv::Point2f x(lines_template_trans[k][0], lines_template_trans[k][1]), y(lines_template_trans[k][2], lines_template_trans[k][3]);
		cv::line(draw_lines_results, x, y, cv::Scalar(255, 0, 0));
	}
	if (is_there) cv::rectangle(draw_lines_results, tl, br, sca);

#if 1
	std::string save_path = "D:/CppCode/Segment-Matching/results/results_" + std::to_string(check_num) + "/" + std::to_string(check_num) + "_" + imgPath.substr(0, 7) + "_" + std::to_string(ind) + ".jpg";
	cv::imwrite(save_path, draw_lines_results);
#else
	cv::imshow("Result Image", draw_lines_results);
	cv::waitKey(1);
#endif
	lines_template_trans.clear();
}

void getAllFiles(std::string path, std::vector<std::string>& files)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main()
{
	/** initialize sin_[] cos_[] */
	init_sin_cos();

	/** empty lines_rotated */
	for (int i = 0; i < ANGLE_NUM * 2; i++) lines_rotated[i].clear();

	//std::cout << "Input template: ";
	load_temp = "E:/数据集/CVLab_UniBO_D-Textureless/D-Textureless/train/model_0" + std::to_string(check_num) + ".png";
	img_template = cv::imread(load_temp);

	/** annotations */
	std::ifstream fin;
	std::string originalPath = "E:/数据集/CVLab_UniBO_D-Textureless/D-Textureless/";
	std::vector<std::string> fileList;

	getAllFiles((originalPath + "Annotations"), fileList);
	std::cout << "Successfully load image list.\n";

	std::string txtFile = "D:/CppCode/Segment-Matching/results/results_" + std::to_string(check_num) + ".txt";
	fout.open(txtFile);
	int true_choice = 0, false_choice = 0, miss_choice = 0;
	src_size_avg = 0; avg_cnt = 0; detection_time = 0.0; matching_time = 0.0;

	for (int i = 0; i < fileList.size(); i++)
	{
		std::string s;
		int obj, xmin, ymin, xmax, ymax, ind, ind2;
		cv::Mat img;
		cv::Point2d tl, br;
		is_there = false;
		fin.open(fileList[i]);
		while (std::getline(fin, s))
		{
			ind = s.find_first_of('<') + 1;
			if (s[ind] == 'f' && s[ind + 1] == 'i')
			{
				ind2 = s.find_first_of('>') + 1;
				imgPath = s.substr(ind2, 11);
				img_src = cv::imread(originalPath + "test/" + imgPath);
				//if (!(img_template.data && img_src.data)) return -1;
				CV_Assert(img_template.data && img_src.data);
			}
			else if (s[ind] == 'o')
			{
				std::getline(fin, s);
				ind2 = s.find_first_of('>') + 1;
				obj = (int)s[ind2] - 48;
				for (int j = 0; j < 4; j++) std::getline(fin, s);
				if (obj == check_num)
				{
					is_there = true;
					std::getline(fin, s);
					ind = s.find_first_of('>') + 1;
					ind2 = s.find_first_of('<', ind);
					xmin = 0;
					for (int j = ind; j < ind2; j++)
						xmin = xmin * 10 + (int)s[j] - 48;
					std::getline(fin, s);
					ind = s.find_first_of('>') + 1;
					ind2 = s.find_first_of('<', ind);
					ymin = 0;
					for (int j = ind; j < ind2; j++)
						ymin = ymin * 10 + (int)s[j] - 48;
					std::getline(fin, s);
					ind = s.find_first_of('>') + 1;
					ind2 = s.find_first_of('<', ind);
					xmax = 0;
					for (int j = ind; j < ind2; j++)
						xmax = xmax * 10 + (int)s[j] - 48;
					std::getline(fin, s);
					ind = s.find_first_of('>') + 1;
					ind2 = s.find_first_of('<', ind);
					ymax = 0;
					for (int j = ind; j < ind2; j++)
						ymax = ymax * 10 + (int)s[j] - 48;
					tl.x = xmin; tl.y = ymin; br.x = xmax; br.y = ymax;
				}
				else for (int j = 0; j < 4; j++) std::getline(fin, s);
				
			}
		}
		//cv::imshow("img", img);
		//cv::waitKey(0);
		fin.close();

		cv::cvtColor(img_template, img_template_gray, CV_BGR2GRAY);
		/** Use a padded image, to avoid boundary checking */
		img_src_padded.create(img_src.rows * PAD_NUM, img_src.cols * PAD_NUM, img_src.type());
		img_src.copyTo(cv::Mat(img_src_padded, cv::Rect(img_src.cols * (PAD_NUM - 1) / 2, img_src.rows* (PAD_NUM - 1) / 2, img_src.cols, img_src.rows)));
		cv::cvtColor(img_src_padded, img_src_gray, CV_BGR2GRAY);
		cv::Point2f top_left(0, 0), bottom_right(img_src_gray.cols - 1, img_src_gray.rows - 1);

		line_detection(img_template_gray, img_src_gray, draw_lines_template, draw_lines_src, lines_template, lines_src);

		if (lines_rotated[0].empty())
		{
			/** template rotation */
			template_rotation();

			/** template segments to points */
			for (int i = 0; i < ANGLE_NUM * 2; i++)
				points_template[i] = template_segments_to_points(lines_rotated[i]);
		}

		if (centroids_map[0].empty()) centroids_rotation(lines_template);

		/** initialize DT map */
		cv::Mat dt = init_DT(draw_lines_src);

		/** visualization of DT map */
	#if 0
		visualization_DT(dt);
	#endif

		/** for DT3 map */
	#if 0
		init_DT3(img_src_gray, lines_src);
	#endif

		/** visualization of DT3 map */
	#if 0
		for (int i = 0; i < ANGLE_NUM; i++)
		{
			cv::imshow("dt_channel", dt_channel[i]);
			visualization_DT(dt3[i]);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	#endif

		/** the score of all results */
		std::vector<tops> top_choice;
		top_choice.clear();


		/** centroid distribution: x, y, weight, index of lines_src */
		std::vector<cv::Vec4f> centroid_distribution;
		centroid_distribution.clear();
		for (int j = 0; j < lines_src.size(); j++)
		{
			cv::Point2f temp_l, temp_r, temp_m;
			temp_l.x = lines_src[j][0];
			temp_l.y = lines_src[j][1];
			temp_r.x = lines_src[j][2];
			temp_r.y = lines_src[j][3];
			temp_m.x = (temp_r.x + temp_l.x) / 2;
			temp_m.y = (temp_r.y + temp_l.y) / 2;
			float temp_dist = sqrt((temp_r.x - temp_l.x) * (temp_r.x - temp_l.x) + (temp_r.y - temp_l.y) * (temp_r.y - temp_l.y));
			float theta = atan2(temp_r.y - temp_l.y, temp_r.x - temp_l.x);
			int theta_deg = ((int)std::round(theta / M_PI * ANGLE_NUM) + ANGLE_NUM * 2) % (ANGLE_NUM * 2);
			for (int i = 0; i < centroids_map[theta_deg].size(); i++)
			{
				cv::Vec4f temp_centroid;
				temp_centroid[0] = temp_m.x + centroids_map[theta_deg][i].x * temp_dist;
				temp_centroid[1] = temp_m.y + centroids_map[theta_deg][i].y * temp_dist;
				temp_centroid[2] = centroids_map[theta_deg][i].z;
				temp_centroid[3] = j;
				if (is_in(temp_centroid[0], temp_centroid[1], top_left, bottom_right)) centroid_distribution.push_back(temp_centroid);
			}
		}
		/** partitions of the points */
		cv::Mat partitions = cv::Mat::zeros(img_src_gray.size(), CV_32FC1);
		for (int i = 0; i < centroid_distribution.size(); i++)
		{
			int xind = (int)centroid_distribution[i][0];
			int yind = (int)centroid_distribution[i][1];
			partitions.at<float>(yind, xind) += centroid_distribution[i][2];
		}
		cv::Mat boxFilterDst, boxFilterDst_vis;
		cv::boxFilter(partitions, boxFilterDst, -1, cv::Size(50, 50));
		cv::normalize(boxFilterDst, boxFilterDst_vis, 1.0, 0, cv::NORM_INF);
		//cv::imshow("boxFilterResult", boxFilterDst_vis);
		//cv::waitKey(0);
		

		std::vector<bool> lines_src_selected_index;
		lines_src_selected_index.clear();
		for (int i = 0; i < lines_src.size(); i++) lines_src_selected_index.push_back(false);
		
		cv::Mat img_points;
		img_src.copyTo(img_points);
		for (int i = 0; i < centroid_distribution.size(); i++)
		{
			int xind = (int)centroid_distribution[i][0];
			int yind = (int)centroid_distribution[i][1];
			if (boxFilterDst_vis.at<float>(yind, xind) > 0.7)
			{
				lines_src_selected_index[centroid_distribution[i][3]] = true;
				cv::circle(img_points, cv::Point2f(xind, yind), 2, cv::Scalar(0, 255, 255), -1);
			}
		}
		for (int i = 0; i < lines_src.size(); i++)
			cv::line(img_points, cv::Point2f(lines_src[i][0], lines_src[i][1]), cv::Point2f(lines_src[i][2], lines_src[i][3]), cv::Scalar(0, 0, 255));
		cv::imshow("img_points", img_points);
		cv::waitKey(0);




		double start = double(cv::getTickCount());

		for (int i = 0; i < lines_template.size(); i++)
		{
			/** both two orientations of the segments */
			int o = 0;
			//for (int o = 0; o < 2; o++)
			{
				cv::Point2f trans, ori;
				float theta, scaled, scaled_s;
				for (int j = 0; j < lines_src.size(); j++)
				{
					if (!lines_src_selected_index[j]) continue;
					get_parameter(lines_template[i][(0 + o * 2) % 4], lines_template[i][(1 + o * 2) % 4], lines_template[i][(2 + o * 2) % 4], lines_template[i][(3 + o * 2) % 4], lines_src[j][0], lines_src[j][1], lines_src[j][2], lines_src[j][3], theta, scaled);
					if (scaled < scale_MIN || scaled > scale_MAX) continue;
					int theta_deg = ((int)std::round(theta / M_PI * ANGLE_NUM) + ANGLE_NUM * 2) % (ANGLE_NUM * 2);
					ori.x = lines_rotated[theta_deg][i][(0 + o * 2) % 4];
					ori.y = lines_rotated[theta_deg][i][(1 + o * 2) % 4];
					trans.x = lines_src[j][0] - ori.x;
					trans.y = lines_src[j][1] - ori.y;
					cv::Point2f temp_points;

					int s = 0;
					//for (int s = 0; s < scale_choice_num; s++)
					{
						scaled_s = scaled * scale_choice[s];
						/** if it is out of the bounding */
						if (!(is_in(trans + ori - ori * scaled_s, top_left, bottom_right) && is_in(trans + ori + (lines_bbox[theta_deg] - ori) * scaled_s, top_left, bottom_right))) continue;

						double distance_total = 0.0;
						for (int k = 0; k < points_template[theta_deg].size(); k++)
						{
							temp_points = (points_template[theta_deg][k] - ori) * scaled_s + ori + trans;
							distance_total += dt.at<float>(temp_points.y, temp_points.x);
						}
						double distance_chamfer_matching = distance_total / points_template[theta_deg].size() / scaled_s;
						//std::cout << "Distance: " << distance_chamfer_matching << "\n";

						tops temp_choice;
						temp_choice.trans_ = trans;
						temp_choice.theta_ = theta_deg;
						temp_choice.scaled_ = scaled_s;
						temp_choice.ori_ = ori;
						temp_choice.dist_total_ = distance_total;
						temp_choice.dist_ = distance_chamfer_matching;
						temp_choice.bbox_tl = (trans + ori - ori * scaled_s);
						temp_choice.bbox_br = (trans + ori + (lines_bbox[theta_deg] - ori) * scaled_s);
						top_choice.push_back(temp_choice);
					}
				}
			}
		}

		double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
		std::cout << "\nIt took " << duration_ms << " ms for matching.\n";
		fout << "\nIt took " << duration_ms << " ms for matching.\n";
		matching_time += duration_ms;

		std::cout << "Source image: " << imgPath << "\n";
		std::cout << "Tops: \n";
		fout << "Source image: " << imgPath << "\n";
		fout << "Tops: \n";
		std::sort(top_choice.begin(), top_choice.end(), cmp_tops);

		int model_num;
		cv::Point2f bbox_tl[10], bbox_br[10];
		float S[10], S_;
		//std::cout << "Input numbers of models: ";
		//std::cin >> model_num;
		model_num = 1;

		for (int i = 0; i < model_num; i++)
		{
			//std::cout << "input bounding box axis: ";
			//std::cin >> bbox_tl[i].x >> bbox_tl[i].y >> bbox_br[i].x >> bbox_br[i].y;
			bbox_tl[i] = tl; bbox_br[i] = br;
			//std::cout << std::endl;

			S[i] = (bbox_br[i].x - bbox_tl[i].x) * (bbox_br[i].y - bbox_tl[i].y);
		}
		for (int i = 0; i < top_choice.size(); i++)
		{
			float tx = top_choice[i].bbox_tl.x, ty = top_choice[i].bbox_tl.y, bx = top_choice[i].bbox_br.x, by = top_choice[i].bbox_br.y;
			S_ = (bx - tx) * (by - ty);
			float minx, miny, maxx, maxy, S_union, S_intersection, IoU = 0.0;
			for (int j = 0; j < model_num; j++)
			{
				minx = std::max(tx, bbox_tl[j].x);
				miny = std::max(ty, bbox_tl[j].y);
				maxx = std::min(bx, bbox_br[j].x);
				maxy = std::min(by, bbox_br[j].y);
				if (minx < maxx && miny < maxy)
				{
					S_intersection = (maxx - minx) * (maxy - miny);
					S_union = S_ + S[j] - S_intersection;
					IoU = std::max(IoU, S_intersection / S_union);
				}
			}
			if (top_choice[i].dist_ < 8 && i < TOP_CHOICE)
			{
				std::cout << "No. " << i + 1 << "\t\t";
				fout << "No. " << i + 1 << "\t\t";
				if (IoU >= ACC_THRESHOLD)
				{
					true_choice++;
					std::cout << "T\tIoU: " << IoU << "\t";
					fout << "T\tIoU: " << IoU << "\t";
				}
				else
				{
					false_choice++;
					std::cout << "F\tIoU: " << IoU << "\t";
					fout << "F\tIoU: " << IoU << "\t";
				}
				draw_choice(top_choice[i], lines_rotated, lines_src, img_src, tl, br, cv::Scalar(0, 255, 0), i + 1);
			}
			else
			{
				if (IoU >= ACC_THRESHOLD)
				{
					std::cout << "No. " << i + 1 << "\t\t";
					fout << "No. " << i + 1 << "\t\t";
					miss_choice++;
					std::cout << "M\tIoU: " << IoU << "\t";
					fout << "M\tIoU: " << IoU << "\t";
					draw_choice(top_choice[i], lines_rotated, lines_src, img_src, tl, br, cv::Scalar(0, 255, 0), i + 1);
				}
			}
		}
		cv::destroyAllWindows();

		//system("Pause");
		std::cout << "\n";
		fout << "\n\n";
	}
	std::cout << "True counts: " << true_choice << "\nFalse counts: " << false_choice << "\nMiss counts: " << miss_choice << "\nAccuracy rate: " << (double)true_choice / (true_choice + false_choice) << "\nPrecision rate: " << (double)true_choice / (true_choice + miss_choice) << "\n";
	fout << "True counts: " << true_choice << "\nFalse counts: " << false_choice << "\nMiss counts: " << miss_choice << "\nAccuracy rate: " << (double)true_choice / (true_choice + false_choice) << "\nPrecision rate: " << (double)true_choice / (true_choice + miss_choice) << "\n";
	fout << "src size average: " << (double)src_size_avg / avg_cnt << "\ndetection time average: " << detection_time / avg_cnt << "\nmatching timg average: " << matching_time / avg_cnt << "\n";

	fin.close();

	return 0;
}
