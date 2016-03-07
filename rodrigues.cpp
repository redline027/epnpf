#include "pnpf.h"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;
/* add errors, dimensional check*/
void rodrigues(MatrixXd &R, double *V, int type)
{
	Mat cvR(3, 3, CV_64F);
	Mat cvV(3, 1, CV_64F);

	if (type == FROM_RODR_TO_MATR)
	{
		for (int i = 0; i < 3; i++)
			cvV.at<double>(i, 0) = V[i];

		Rodrigues(cvV, cvR);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				R(i, j) = cvR.at<double>(i, j);
	}
	else if (type == FROM_MATR_TO_RODR)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				cvR.at<double>(i, j) = R(i, j);

		Rodrigues(cvR, cvV);

		for (int i = 0; i < 3; i++)
			V[i] = cvV.at<double>(i, 0);
	}
}