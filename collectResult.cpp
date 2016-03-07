#include "pnpf.h"

using namespace Eigen;
using namespace std;

void decodeEucCamera(double *camparam, MatrixXd &R, VectorXd &t)
{
	rodrigues(R, camparam, FROM_RODR_TO_MATR);
	t.resize(0);
	t = VectorXd::Zero(3);
	for (int i = 3; i < 6; i++)
		t(i - 3) = camparam[i];
}

void encodeCamera(MatrixXd &R, Ref<const VectorXd> t, double *camparam)
{
	rodrigues(R, camparam, FROM_MATR_TO_RODR);
	for (int i = 3; i < 6; i++)
		camparam[i] = t(i - 3);
}

typedef struct refine_pos_foc_res2
{
	Ref<const MatrixXd> pts3d;
} refine_pos_foc_res2_struct;

void distort1(VectorXd &proj, double k1, double f)
{
	double r = proj.norm();
	proj = f*proj*(1 + k1*r*r);

}

void refine_pos_foc_res2(double *x, double *res, int m, int n, void *data)
{
	refine_pos_foc_res2_struct* tmp_data = (refine_pos_foc_res2_struct*)data;
	int ptNum = tmp_data->pts3d.cols();
	double f = x[6];
	MatrixXd R(3, 3);
	VectorXd t(3);
	decodeEucCamera(x, R, t);
	MatrixXd P(3, 4);
	P.block(0, 0, 3, 3) = R;
	P.col(3) = t;
	double d = 0;
	for (int i = 0; i < ptNum; i++)
	{
		VectorXd pt3d(tmp_data->pts3d.rows() + 1);
		pt3d.head(tmp_data->pts3d.rows()) = tmp_data->pts3d.col(i);
		pt3d(tmp_data->pts3d.rows()) = 1;
		VectorXd proj = P * pt3d;
		proj /= proj(2);
		distort1(proj, d, f);
		for (int j = 2 * i, ind = 0; j < 2 * i + 2; j++, ind++)
			res[j] = proj(ind);
	}
}

void refine_pos_dist(MatrixXd &Rest, VectorXd &test, double &fest, double &dest, Ref<const MatrixXd> projs, Ref<const MatrixXd> pts3d, int u, int v, int dist, int isFast, double fMin, double fMax, int isFastBA, double &avgerr)
{
	int noConstr = 0;
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];

	if (isFast == 0)
		opts[0] = 1E-01, opts[1] = 1E-15, opts[2] = 1E-15, opts[3] = 1E-15, opts[4] = 1E-6;
	else
	{
		if (isFastBA == 1)
			opts[0] = 1E-03, opts[1] = 1E-5, opts[2] = 1E-5, opts[3] = 1E-5, opts[4] = 1E-6;
		else
			opts[0] = 1E-01, opts[1] = 1E-10, opts[2] = 1E-10, opts[3] = 1E-10, opts[4] = 1E-6;
		noConstr = 1;
	}

	int itmax = 200;
	int ret;
	int n = projs.size();

	double *imvect = new double[n];
	for (int col_ind = 0; col_ind < projs.cols(); col_ind++)
		for (int row_ind = 0; row_ind < projs.rows(); row_ind++)
			imvect[col_ind*projs.rows() + row_ind] = projs(row_ind, col_ind);

	int m = 7;
	double *x = new double[m];
	for (int i = 0; i < m; i++)
		x[i] = 0;

	JacobiSVD<MatrixXd> svd(Rest, ComputeFullV | ComputeFullU);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	Rest = U * V.transpose();

	if (Rest.determinant() < 0)
	{
		Rest = -1 * Rest;
		test = -1 * test;
	}

	if (dist == 0)
	{
		encodeCamera(Rest, test, x);
		x[6] = fest;

		refine_pos_foc_res2_struct data = { pts3d };

		if (noConstr == 1)
		{
			ret = dlevmar_dif(refine_pos_foc_res2, x, imvect, m, n, itmax, opts, info, NULL, NULL, (void*)&data);
		}
		else
		{
			double *lb = new double[m];
			double *ub = new double[m];
			for (int i = 0; i < m; i++)
			{
				lb[i] = -1e10;
				ub[i] = 1e10;
			}
			lb[6] = fMin;
			ub[6] = fMax;

			/* real(x) ??? */
			ret = dlevmar_bc_dif(refine_pos_foc_res2, x, imvect, m, n, lb, ub, NULL, itmax, opts, info, NULL, NULL, (void*)&data);

			delete[] lb;
			delete[] ub;
		}
		decodeEucCamera(x, Rest, test);
		fest = x[6];
		avgerr = sqrt(info[1] / pts3d.cols());
	}

	delete[] imvect;
	delete[] x;
}

void findBetterSolution(class pnpf &newSolution, class pnpf &currentSolution)
{
	if (newSolution.avgErr < 0)
		return;
	if (currentSolution.avgErr < 0 || newSolution.avgErr < currentSolution.avgErr)
	{
		currentSolution.avgErr = newSolution.avgErr;
		currentSolution.R = newSolution.R;
		currentSolution.t = newSolution.t;
		currentSolution.f = newSolution.f;
		return;
	}
}

void collectResult(double retVal, MatrixXd &Rest, VectorXd &test, double &fest, Ref<const MatrixXd> Uc, Ref<const MatrixXd> pts, class pnpf &currentSolution, int isFast, const class pnpOpts &pnpOpts)
{
	if (retVal > 0)
	{
		int noDist = 0;
		double dest = 0;
		double avgErr;
		refine_pos_dist(Rest, test, fest, dest, Uc, pts, 0, 0, noDist, isFast, pnpOpts.fMin, pnpOpts.fMax, pnpOpts.isFastBA, avgErr);
		class pnpf solution;
		solution.R = Rest;
		solution.t = test;
		solution.f = fest;
		solution.avgErr = avgErr;
		findBetterSolution(solution, currentSolution);
	}
}
