#include "pnpf.h"

using namespace Eigen;
using namespace std;

/*
* Need to initialize Ab, bb, AbF, bbF. Depends on adjustInLKer.
*/
void formConstraints(int betaNum, Ref<const MatrixXd> Vq, Ref<const MatrixXd> kerVect, Ref<const VectorXd> x0, Ref<const VectorXd> w, int varNum, Ref<MatrixXd> Ab, Ref<VectorXd> bb, Ref<MatrixXd> AbF, Ref<VectorXd> bbF)
{
	for (int betaInd = 0; betaInd < betaNum; betaInd++)
	{
		Ab.row(betaInd) = kerVect.row(betaInd) * Vq;
		VectorXd temp = kerVect.row(betaInd) * w;
		bb(betaInd) = temp(0) - x0(betaInd);
		AbF.row(betaInd) = kerVect.row(varNum + betaInd) * Vq;
		temp = kerVect.row(varNum + betaInd) * w;
		bbF(betaInd) = temp(0) - x0(varNum + betaInd);
	}
}

/*
* Didn't test.
*/
void adjustInLKer(Ref<const VectorXd> x0, Ref<const MatrixXd> kerVect, int betaNum, Ref<const MatrixXd> L, int isFast, VectorXd &vMy, double &alphaMy, int &errSign)
{
	errSign = 0;

	int nKV = kerVect.cols();

	MatrixXd Q = MatrixXd::Zero(nKV, nKV);
	VectorXd b = VectorXd::Zero(nKV);
	int varNum = betaNum + betaNum*(betaNum - 1) / 2;
	VectorXd x1 = x0.head(varNum);
	VectorXd x2 = x0.segment(varNum, varNum);
	VectorXd temp = x1.transpose() * x2;
	double c = temp(0);

	for (int kerInd = 0; kerInd < nKV; kerInd++)
	{
		VectorXd temp = x1.transpose() * kerVect.col(kerInd).segment(varNum, varNum) + kerVect.col(kerInd).head(varNum).transpose() * x2;
		b(kerInd) = temp(0);
	}

	for (int kerInd1 = 0; kerInd1 < nKV; kerInd1++)
		for (int kerInd2 = 0; kerInd2 < nKV; kerInd2++)
		{
			VectorXd temp = kerVect.col(kerInd1).head(varNum).transpose() * kerVect.col(kerInd2).segment(varNum, varNum) + kerVect.col(kerInd2).head(varNum).transpose() * kerVect.col(kerInd1).segment(varNum, varNum);
			Q(kerInd1, kerInd2) = 0.5 * temp(0);
		}

	EigenSolver<MatrixXd> eig(Q);
	MatrixXd Vq = eig.eigenvectors().real();
	MatrixXd Sq = eig.eigenvalues().real().asDiagonal();

	VectorXd w = 0.5 * pseudoInverse(Q) * b;
	temp = b.transpose() * pseudoInverse(Q) * b;
	double cn = -0.25 * temp(0) + c;

	VectorXd svect = eig.eigenvalues().real();

	MatrixXd Ab = MatrixXd::Zero(betaNum, Vq.cols());
	VectorXd bb = VectorXd::Zero(betaNum);
	MatrixXd AbF = Ab;
	VectorXd bbF = bb;
	formConstraints(betaNum, Vq, kerVect, x0, w, varNum, Ab, bb, AbF, bbF);

	int sum = 0;
	for (int i = 0; i < svect.size(); i++)
		if (svect(i) > 0)
			sum++;
		else if (svect(i) < 0)
			sum--;

	VectorXd xkAns;
	if (sum == -nKV)
	{
		xkAns = -w;
	}
	else
	{
		double smallStep = 1e-1;
		VectorXd posSVect = VectorXd::Zero(svect.size());
		for (int i = 0; i < svect.size(); i++)
			if (svect(i) > 0)
				posSVect(i) = 1;
		VectorXd xnAns = VectorXd::Zero(svect.size());
		int cnPos = 0;
		VectorXd xnAnsBounds = xnAns;
		if (cn > 0)
		{
			xnAnsBounds = VectorXd::Zero(svect.size());
			cnPos = 1;
		}
		else
		{
			for (int diagAnsComp = 0; diagAnsComp < svect.size(); diagAnsComp++)
			{
				if (svect(diagAnsComp) > 0)
				{
					xnAnsBounds(diagAnsComp) = sqrt((-cn) / svect(diagAnsComp)) + smallStep;
				}
			}
		}

		vector<int> workInds;
		for (int i = 0; i < posSVect.size(); i++)
			if (posSVect(i) > 0)
				workInds.push_back(i);
		/* ??? begin */
		VectorXd temp = Ab * xnAns;
		VectorXd ineqHoldInds = VectorXd::Zero(bb.size());
		for (int i = 0; i < bb.size(); i++)
		{
			if (temp(i) > bb(i))
				ineqHoldInds(i) = 1;
		}
		int flag = 0;
		for (int i = 0; i < ineqHoldInds.size(); i++)
			if (ineqHoldInds(i) >= Ab.rows())
				flag++;
		/* ??? end */
		if (flag == 0)
		{
			int noGoodAns = 1;
			for (int quadInd = 0; quadInd < pow(2, workInds.size()); quadInd++)
			{
				VectorXd binVec = VectorXd::Zero(workInds.size());
				de2bi(quadInd, workInds.size(), binVec);

				MatrixXd A2 = MatrixXd::Zero(workInds.size(), workInds.size());
				VectorXd b2 = VectorXd::Zero(workInds.size());
				VectorXd f = b2;
				for (int varInd = 0; varInd < workInds.size(); varInd++)
				{
					if (binVec(varInd) == 0)
					{
						f(varInd) = -1;
						A2(varInd, varInd) = -1;
						if (!cnPos)
						{
							b2(varInd) = xnAnsBounds(workInds.at(varInd));
						}
					}
					else
					{
						f(varInd) = 1;
						A2(varInd) = 1;
						if (!cnPos)
						{
							b2(varInd) = xnAnsBounds(workInds.at(varInd));
						}
					}
				}

				for (int conInd = 0; conInd < Ab.rows(); conInd++)
				{
					MatrixXd Ab2 = MatrixXd::Zero(A2.rows() + 2, A2.cols());
					VectorXd bb2 = VectorXd::Zero(b2.size() + 2);

					for (int i = 0; i < workInds.size(); i++)
					{
						Ab2(0, i) = Ab(conInd, workInds.at(i));
						Ab2(1, i) = AbF(conInd, workInds.at(i));
					}
					Ab2.block(2, 0, A2.rows(), A2.cols()) = A2;
					Ab2 = -Ab2;

					bb2(0) = bb(conInd);
					bb2(1) = bbF(conInd);
					bb2.tail(b2.size()) = b2;
					bb2 = -bb2;

					/* linprog: begin */
					ClpSimplex  model;
					int n_cols = f.size();
					double* objective = new double[n_cols];
					double* col_lb = new double[n_cols];
					double* col_ub = new double[n_cols];
					for (int i = 0; i < n_cols; i++)
						objective[i] = f(i);
					col_lb[0] = -1 * COIN_DBL_MAX;
					col_lb[1] = -1 * COIN_DBL_MAX;
					col_ub[0] = COIN_DBL_MAX;
					col_ub[1] = COIN_DBL_MAX;

					int n_rows = Ab2.rows();
					double * row_lb = new double[n_rows];
					double * row_ub = new double[n_rows];
					CoinPackedMatrix* matrix = new CoinPackedMatrix(false, 0, 0);
					matrix->setDimensions(0, n_cols);
					for (int i = 0; i < n_rows; i++)
					{
						CoinPackedVector row;
						for (int j = 0; j < Ab2.cols(); j++)
							row.insert(j, Ab2(i, j));
						row_lb[i] = -1 * COIN_DBL_MAX;
						row_ub[i] = bb2(i);
						matrix->appendRow(row);
					}

					model.loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);

					model.primal();
					/* maybe not only isProvenOptimal??? */
					if (model.isProvenOptimal())
					{
						VectorXd xl = VectorXd::Zero(model.getNumCols());
						const double *solution;
						solution = model.getColSolution();
						for (int i = 0; i < xl.size(); i++)
							xl(i) = solution[i];

						VectorXd constrInds = VectorXd::Zero(bb2.size());
						VectorXd temp = Ab2*xl - bb2;
						for (int i = 0; i < temp.size(); i++)
							if (temp(i) <= 0)
								constrInds(i) = 1;

						if (constrInds.sum() == constrInds.size())
						{
							for (int i = 0; i < workInds.size(); i++)
								xnAns(workInds[i]) = xl(i);
							noGoodAns = 0;
							break;
						}
					}
					/* linprog: end*/
				}

				if (!noGoodAns)
					break;
			}

			if (noGoodAns)
			{
				vMy.resize(0, 0);
				alphaMy = -1;
				//popt???
				return;
			}
		}
		xkAns = Vq * xnAns - w;
	}

	VectorXd xkN = x0 + kerVect*xkAns;

	double alphaNMy;
	VectorXd vNMy = VectorXd::Zero(varNum);
	alphaFormula(xkN.head(varNum), xkN.segment(varNum, varNum), vNMy, alphaNMy);

	if (alphaNMy < 0)
	{
		vMy = vNMy;
		alphaMy = alphaNMy;
		//popt???
		return;
	}

	if (isFast == 1)
	{
		vMy = vNMy;
		alphaMy = alphaNMy;
		//popt???
		return;
	}

	/*
	...start - same as ...N
	there are many questions

	double alphaMyStart = -1;
	VectorXd vMyStart = VectorXd::Zero(varNum);
	alphaFormula(xkN.head(varNum), xkN.segment(varNum, varNum), vNMy, alphaNMy);
	*/

	VectorXd popt = xkAns;
	VectorXd xAns = x0 + kerVect * popt.head(nKV);
	alphaFormula(xAns.head(varNum), xAns.segment(varNum, varNum), vMy, alphaMy);
	if (alphaMy < 0)
	{
		vMy = vNMy;
		alphaMy = alphaNMy;
	}
}


void generateBetaSqsFromBetas(int betaNum, Ref<const VectorXd> x, VectorXd &sol1)
{
	int varNum = betaNum + betaNum*(betaNum - 1) / 2;
	sol1 = VectorXd::Zero(varNum);
	int ind = betaNum;
	for (int i1 = 0; i1 < betaNum; i1++)
	{
		sol1(i1) = x(i1) * x(i1);
		for (int i2 = i1 + 1; i2 < betaNum; i2++)
		{
			sol1(ind) = x(i1) * x(i2);
			ind++;
		}
	}
}

typedef struct LKerResMultiC_struct
{
	Ref<const MatrixXd> kerVect;
	Ref<const VectorXd> initSol;
	int varNum;
	Ref<const MatrixXd> L;
}LKerResMultiC_struct;

void LKerResMultiC(double *x, double *res, int m, int n, void *data)
{
	LKerResMultiC_struct* tmp_data = (LKerResMultiC_struct*)data;
	VectorXd x1(m);
	for (int i = 0; i < m; i++)
		x1(i) = x[i];
	int nKV = tmp_data->kerVect.cols();
	VectorXd sol = tmp_data->initSol + tmp_data->kerVect*x1.head(nKV);
	VectorXd vMy;
	double alphaMy;
	alphaFormula(sol.head(tmp_data->varNum), sol.segment(tmp_data->varNum, tmp_data->varNum), vMy, alphaMy);
	if (alphaMy < 0)
		alphaMy = 0;
	VectorXd x1est = vMy - sol.head(tmp_data->varNum);
	VectorXd x2est = alphaMy*vMy - sol.segment(tmp_data->varNum, tmp_data->varNum);
	VectorXd tmp(x1est.size() + x2est.size());
	tmp.head(x1est.size()) = x1est;
	tmp.tail(x2est.size()) = x2est;
	tmp = tmp_data->L * tmp;
	for (int i = 0; i < n; i++)
		res[i] = tmp(i);
}

void adjustInLKerMulti(Ref<const VectorXd> x0, Ref<const MatrixXd> kerVect, int betaNum, Ref<const MatrixXd> L, int isFast, MatrixXd &xkNs)
{
	MatrixXd xks(0, 0);
	int nKV = kerVect.cols();

	MatrixXd Q = MatrixXd::Zero(nKV, nKV);
	VectorXd b = VectorXd::Zero(nKV);
	int varNum = betaNum + betaNum*(betaNum - 1) / 2;
	VectorXd x1 = x0.head(varNum);
	VectorXd x2 = x0.segment(varNum, varNum);
	VectorXd temp = x1.transpose() * x2;
	double c = temp(0);

	for (int kerInd = 0; kerInd < nKV; kerInd++)
	{
		VectorXd temp = x1.transpose() * kerVect.col(kerInd).segment(varNum, varNum) + kerVect.col(kerInd).head(varNum).transpose() * x2;
		b(kerInd) = temp(0);
	}

	for (int kerInd1 = 0; kerInd1 < nKV; kerInd1++)
		for (int kerInd2 = 0; kerInd2 < nKV; kerInd2++)
		{
			VectorXd temp = kerVect.col(kerInd1).head(varNum).transpose() * kerVect.col(kerInd2).segment(varNum, varNum) + kerVect.col(kerInd2).head(varNum).transpose() * kerVect.col(kerInd1).segment(varNum, varNum);
			Q(kerInd1, kerInd2) = 0.5 * temp(0);
		}

	EigenSolver<MatrixXd> eig(Q);
	MatrixXd Vq = eig.eigenvectors().real();
	MatrixXd Sq = eig.eigenvalues().real().asDiagonal();

	VectorXd w = 0.5 * pseudoInverse(Q) * b;
	temp = b.transpose() * pseudoInverse(Q) * b;
	double cn = -0.25 * temp(0) + c;

	VectorXd svect = eig.eigenvalues().real();

	MatrixXd Ab = MatrixXd::Zero(betaNum, Vq.cols());
	VectorXd bb = VectorXd::Zero(betaNum);
	MatrixXd AbF = Ab;
	VectorXd bbF = bb;
	formConstraints(betaNum, Vq, kerVect, x0, w, varNum, Ab, bb, AbF, bbF);

	int sum = 0;
	for (int i = 0; i < svect.size(); i++)
		if (svect(i) > 0)
			sum++;
		else if (svect(i) < 0)
			sum--;

	VectorXd xkAns;
	if (sum == -nKV)
	{
		xkAns = -w;
		MatrixXd temp = MatrixXd::Zero(xkNs.rows(), xkNs.cols() + 1);
		VectorXd xkN = x0 + kerVect*xkAns;
		temp.block(0, 0, xkNs.rows(), xkNs.cols()) = xkNs;
		temp.col(xkNs.cols()) = xkN;
		xkNs = temp;

		temp = MatrixXd::Zero(xks.rows(), xks.cols() + 1);
		temp.block(0, 0, xks.rows(), xks.cols()) = xks;
		temp.col(xks.cols()) = xkAns;
		xks = temp;
	}
	else
	{
		double smallStep = 1e-1;
		VectorXd posSVect = VectorXd::Zero(svect.size());
		for (int i = 0; i < svect.size(); i++)
			if (svect(i) > 0)
				posSVect(i) = 1;
		VectorXd xnAns = VectorXd::Zero(svect.size());
		int cnPos = 0;
		VectorXd xnAnsBounds = xnAns;
		if (cn > 0)
		{
			xnAnsBounds = VectorXd::Zero(svect.size());
			cnPos = 1;
		}
		else
		{
			for (int diagAnsComp = 0; diagAnsComp < svect.size(); diagAnsComp++)
			{
				if (svect(diagAnsComp) > 0)
				{
					xnAnsBounds(diagAnsComp) = sqrt((-cn) / svect(diagAnsComp)) + smallStep;
				}
			}
		}

		vector<int> workInds;
		for (int i = 0; i < posSVect.size(); i++)
			if (posSVect(i) > 0)
				workInds.push_back(i);
		/* ??? begin */
		VectorXd temp = Ab * xnAns;
		VectorXd ineqHoldInds = VectorXd::Zero(bb.size());
		for (int i = 0; i < bb.size(); i++)
		{
			if (temp(i) > bb(i))
				ineqHoldInds(i) = 1;
		}
		int flag = 0;
		for (int i = 0; i < ineqHoldInds.size(); i++)
			if (ineqHoldInds(i) >= Ab.rows())
				flag++;
		/* ??? end */
		if (flag == 0)
		{
			int noGoodAns = 1;
			for (int quadInd = 0; quadInd < pow(2, workInds.size()); quadInd++)
			{
				VectorXd binVec = VectorXd::Zero(workInds.size());
				de2bi(quadInd, workInds.size(), binVec);

				MatrixXd A2 = MatrixXd::Zero(workInds.size(), workInds.size());
				VectorXd b2 = VectorXd::Zero(workInds.size());
				VectorXd f = b2;
				for (int varInd = 0; varInd < workInds.size(); varInd++)
				{
					if (binVec(varInd) == 0)
					{
						f(varInd) = -1;
						A2(varInd, varInd) = -1;
						if (!cnPos)
						{
							b2(varInd) = xnAnsBounds(workInds.at(varInd));
						}
					}
					else
					{
						f(varInd) = 1;
						A2(varInd) = 1;
						if (!cnPos)
						{
							b2(varInd) = xnAnsBounds(workInds.at(varInd));
						}
					}
				}

				for (int conInd = 0; conInd < Ab.rows(); conInd++)
				{
					MatrixXd Ab2 = MatrixXd::Zero(A2.rows() + 2, A2.cols());
					VectorXd bb2 = VectorXd::Zero(b2.size() + 2);

					for (int i = 0; i < workInds.size(); i++)
					{
						Ab2(0, i) = Ab(conInd, workInds.at(i));
						Ab2(1, i) = AbF(conInd, workInds.at(i));
					}
					Ab2.block(2, 0, A2.rows(), A2.cols()) = A2;
					Ab2 = -Ab2;

					bb2(0) = bb(conInd);
					bb2(1) = bbF(conInd);
					bb2.tail(b2.size()) = b2;
					bb2 = -bb2;

					/* linprog: begin */
					ClpSimplex  model;
					int n_cols = f.size();
					double* objective = new double[n_cols];
					double* col_lb = new double[n_cols];
					double* col_ub = new double[n_cols];
					for (int i = 0; i < n_cols; i++)
						objective[i] = f(i);
					col_lb[0] = -1 * COIN_DBL_MAX;
					col_lb[1] = -1 * COIN_DBL_MAX;
					col_ub[0] = COIN_DBL_MAX;
					col_ub[1] = COIN_DBL_MAX;

					int n_rows = Ab2.rows();
					double * row_lb = new double[n_rows];
					double * row_ub = new double[n_rows];
					CoinPackedMatrix* matrix = new CoinPackedMatrix(false, 0, 0);
					matrix->setDimensions(0, n_cols);
					for (int i = 0; i < n_rows; i++)
					{
						CoinPackedVector row;
						for (int j = 0; j < Ab2.cols(); j++)
							row.insert(j, Ab2(i, j));
						row_lb[i] = -1 * COIN_DBL_MAX;
						row_ub[i] = bb2(i);
						matrix->appendRow(row);
					}

					model.loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);

					model.primal();
					/* maybe not only isProvenOptimal??? */
					if (model.isProvenOptimal())
					{
						VectorXd xl = VectorXd::Zero(model.getNumCols());
						const double *solution;
						solution = model.getColSolution();
						for (int i = 0; i < xl.size(); i++)
							xl(i) = solution[i];

						VectorXd constrInds = VectorXd::Zero(bb2.size());
						VectorXd temp = Ab2*xl - bb2;
						for (int i = 0; i < temp.size(); i++)
							if (temp(i) <= 0)
								constrInds(i) = 1;

						if (constrInds.sum() == constrInds.size())
						{
							for (int i = 0; i < workInds.size(); i++)
								xnAns(workInds[i]) = xl(i);
							/* different part */
							MatrixXd temp = MatrixXd::Zero(xkNs.rows(), xkNs.cols() + 1);
							VectorXd xkAns = Vq*xnAns - w;
							VectorXd xkN = x0 + kerVect*xkAns;
							temp.block(0, 0, xkNs.rows(), xkNs.cols()) = xkNs;
							temp.col(xkNs.cols()) = xkN;
							xkNs = temp;

							temp = MatrixXd::Zero(xks.rows(), xks.cols() + 1);
							temp.block(0, 0, xks.rows(), xks.cols()) = xks;
							temp.col(xks.cols()) = xkAns;
							xks = temp;
						}
					}
					/* linprog: end*/
				}

				if (!noGoodAns)
					break;
			}
			if (noGoodAns)
			{
				return;
			}
		}
	}

	if (isFast == 1)
		return;

	xkNs.resize(0, 0);

	for (int candInd = 0; candInd < xks.cols(); candInd++)
	{
		double *C = new double[2 * betaNum*kerVect.cols()];
		double *d = new double[2 * betaNum];
		for (int i = 0; i < betaNum; i++)
		{
			for (int j = 0; j < kerVect.cols(); j++)
				C[i*kerVect.cols() + j] = kerVect(i, j);
			d[i] = -x0(i);
		}
		for (int i = varNum; i < varNum + betaNum; i++)
		{
			for (int j = 0; j < kerVect.cols(); j++)
				C[i*kerVect.cols() + j] = kerVect(i, j);
			d[i] = -x0(i);
		}

		int m = xks.rows(), n = L.rows();
		double *lb = new double[m];
		double *ub = new double[m];
		double *xStart = new double[m];
		for (int i = 0; i < m; i++)
		{
			lb[i] = -1e100;
			ub[i] = 1e100;
			xStart[i] = xks(i, candInd);
		}
		double *imvect = new double[n];
		for (int i = 0; i < n; i++)
			imvect[i] = 0;
		double opts[LM_OPTS_SZ] = { 1E-03, 1E-20, 1E-20, 1E-20, 1E-6 }, info[LM_INFO_SZ];
		int itmax = 200;
		int ret;
		LKerResMultiC_struct data = { kerVect, x0, varNum, L };

		ret = dlevmar_bleic_dif(LKerResMultiC, xStart, imvect, m, n, lb, ub, NULL, NULL, 0, C, d, 2 * betaNum, itmax, opts, info, NULL, NULL, (void *)&data);

		VectorXd d_tmp(2 * betaNum);
		d_tmp.head(betaNum) = -x0.head(betaNum);
		d_tmp.tail(betaNum) = -x0.segment(varNum, betaNum);
		MatrixXd C_tmp(2 * betaNum, kerVect.cols());
		C_tmp.block(0, 0, betaNum, kerVect.cols()) = kerVect.block(0, 0, betaNum, kerVect.cols());
		C_tmp.block(betaNum, 0, betaNum, kerVect.cols()) = kerVect.block(varNum, 0, betaNum, kerVect.cols());
		VectorXd res(m);
		for (int i = 0; i < m; i++)
			res(i) = xStart[i];
		res = C_tmp*res - d_tmp;
		int sum = 0;
		for (int i = 0; i < res.size(); i++)
			if (res(i) > -1e-3)
				sum++;
		if (ret < 0 || sum < C_tmp.rows())
		{
			double *C_1 = new double[betaNum*kerVect.cols()];
			double *d_1 = new double[betaNum];
			for (int i = 0; i < betaNum; i++)
			{
				for (int j = 0; j < kerVect.cols(); j++)
					C_1[i*kerVect.cols() + j] = kerVect(i, j);
				d_1[i] = -x0(i);
			}

			ret = dlevmar_bleic_dif(LKerResMultiC, xStart, imvect, m, n, lb, ub, NULL, NULL, 0, C_1, d_1, betaNum, itmax, opts, info, NULL, NULL, (void *)&data);

			delete[] C_1;
			delete[] d_1;
		}

		res.resize(m);
		for (int i = 0; i < m; i++)
			res(i) = xStart[i];
		VectorXd xkN = x0 + kerVect*res;
		MatrixXd temp = MatrixXd::Zero(xkNs.rows(), xkNs.cols() + 1);
		temp.block(0, 0, xkNs.rows(), xkNs.cols()) = xkNs;
		temp.col(xkNs.cols()) = xkN;
		xkNs = temp;

		delete[] C;
		delete[] d;
		delete[] lb;
		delete[] ub;
		delete[] xStart;
		delete[] imvect;
	}
}

