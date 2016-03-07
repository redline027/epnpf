#include "pnpf.h"

using namespace Eigen;
using namespace std;

void rtFromC(Ref<const MatrixXd> C, Ref<const MatrixXd> BPans, int tarPtNum, int &errFlag, MatrixXd &R, VectorXd &t)
{
	MatrixXd Ch(C.rows() + 1, C.cols());
	VectorXd ones = VectorXd::Constant(tarPtNum, 1);
	MatrixXd BPansh(BPans.rows() + 1, BPans.cols());
	Ch.block(0, 0, C.rows(), C.cols()) = C;
	Ch.row(C.rows()) = ones;
	BPansh.block(0, 0, BPans.rows(), BPans.cols()) = BPans;
	BPansh.row(BPans.rows()) = ones;
	errFlag = 0;
	MatrixXd Rth = BPans * Ch.inverse();		//also could be A/B = (B'\A')'
	double det = Rth.block(0, 0, 3, 3).determinant();
	if (det < 0)
		Rth = -Rth;

	if (det == 0)
		errFlag = 1;
	else
		Rth = Rth / pow(det, 1.0 / 3.0);		//it was without else

	MatrixXd Rt = Rth.block(0, 0, 3, 4);
	R = Rt.block(0, 0, Rt.rows(), 3);
	t = Rt.col(3);
}

void formBPans(vector<MatrixXd> &BP, Ref<const VectorXd> bvec, double fest, int tarPtNum, MatrixXd &BPans)
{
	BPans = MatrixXd::Zero(3, tarPtNum);
	for (int kerInd = 0; kerInd < BP.size(); kerInd++)
		BPans = BPans + BP[kerInd] * bvec(kerInd);
	BPans.row(2) = BPans.row(2) * fest;
}

void vote(Ref<const MatrixXd> cands, vector<MatrixXd> &BP, Ref<const MatrixXd> C, Ref<const VectorXd> resCosts, int tarPtNum, Ref<const MatrixXd> P, MatrixXd &R, VectorXd &t, double &f, VectorXd &candsBest)
{
	VectorXd resrate = VectorXd::Zero(cands.rows());			//int
	int minrate = P.cols();
	int betaNum = cands.cols() - 1;
	int candNum = cands.rows();
	vector<MatrixXd> Ta(candNum);
	for (int i = 0; i < candNum; i++)
		Ta[i] = MatrixXd::Zero(3, 4);
	VectorXd Tfa = VectorXd::Zero(candNum);

	if (candNum == 0)
	{
		f = -1;
		R.resize(0, 0);
		t.resize(0);
		return;
	}

	int validCandNum = 0;
	for (int i = 0; i < candNum; i++)
	{
		double fest = cands(i, betaNum);
		VectorXd betaVect = VectorXd::Zero(betaNum);
		for (int betaInd = 0; betaInd < betaNum; betaInd++)
			betaVect(betaInd) = cands(i, betaInd);
		MatrixXd BPans;
		formBPans(BP, betaVect, fest, tarPtNum, BPans);
		if (tarPtNum == 4)
		{
			int errFlag;
			rtFromC(C, BPans, tarPtNum, errFlag, R, t);
			if (errFlag == 1)
				continue;
		}
		else
		{
			formBPans(BP, cands.row(i).head(betaNum), fest, tarPtNum, BPans);
			MatrixXd F(C.rows(), 3);
			MatrixXd G(BPans.rows(), 3);
			int ind = 0;
			for (int ptInd = 0; ptInd < 3; ptInd++)
				for (int ptInd2 = ptInd + 1; ptInd2 < 3; ptInd2++)
				{
					VectorXd fcol = C.col(ptInd) - C.col(ptInd2);
					VectorXd gcol = BPans.col(ptInd) - BPans.col(ptInd2);
					F.col(ind) = fcol;
					G.col(ind) = gcol;
					ind++;
				}
			Vector3d cross_fac_1(F.col(0));
			Vector3d cross_fac_2(F.col(1));
			F.col(2) = cross_fac_1.cross(cross_fac_2);		//overwrite col(2)?
			cross_fac_1 = G.col(0);
			cross_fac_2 = G.col(1);
			G.col(2) = cross_fac_1.cross(cross_fac_2);
			R = G * F.inverse();						//divide
			t = BPans.col(0) - R * C.col(0);
		}
		Ta[i].block(0, 0, R.rows(), R.cols()) = R;
		Ta[i].col(R.cols()) = t;
		Tfa(i) = fest;
		f = fest;
		MatrixXd temp = MatrixXd::Constant(P.rows() + 1, P.cols(), 1);
		temp.block(0, 0, P.rows(), P.cols()) = P;
		MatrixXd Pm = Ta[i] * temp;
		resrate(i) = 0;
		for (int ptInd = 0; ptInd < Pm.cols(); ptInd++)
			if (Pm(2, ptInd) < 0)
				resrate(i)++;
		if (resrate(i) <= minrate)
		{
			minrate = resrate(i);
			//Ra, ta, festa
		}
		validCandNum++;
	}
	vector<int> goodInds;
	for (int i = 0; i < resrate.size(); i++)
		if (resrate(i) == minrate)
			goodInds.push_back(i);
	VectorXd res4good(goodInds.size());
	for (int i = 0; i < goodInds.size(); i++)
		res4good(i) = resCosts(goodInds.at(i));
	double min = res4good.minCoeff();
	int minInd = 0;
	for (int i = 0; i < res4good.size(); i++)
		if (res4good(i) == min)
		{
			minInd = i;
			break;
		}
	MatrixXd Tta = Ta[goodInds.at(minInd)];
	candsBest = cands.row(goodInds.at(minInd)).head(betaNum);
	if (validCandNum == 0)
	{
		R.resize(0, 0);
		t.resize(0);
		f = -1;
		return;
	}
	R = Tta.block(0, 0, 3, 3);
	t = Tta.col(3).head(3);
	f = Tfa(goodInds.at(minInd));
}


/*
* refine_f_res - function and structure that used in levmar
* data = pointer to structure
*/
typedef struct refine_f_res_struct
{
	Ref<const MatrixXd> L;
	Ref<const VectorXd> sol1;
} refine_f_res_struct;

void refine_f_res(double *x, double *res, int m, int n, void *data)
{
	refine_f_res_struct* tmp_data = (refine_f_res_struct*)data;
	double fGuess = x[0];
	VectorXd sol(2 * tmp_data->sol1.size());
	sol.head(tmp_data->sol1.size()) = tmp_data->sol1;
	sol.tail(tmp_data->sol1.size()) = tmp_data->sol1 * fGuess * fGuess;
	VectorXd tmp = tmp_data->L * sol;
	for (int i = 0; i < n; i++)
		res[i] = tmp(i);
}

/*
* refine_wb_res - function and structure that used in levmar
* data = pointer to structure
*/
typedef struct refine_wb_res_struct
{
	double fGuess;
	Ref<const MatrixXd> L;
	int betaNum;
} refine_wb_res_struct;

void refine_wb_res(double *x, double *res, int m, int n, void *data)
{
	refine_wb_res_struct* tmp_data = (refine_wb_res_struct*)data;
	int varNum = tmp_data->betaNum + tmp_data->betaNum*(tmp_data->betaNum - 1) / 2;
	VectorXd sol1 = VectorXd::Zero(varNum);
	int ind = tmp_data->betaNum;
	for (int i1 = 0; i1 < tmp_data->betaNum; i1++)
	{
		sol1(i1) = x[i1] * x[i1];
		for (int i2 = i1 + 1; i2 < tmp_data->betaNum; i2++)
		{
			sol1(ind) = x[i1] * x[i2];
			ind++;
		}
	}
	VectorXd sol(sol1.size() * 2);
	sol.head(sol1.size()) = sol1;
	sol.tail(sol1.size()) = tmp_data->fGuess * tmp_data->fGuess *sol1;
	VectorXd tmp = tmp_data->L * sol;
	for (int i = 0; i < n; i++)					//need to do it faster
		res[i] = tmp(i);
}

/*
* Empty answers
*/
void generateErrorReturn(MatrixXd &rest, VectorXd &test, double &fest, double &retVal)
{
	rest.resize(0, 0);
	test.resize(0);
	fest = -1;
	retVal = -1;
}

/* generateBetaSqsFromBetas, de2bi, makeBetaGuess3 depends on addToCands*/
void generateBetaSqsFromBetas(int betaNum, Ref<const VectorXd> x, Ref<VectorXd> sol1)
{
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

void de2bi(int i, int n, Ref<VectorXd> binVec)
{
	char s[128] = { 0 };
	_itoa_s(i, s, n + 1, 2);
	vector<int> temp;
	for (int j = 0; s[j] == '0' || s[j] == '1'; j++)
		temp.push_back(s[j] - '0');
	reverse(temp.begin(), temp.end());
	for (int j = 0; j < temp.size(); j++)
		binVec(j) = temp.at(j);
}

void makeBetaGuess3(Ref<const VectorXd> wguess, int betaNum, double fGuess, MatrixXd &betaGuess)
{
	vector<int> posInds;
	for (int i = 0; i < betaNum; i++)
		if (wguess(i) > 0)
			posInds.push_back(i);
	int posIndNum = posInds.size();
	if (posIndNum == 0)
		return;

	for (int i = 1; i < pow(2, posIndNum) + 1; i++)
	{
		VectorXd binVec = VectorXd::Zero(posIndNum);
		de2bi(i - 1, posIndNum, binVec);
		VectorXd row = VectorXd::Zero(betaNum);
		for (int varInd = 0; varInd < posIndNum; varInd++)
			if (wguess(posInds[varInd]) < 0)
				row(posInds[varInd]) = 0;
			else
				row(posInds[varInd]) = pow(-1, binVec(varInd)) * sqrt(wguess(posInds[varInd]));
		MatrixXd temp(betaGuess.rows() + 1, betaNum);
		if (betaGuess.cols() != 0)
			temp.block(0, 0, betaGuess.rows(), betaNum) = betaGuess;
		temp.row(betaGuess.rows()) = row;
		betaGuess.resize(0, 0);
		betaGuess = temp;
	}
	/* Rarely used, bad tests, end is not tested */
	if (betaNum > 1)
	{
		VectorXd dists = VectorXd::Zero(betaGuess.rows());
		for (int i = 0; i < betaGuess.rows(); i++)
		{
			int varNum = betaNum + betaNum*(betaNum - 1) / 2;
			VectorXd sol1 = VectorXd::Zero(varNum);
			generateBetaSqsFromBetas(betaNum, betaGuess.row(i).head(betaNum), sol1);
			VectorXd temp(2 * sol1.size());
			temp.head(sol1.size()) = sol1;
			temp.tail(sol1.size()) = fGuess * fGuess * sol1;
			temp -= wguess;																// don't sure that it works
			dists(i) = temp.norm();
		}
		double minVal = dists.minCoeff();
		int j = 0;
		MatrixXd temp(betaGuess.rows(), betaGuess.cols());
		for (int i = 0; i < dists.size(); i++)
			if (abs(dists(i) - minVal) < 1e-10)
			{
				temp.row(j) = betaGuess.row(i);
				j++;
			}
		temp.resize(j, betaGuess.cols());
		betaGuess.resize(0, 0);
		betaGuess = temp;
	}
}

/*
* Tested, but not all.
* Dependings are generateBetaSqsFromBetas, de2bi, makeBetaGuess3.
* Need to initialize MatrixXd cands(0, 0) before the first use.
*/
void addToCands(MatrixXd &cands, double alphaMy, Ref<const VectorXd> vMy, int N)
{
	if (alphaMy > 0)
	{
		double fGuess = sqrt(alphaMy);
		fGuess = max(fGuess, 50.0);
		VectorXd wguess(2 * vMy.size());
		wguess.head(vMy.size()) = vMy;
		wguess.tail(vMy.size()) = alphaMy * vMy;
		MatrixXd betaGuess(0, 0);
		makeBetaGuess3(wguess, N, fGuess, betaGuess);

		/* betaGuess.cols + 1 always is same? */
		MatrixXd temp = cands;
		int cols;
		if (temp.cols() == 0)
			cols = betaGuess.cols() + 1;
		else
			cols = temp.cols();
		cands.resize(temp.rows() + betaGuess.rows(), cols);
		cands.block(0, 0, temp.rows(), temp.cols()) = temp;
		cands.block(temp.rows(), 0, betaGuess.rows(), betaGuess.cols()) = betaGuess;
		VectorXd VecfGuess = VectorXd::Constant(betaGuess.rows(), fGuess);
		cands.col(cands.cols() - 1).tail(betaGuess.rows()) = VecfGuess;
	}
}

/*
* vMy - same dimension as x1 and x2
* tested
*/
void alphaFormula(Ref<const VectorXd> x1, Ref<const VectorXd> x2, VectorXd &vMy, double &alphaMy)
{
	VectorXd tmp = x1.transpose() * x2;
	double cosG = tmp(0) / x1.norm() / x2.norm();

	double sinG;
	if (1 - cosG * cosG < 0)
		sinG = 0;
	else
		sinG = sqrt(1 - cosG * cosG);

	double norm_x2_quad = x2.norm() * x2.norm();
	double norm_x1_quad = x1.norm() * x1.norm();
	double ang = 0.5 * atan2(norm_x2_quad * 2 * sinG * cosG, norm_x1_quad + norm_x2_quad * (2 * cosG * cosG - 1));

	tmp = x2.transpose() * x1;
	VectorXd x2o = x2 - tmp(0) / norm_x1_quad * x1;
	if (x2o.norm() > 0)
		x2o = x2o / x2o.norm();

	VectorXd vNormed = x1 / x1.norm() * cos(ang) + x2o * sin(ang);
	tmp = x1.transpose() * vNormed;
	vMy = tmp(0) * vNormed;
	tmp = x2.transpose() * vMy;
	alphaMy = tmp(0) / (vMy.norm() * vMy.norm());
}

/*
* Pseudoinverse matrix
* tested
*/
template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon)
{
	JacobiSVD< _Matrix_Type_ > svd(a, ComputeThinU | ComputeThinV);
	double tolerance = epsilon * max(a.cols(), a.rows())*svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

/* didn't test: begin */
void formBasePoints(Ref<const MatrixXd> Ker, int tarPtNum, vector<MatrixXd> &BP)
{
	for (int i = 0; i < Ker.cols(); i++)
	{
		BP[i] = MatrixXd::Zero(3, tarPtNum);

		MatrixXd tmp = Ker.col(i);
		tmp.resize(3, tarPtNum);
		BP[i] = tmp;
	}
}

void formLR(vector<MatrixXd> &BP, Ref<const MatrixXd> C, int N, Ref<MatrixXd> L, Ref<VectorXd> R)
{
	int cnt = 1;
	int maxptnum = BP[0].cols();

	for (int i = 0; i < maxptnum - 1; i++)
	{
		for (int j = i + 1; j < maxptnum; j++)
		{
			int colCnt = 1;
			for (int bpInd = 0; bpInd < N; bpInd++)
			{
				MatrixXd tmp = BP[bpInd].block(0, i, 2, 1) - BP[bpInd].block(0, j, 2, 1);
				double norm = tmp.norm();
				L(cnt, colCnt) = norm*norm;
				colCnt++;
			}
			for (int bpInd1 = 0; bpInd1 < N; bpInd1++)
			{
				for (int bpInd2 = bpInd1 + 1; bpInd2 < N; bpInd2++)
				{
					MatrixXd tmp1 = BP[bpInd1].block(0, i, 2, 1) - BP[bpInd1].block(0, j, 2, 1);
					MatrixXd tmp2 = BP[bpInd2].block(0, i, 2, 1) - BP[bpInd2].block(0, j, 2, 1);
					tmp1 = 2 * tmp1.transpose()*tmp2;
					L(cnt, colCnt) = tmp1(0, 0);
					colCnt++;
				}
			}

			for (int bpInd = 0; bpInd < N; bpInd++)
			{
				double tmp = BP[bpInd](2, i) - BP[bpInd](2, j);
				L(cnt, colCnt) = tmp*tmp;
				colCnt++;
			}
			for (int bpInd1 = 0; bpInd1 < N; bpInd1++)
			{
				for (int bpInd2 = bpInd1 + 1; bpInd2 < N; bpInd2++)
				{
					double tmp1 = BP[bpInd1](2, i) - BP[bpInd1](2, j);
					double tmp2 = BP[bpInd2](2, i) - BP[bpInd2](2, j);
					tmp1 = 2 * tmp1*tmp2;
					L(cnt, colCnt) = tmp1;
					colCnt++;
				}
			}

			MatrixXd tmp = C.col(i) - C.col(j);
			double norm = tmp.norm();
			R(cnt) = norm * norm;
			cnt++;
		}
	}
}

void formLRRegularized(vector<MatrixXd> &BP, Ref<const MatrixXd> C, int N, double alpha, double beta,
	Ref<const VectorXd> eigvals, int ptNum, Ref<MatrixXd> L, Ref<VectorXd> R)
{
	formLR(BP, C, N, L, R);

	int varNum = N + N*(N - 1) / 2;
	MatrixXd L2 = MatrixXd::Zero(2 * varNum, L.cols());
	VectorXd R2 = VectorXd::Zero(2 * varNum);

	int rowInd = 0;
	for (int varInd = 0; varInd < N; varInd++)
	{
		L2(rowInd, varInd) = alpha*eigvals(varInd)*eigvals(varInd);
		rowInd++;
	}

	int colInd = 0;
	for (int i1 = 0; i1 < N; i1++)
		for (int i2 = i1 + 1; i2 < N; i2++)
		{
			L2(rowInd, colInd) = alpha * abs(eigvals(i1) * eigvals(i2));
			rowInd++;
			colInd++;
		}

	if (alpha > 1e-10)
	{
		MatrixXd Ltmp(L.rows() + L2.rows(), L.cols());
		VectorXd Rtmp(R.size() + R2.size());

		Ltmp.block(0, 0, L.rows(), L.cols()) = L;
		Ltmp.block(L.rows(), 0, L2.rows(), L2.cols()) = L2;
		L = Ltmp;

		for (int i = 0; i < R.size(); i++)
			Rtmp(i) = R(i);
		for (int i = 0; i < R2.size(); i++)
			Rtmp(i + R.size()) = R2(i);
		R = Rtmp;
	}
}

void pnp1(Ref<const MatrixXd> C, Ref<const MatrixXd> V, Ref<const MatrixXd> D, int tarPtNum,
	Ref<const VectorXd> Cf, int fans, Ref<const MatrixXd> pts, int isFast, MatrixXd &rest, VectorXd &test, double &fest, double &retVal)//rest test fest retVal
{
	int N = 1;
	int varNum = N + (N*(N - 1)) / 2;                //????? later varNum = 2 * varNum

	MatrixXd Ker = MatrixXd::Zero(3 * tarPtNum, N);

	for (int i = 0; i < N; i++)
		Ker.col(i) = V.col(3 * tarPtNum - i - 1);   //index 1->0, need to be tested, changed ...-(i-1)

	VectorXd eigDiag = VectorXd::Zero(12);
	for (int i = 0; i < D.cols(); i++)
		eigDiag(i) = D(i, i);

	VectorXd eigvals = VectorXd::Zero(tarPtNum);

	for (int i = 0; i < 3 * tarPtNum; i++)
		eigvals(i) = eigDiag(3 * tarPtNum - i - 1);

	vector<MatrixXd> BP(Ker.cols());
	formBasePoints(Ker, tarPtNum, BP);

	double alpha = 0, beta = 0;
	if (tarPtNum == 4)
	{
		alpha = 0.00001;
		beta = alpha / (50 * 50);
	}

	int maxptnum = BP[0].cols();
	int eqnum = maxptnum*(maxptnum - 1) / 2;
	int varNum_LR = 2 * (N + N*(N - 1) / 2);                   //again varNum inside formLRRegularized -> varNum_LR
	MatrixXd L = MatrixXd::Zero(eqnum, varNum_LR);
	VectorXd R = VectorXd::Zero(eqnum);
	formLRRegularized(BP, C, N, alpha, beta, eigvals, pts.cols(), L, R);

	MatrixXd tmp = L.transpose() * L;
	FullPivLU<MatrixXd> lu(tmp);
	int rankBetaSqs = lu.rank();
	VectorXd betaSqs = pseudoInverse(tmp) * L.transpose() * R;

	JacobiSVD<MatrixXd> svd(L, ComputeFullV);
	MatrixXd VL = svd.matrixV();
	MatrixXd kerVect = VL.block(0, VL.cols() - (VL.rows() - rankBetaSqs), VL.rows(), VL.rows() - rankBetaSqs); //??? seems right

	MatrixXd cands(0, 0);
	int errSign = 0;
	double alphaMy = -1;
	VectorXd vMy = VectorXd::Zero(N);
	if (rankBetaSqs == betaSqs.size())
	{
		alphaFormula(betaSqs.head(varNum), betaSqs.segment(varNum, varNum), vMy, alphaMy);
		addToCands(cands, alphaMy, vMy, N);
	}

	while ((alphaMy < 0 || (vMy.head(N).array() <= 0).all() || errSign == 1) && rankBetaSqs >= varNum)
	{
		if (rankBetaSqs == kerVect.rows())
			rankBetaSqs--;
		kerVect = VL.block(0, VL.cols() - (VL.rows() - rankBetaSqs), VL.rows(), VL.rows() - rankBetaSqs); //again
																										  //adjustInLKer!
		adjustInLKer(betaSqs, kerVect, N, L, isFast, vMy, alphaMy, errSign);

		if (alphaMy > 0)
			addToCands(cands, alphaMy, vMy, N);

		rankBetaSqs--;
	}
	if (alphaMy < 0)
	{
		generateErrorReturn(rest, test, fest, retVal);
		return;
	}

	VectorXd resCosts = VectorXd::Zero(cands.rows());
	for (int candsInd = 0; candsInd < cands.rows(); candsInd++)
	{
		int m = 1;							//size of x for levmar
		double x[1] = { cands(candsInd, 0) };
		double fGuess = cands(candsInd, 1);
		double fGuess_lm[1] = { fGuess };
		int n = R.size();
		double* R_lm = new double[n];
		for (int i = 0; i < n; i++)
			R_lm[i] = R(i);
		int itmax = 200;
		double opts[LM_OPTS_SZ] = { 1E-01, 1E-15, 1E-15, 1E-20, 1E-15 }, info[LM_INFO_SZ];
		int ret;
		if (isFast == 0)
		{
			refine_wb_res_struct data = { fGuess, L, N };
			//levmar
			ret = dlevmar_dif(refine_wb_res, x, R_lm, m, n, itmax, opts, info, NULL, NULL, (void*)&data);
		}
		else
		{
			ret = -1;
		}

		VectorXd sol1(1);
		sol1(0) = x[0] * x[0];
		double xpopt = x[0];

		refine_f_res_struct data = { L, sol1 };
		if (isFast == 0)
		{
			ret = dlevmar_dif(refine_f_res, fGuess_lm, R_lm, m, n, itmax, opts, info, NULL, NULL, (void*)&data);
		}
		else
		{
			double* res = new double[n];
			refine_f_res(fGuess_lm, res, m, n, (void*)&data);
			VectorXd tmp(n);
			for (int i = 0; i < n; i++)
				tmp(i) = res[i] - R(i);
			info[0] = tmp.norm();
			info[1] = info[0];
			ret = 1;				//maybe -1???
			delete[] res;
		}
		double poptf = fGuess_lm[0];

		if (ret >= 0)
		{
			VectorXd tmp(2);
			tmp << xpopt, poptf;
			cands.row(candsInd) = tmp;
			resCosts(candsInd) = info[1];
		}
		else
		{
			resCosts(candsInd) = info[0];   //why? info[0]=info[1]
		}
		delete[] R_lm;
	}
	MatrixXd R_res(0, 0);
	VectorXd t_res(0);
	double f;
	VectorXd candsBest(0);
	vote(cands, BP, C, resCosts, tarPtNum, pts, R_res, t_res, f, candsBest);
	if (f < 0)
	{
		rest.resize(0, 0);
		test.resize(0);
		fest = -1;
		retVal = -1;
	}
	else
	{
		rest = R_res;
		test = t_res;
		fest = f;
		retVal = 1;
	}
}

void pnpNd1204Multi(Ref<const MatrixXd> C, Ref<const MatrixXd> V, Ref<const MatrixXd> D, int tarPtNum,
	Ref<const VectorXd> Cf, int fans, int N, int stopN, Ref<const MatrixXd> pts, int isFast, MatrixXd &rest, VectorXd &test, double &fest, double &retVal)//rest test fest retVal
{
	int ptDim = 3;
	int kerVectDim = ptDim * tarPtNum;
	int varNum = N + N*(N - 1) / 2;

	MatrixXd Ker = MatrixXd::Zero(kerVectDim, N);
	for (int i = 0; i < N; i++)
	{
		Ker.col(i) = V.col(kerVectDim - i - 1);		//different?
	}

	VectorXd eigDiag = VectorXd::Zero(kerVectDim);
	for (int i = 0; i < D.cols(); i++)
		eigDiag(i) = D(i, i);

	VectorXd eigvals = VectorXd::Zero(tarPtNum);

	for (int i = 0; i < kerVectDim; i++)
		eigvals(i) = eigDiag(kerVectDim - i - 1);

	vector<MatrixXd> BP(Ker.cols());
	formBasePoints(Ker, tarPtNum, BP);

	double alpha = 0, beta = 0;
	if (tarPtNum == 4)
	{
		alpha = 0.00001;
		beta = alpha / (50 * 50);
	}

	int maxptnum = BP[0].cols();
	int eqnum = maxptnum*(maxptnum - 1) / 2;
	int varNum_LR = 2 * (N + N*(N - 1) / 2);                   //again varNum inside formLRRegularized -> varNum_LR
	MatrixXd L = MatrixXd::Zero(eqnum, varNum_LR);
	VectorXd R = VectorXd::Zero(eqnum);
	formLRRegularized(BP, C, N, alpha, beta, eigvals, pts.cols(), L, R);

	MatrixXd tmp = L.transpose() * L;
	FullPivLU<MatrixXd> lu(tmp);
	int rankBetaSqs = lu.rank();
	VectorXd betaSqs = pseudoInverse(tmp) * L.transpose() * R;

	JacobiSVD<MatrixXd> svd(L, ComputeFullV);
	MatrixXd VL = svd.matrixV();
	MatrixXd kerVect = VL.block(0, VL.cols() - (VL.rows() - rankBetaSqs), VL.rows(), VL.rows() - rankBetaSqs); //??? seems right

	MatrixXd cands(0, 0);
	int errSign = 0;
	double alphaMy = -1;

	while (cands.cols() == 0 && rankBetaSqs >= varNum)
	{
		if (rankBetaSqs == kerVect.rows())
			rankBetaSqs--;
		kerVect = VL.block(0, VL.cols() - (VL.rows() - rankBetaSqs), VL.rows(), VL.rows() - rankBetaSqs); //again
		MatrixXd xkNs(0, 0);
		adjustInLKerMulti(betaSqs, kerVect, N, L, isFast, xkNs);
		int xknLen = xkNs.rows() / 2;
		for (int i = 0; i < xkNs.cols(); i++)
		{
			VectorXd vMy(0);
			alphaFormula(xkNs.col(i).head(xknLen), xkNs.col(i).segment(xknLen, xkNs.rows() - xknLen), vMy, alphaMy);
			if (alphaMy > 0)
				addToCands(cands, alphaMy, vMy, N);
		}
		rankBetaSqs--;
	}

	if (cands.size() == 0)
	{
		generateErrorReturn(rest, test, fest, retVal);
		return;
	}

	VectorXd resCosts = VectorXd::Zero(cands.rows());
	vector<int> badInd;
	for (int candsInd = 0; candsInd < cands.rows(); candsInd++)
	{
		int m = N;
		double *x = new double[m];
		for (int i = 0; i < m; i++)
			x[i] = cands(candsInd, i);
		double fGuess = cands(candsInd, N);
		double fGuess_lm[1] = { fGuess };
		int n = R.size();
		double* R_lm = new double[n];
		for (int i = 0; i < n; i++)
			R_lm[i] = R(i);
		int itmax = 200;
		double opts[LM_OPTS_SZ] = { 1E-01, 1E-15, 1E-15, 1E-20, 1E-15 }, info[LM_INFO_SZ];
		int ret;

		if (isFast == 0)
		{
			refine_wb_res_struct data = { fGuess, L, N };
			ret = dlevmar_dif(refine_wb_res, x, R_lm, m, n, itmax, opts, info, NULL, NULL, (void*)&data);
		}
		else
		{
			info[0] = 0;
			info[1] = 0;
			ret = -1;
		}

		VectorXd sol1(0);
		VectorXd xpopt(m);
		double resVal;
		for (int i = 0; i < m; i++)
			xpopt(i) = x[i];

		generateBetaSqsFromBetas(N, xpopt, sol1);
		if (ret >= 0)
			resVal = info[1];
		else
			resVal = info[0];

		resCosts(candsInd) = resVal;
		refine_f_res_struct data = { L, sol1 };
		m = 1;
		if (isFast == 0)
		{
			ret = dlevmar_dif(refine_f_res, fGuess_lm, R_lm, m, n, itmax, opts, info, NULL, NULL, (void*)&data);
		}
		else
		{
			double* res = new double[n];
			refine_f_res(fGuess_lm, res, m, n, (void*)&data);
			VectorXd tmp(n);
			for (int i = 0; i < n; i++)
				tmp(i) = res[i] - R(i);
			info[0] = tmp.norm();
			info[1] = info[0];
			ret = 1;				//maybe -1???
			delete[] res;
		}
		double poptf = fGuess_lm[0];
		if (ret >= 0)
		{
			if (poptf < 50)
				badInd.push_back(candsInd);
			resCosts(candsInd) = info[1];
		}
		else
		{
			resCosts(candsInd) = info[0];
		}
		VectorXd tmp(xpopt.size() + 1);
		tmp.head(xpopt.size()) = xpopt;
		tmp(xpopt.size()) = poptf;
		cands.row(candsInd).head(N + 1) = tmp;

		delete[] R_lm;
		delete[] x;
	}

	MatrixXd R_res(0, 0);
	VectorXd t_res(0);
	double f;
	VectorXd candsBest(0);
	vote(cands, BP, C, resCosts, tarPtNum, pts, R_res, t_res, f, candsBest);
	if (f < 0 && N < stopN)
	{
		generateErrorReturn(rest, test, fest, retVal);
		return;
	}
	else
	{
		if (f < 0)
			f = 50;
		rest = R_res;
		test = t_res;
		fest = f;
		retVal = 1;
	}
}
