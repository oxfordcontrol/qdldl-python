#include "qdldl.hpp"

#include "qdldl/include/qdldl_configure.h"
using namespace qdldl;



Solver::Solver(QDLDL_int n, QDLDL_int * Ap, QDLDL_int *Ai, QDLDL_float * Ax){

	// Dimension
	nx = n;
	nnz = Ap[nx];

	// Elimination tree
	etree  = new QDLDL_int[n];
	Lnz = new QDLDL_int[n];

	// L factors
	Lp = new QDLDL_int[n + 1];

	// D
	D = new QDLDL_float[n];
	Dinv = new QDLDL_float[n];

	// Workspace
	iwork = new QDLDL_int[3 * n];
	bwork = new QDLDL_bool[n];
	fwork = new QDLDL_float[n];

	// Permutation
	P = new QDLDL_int[n];
	Pinv = new QDLDL_int[n];

	// Permutation
#ifdef DLONG
	QDLDL_int amd_status = amd_l_order(nx, Ap, Ai, P, NULL, NULL);
#else
	QDLDL_int amd_status = amd_order(nx, Ap, Ai, P, NULL, NULL);
#endif
	if (amd_status < 0)
		throw std::runtime_error(std::string("Error in AMD computation ") + std::to_string(amd_status));

	pinv(P, Pinv, n); // Compute inverse permutation

	// Allocate elements of A permuted
	Aperm_p = new QDLDL_int[n+1];
	Aperm_i = new QDLDL_int[nnz];
	Aperm_x = new QDLDL_float[nnz];
	A2Aperm = new QDLDL_int[nnz];
	QDLDL_int * work_perm = new QDLDL_int[n]();  // Initialize to 0

	// Permute A
	symperm(n, Ap, Ai, Ax, Aperm_p, Aperm_i, Aperm_x, Pinv, A2Aperm, work_perm);

	// Compute elimination tree
    int sum_Lnz = QDLDL_etree(n, Aperm_p, Aperm_i, iwork, Lnz, etree);

	if (sum_Lnz < 0)
		throw std::runtime_error(std::string("Error in computing elimination tree. Matrix not properly upper-triangular, sum_Lnz = ") + std::to_string(sum_Lnz));

	// Allocate factor
	Li = new QDLDL_int[sum_Lnz];
	Lx = new QDLDL_float[sum_Lnz];


	// Compute numeric factorization
	int factor_status = QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			Lp, Li, Lx,
			D, Dinv, Lnz,
			etree, bwork, iwork, fwork);

	if (factor_status < 0){
		throw std::runtime_error(std::string("Error in matric factorization. Input matrix is not quasi-definite, factor_status = ") + std::to_string(factor_status));
	}


    // Delete permutaton workspace
	delete [] work_perm;

}




QDLDL_float * Solver::solve(QDLDL_float * b){

	auto * x = new QDLDL_float[nx];
	auto work = new QDLDL_float[nx];

    permute_x(nx, work, b, P);
    QDLDL_solve(nx, Lp, Li, Lx, Dinv, work);
    permutet_x(nx, x, work, P);

	delete [] work;
	return x;

}



void Solver::update(QDLDL_float * Anew_x){

	// Update matrix
	update_A(nnz, Aperm_x, Anew_x, A2Aperm);

	// Compute numeric factorization
    QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			     Lp, Li, Lx,
				 D, Dinv, Lnz,
				 etree, bwork, iwork, fwork);

}

Solver::~Solver(){

	delete [] Lp;
	delete [] Li;
	delete [] Lx;
	delete [] D;
	delete [] Dinv;
	delete [] P;
	delete [] Pinv;
	delete [] etree;
	delete [] Lnz;
	delete [] iwork;
	delete [] bwork;
	delete [] fwork;
	delete [] Aperm_p;
	delete [] Aperm_i;
	delete [] Aperm_x;
	delete [] A2Aperm;

}
