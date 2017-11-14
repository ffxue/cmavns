/**
 * CMA-VNS2: Covariance Matrix Adaptation Variable neighborhood search
 * Author: Frank F. Xue*# <xuef@hku.hk; fanxue@outlook.com>
 *         Geoffrey Q.P. Shen* <bsqpshen@polyu.edu.hk>
 *         * The Hong Kong Polytechnic University
 *         # The University of Hong Kong
 * This file is for CBBOC2016 <http://web.mst.edu/~tauritzd/CBBOC/>.
 *
 * CMA-VNS2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CMA-VNS2.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * The authors appreciate the libcmaes <https://github.com/beniz/libcmaes>.
 */
 
 /**
 * Please cite the following paper if using this software.
 * 
 * Xue, F., & Shen, G. Q. (2017). Design of an efficient hyper-heuristic 
 * algorithm CMA-VNS for combinatorial black-box optimization problems. 
 * In Proceedings of the Genetic and Evolutionary Computation Conference 
 * Companion, pp. 1157-1162. July 15 - 19, 2017, Berlin, ACM. 
 * doi: 10.1145/3067695.3082054
 */
 
#ifndef CMAVNS2_COMPETITOR_HPP
#define CMAVNS2_COMPETITOR_HPP

//////////////////////////////////////////////////////////////////////

#include "cbboc/CBBOCUtil.hpp"
#include "cbboc/Competitor.hpp"
#include "cbboc/ObjectiveFn.hpp"
#include "cbboc/ProblemClass.hpp"
#include "cbboc/TrainingCategory.hpp"
#include "cbboc/RNG.hpp"

#include "CMAVNS2P1.hpp"
#include "CMAVNS2P4.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "libcmaes/cmaes.h"
#include "libcmaes/esoptimizer.h"
#include "libcmaes/cmastrategy.h"
#include "libcmaes/ipopcmastrategy.h"
#include "libcmaes/bipopcmastrategy.h"

using namespace libcmaes;

//////////////////////////////////////////////////////////////////////

/**
 * CMA-VNS2: Covariance Matrix Adaptation Variable Neighborhood Search 
 * Ver. 2 (for CBBOC 2016)
 *
 * The main class.
 *
 * CMA-VNS2 is a hyperheuristic which employs five profiles of 
 * CMA-ES evolution and iterated local search. The reason of employing 
 * the CMA-ES is the efficient estimation of variables in CMA-ES. Hence
 * the local search, variable neighborhood search in this context, can
 * be facilitated against expensive (and limited) evaluations.
 *
 * Three profiles in this .hpp file; two others in CMAVNS2P1.hpp and 
 * CMAVNS2P4.hpp.
 *
 * i) When using constructor CMAVNS2Competitor() or CMAVNS2Competitor
 * (TrainingCategory::NONE), the parameters of the hyperheuristic were 
 * as trained beforehand.
 * ii) When using constructor CMAVNS2Competitor(TrainingCategory::SHORT),
 * a few main parameters will be tested one by one, and the best 
 * profile is learned.
 * iii) When using constructor CMAVNS2Competitor(TrainingCategory::LONG),
 * more iterations will be involved in the testing.
 *
 * This CMAVNS2Competitor.hpp file implements three profiles: 
 * P2_CMAVNS_D, P3_CMAVNS_W, and P5_ICMA.
 */
 
class CMAVNS2Competitor : public Competitor 
{
private:
	TrainingCategory my_category = TrainingCategory::NONE;
	ObjectiveFn *objfunc;
	
	// shared global mem
	std::unordered_map<std::vector<bool>, double> history;	// cache of evaluations
	std::vector<std::vector<bool>> halloffame;			// elites
	std::vector< int > hof_bb;							// backbone of halloffame
	std::unordered_set<std::vector<bool>> tabu;			// exploited by 1-SWAP heuristic
	std::unordered_set<std::vector<bool>> tabu_hof_bb;	// exploited by icma
	std::vector< int > crosspoints;						// mem for crossover points
	
	// shared variables
	int rand_seed = 2016;
	int evaluations = 0;
	int eval_offset = 0;
	int cma_hof_size = 0;
	int hof_bb_size = 0;
	double bestSoFar = 0.0;
	double MIN_OBJV = 1.0E+8;
	double MAX_OBJV = -1.0;
	double icma_iter_sum = 0.0;
	
	// const flags of CMAVNS2
	const int HH_PROFILE_NUM = 5;
	const int P1_CMAVNS = 1;
	const int P2_CMAVNS_D = 2;  // implemented in this file
	const int P3_CMAVNS_W = 3;  // implemented in this file
	const int P4_CMAVNS_VARIANT = 4;
	const int P5_ICMA = 5;      // implemented in this file
	const int SWAP1_LOW = 0;
	const int SWAP1_MEDIUM = 1;
	const int SWAP1_HIGH = 2;
	
	// params of CMAVNS2
	int 	PARAM_CMA_ALG = 11;
	int 	PARAM_CMA_LAMDA = -1;
	double 	PARAM_CMA_RUN_UNTIL = 0.55;
	double 	PARAM_CMA_RUN_UNTIL_ICMA = 0.55;
	double 	PARAM_AA_TOLERANCE = 0.0;
	double 	PARAM_AA_ELITE = 1.0;
	int 	PARAM_BB_CX_ALG = 0;
	int 	PARAM_BB_CX_TRIALS = 20;
	int 	PARAM_BB_CX_NUM_POINTS = -1;
	double 	PARAM_VNS_SCALE = 1.0;
	int 	PARAM_MIN_HAMMING_DIST = 4;
	int 	PARAM_HOF_MIN_SIZE = 3;
	double 	PARAM_HOF_BB_MAX_SIZE = 0.95;
	double 	PARAM_HOF_SHRINK_THRESHOLD = 0.95;
	int 	PARAM_HH_TOP_PROFILE = P3_CMAVNS_W;
	int 	PARAM_HH_1DS_DEPTH = SWAP1_MEDIUM;
	int 	PARAM_HH_1DS_DEPTH_ICMA = SWAP1_MEDIUM;
	int 	PARAM_HH_1DS_DEPTH_2015 = SWAP1_MEDIUM;
	
	// mem for apriori interpolation points
	std::vector<double> apriori_dim;
	std::vector<int> apriori_lamda_hc;
	std::vector<int> apriori_lamda_lc;
	std::vector<int> apriori_bbtrials_hc;
	std::vector<int> apriori_bbtrials_lc;
	std::vector<int> apriori_cxnum_mc;
	std::vector<int> apriori_cxnum_lc;
	std::vector<int> apriori_hdist_hc;
	std::vector<int> apriori_hdist_mc;
	std::vector<int> apriori_hdist_lc;
	std::vector<int> apriori_rununtil_hc;
	std::vector<int> apriori_rununtil_mc;
	std::vector<int> apriori_rununtil_lc;
	std::vector<int> apriori_rununtil_b_hc;
	std::vector<int> apriori_rununtil_b_mc;
	std::vector<int> apriori_rununtil_b_lc;
	std::vector<int> apriori_rununtil_icma_hc;
	std::vector<int> apriori_rununtil_icma_mc;
	std::vector<int> apriori_rununtil_icma_lc;
	
	// marks for training
	bool trainingMode = false;
	double learnedESratio = -1.0;
	double learnedLamdaCoeff = -1.0;
	double learnedTrialsCoeff = -1.0;
	int LEARNED_PROFILE = -1;
	
	// for self_testing
	bool param_preset = false;
	double* preset_params = new double[30];
	
public:
	double TEST_RESULT = 0.0;
	
	CMAVNS2Competitor() { objfunc = 0; load_apriori_data();	}
	
	CMAVNS2Competitor(TrainingCategory category) : CMAVNS2Competitor() {my_category = category;}
	
	CMAVNS2Competitor(bool selfTraining) : CMAVNS2Competitor() {trainingMode = selfTraining;}

	virtual ~CMAVNS2Competitor() { reset_mem(0); }

	virtual TrainingCategory getTrainingCategory() const { return my_category; }

  /**
  * test(func, time) is the main entry call function of CMA-VNS2
  *
  * In this function, a profile (a variant of CMA-VNS'15 algorithm) is selected
  * first. The parameters of each profile are determined by the profile itself.
  * There are five profiles: P1_CMAVNS, P2_CMAVNS_D, P3_CMAVNS_W, , P5_ICMA
  */
	virtual void test( ObjectiveFn& testCase, long long maxTimeInMilliseconds )
	{
	  // Feature 1: dimension
		const int dim = testCase.getNumGenes();
		// Feature 2: a discount of evaluation (high = expensive problem, low = cheap problem)
		const double EvDiscount = 1.0 * dim * dim / testCase.getRemainingEvaluations();
		
		objfunc = &testCase;
		// consume one evaluation to create a tester 
		// Feature 3: Approximate function value contribution per dimension
		std::vector<bool> tester = random_bitvector(dim);
		double rand_v = getValue(tester);
		// set up all parameters (P1, P2, P5) with above 3 features
		setup_apriori_params(dim, EvDiscount, rand_v > 1.5 * dim);
		
		// select the profile ID. If exists any succesfull training, use the trained
		int profileID = my_category != TrainingCategory::NONE && LEARNED_PROFILE >= 0 ? LEARNED_PROFILE : PARAM_HH_TOP_PROFILE;
		
		// execute the selected algorithmic profile
		if (profileID == P1_CMAVNS)
		{ 
		  // the 2015 version of CMA-VNS
			CMAVNS2P1 employee;
			employee.test(testCase, maxTimeInMilliseconds);
		}
		else if (profileID == P2_CMAVNS_D)
		{
		  // An intensification of the 2015 version of CMA-VNS (deeper search)
			cmavns_d(testCase, maxTimeInMilliseconds, dim, EvDiscount, testCase.getRemainingEvaluations());
		}
		else if (profileID == P3_CMAVNS_W)
		{
		  // An diversification of P2_CMAVNS_D (wider search)
			cmavns_w(testCase, maxTimeInMilliseconds, dim, EvDiscount, testCase.getRemainingEvaluations());
		}
		else if (profileID == P4_CMAVNS_VARIANT)
		{
		  // a redesigned variant of CMA-VNS 2015
			CMAVNS2P4 employee;
			employee.test(testCase, maxTimeInMilliseconds);
		}
		else
		{
		  // Iterated CMAES, otherwise
			icma(testCase, maxTimeInMilliseconds, dim, EvDiscount, testCase.getRemainingEvaluations());
		}
	}
	
	/**
	* train(func[], time) is the online training function. Not executed for 
	* "TrainingCategory::NONE"
	*
	*/
	virtual void train( std::vector<ObjectiveFn>& trainingSet, long long maxTimeInMilliseconds ) 
	{
		LEARNED_PROFILE = -1;
 		int sharedEval = trainingSet[0].getRemainingEvaluations();
		int singleEval = sharedEval / (my_category == TrainingCategory::LONG ? 10 : 1) / trainingSet.size();
		const int dim = trainingSet[0].getNumGenes();
		const double EvDiscount = 1.0 * dim * dim / singleEval;
		int SAMPLE_SIZE = my_category == TrainingCategory::LONG ? 60 : 12;
		double bestMeans[10];
		double testMeans[10];
		std::unordered_set<int> candidates;
		candidates.insert(P1_CMAVNS);
		candidates.insert(P3_CMAVNS_W);
		candidates.insert(P4_CMAVNS_VARIANT);
		if (my_category == TrainingCategory::LONG)
		{
			candidates.insert(P2_CMAVNS_D);
			candidates.insert(P5_ICMA);
		}
		objfunc = &trainingSet[0];
		
		// compare profiles (or hyperheusitics, aka HH)
		std::vector<bool> tester = random_bitvector(dim);
		double rand_v = getValue(tester);
		setup_apriori_params(dim, EvDiscount, rand_v > 1.5 * dim);
		const int DEFAULT_PROFILE = PARAM_HH_TOP_PROFILE;
		for (int j = 0; j < HH_PROFILE_NUM; j++)
			bestMeans[j] = 0.0;
		int probID = rand()%trainingSet.size();
		for (size_t i = 0; i < SAMPLE_SIZE ; i++)
		{
			for (int j = 0; j < HH_PROFILE_NUM; j++)
			{
				if (candidates.find(j) == candidates.end() && j != DEFAULT_PROFILE)
				{
					testMeans[j] = 0.0;
					continue;
				}
				if (j < 3)
				{
					CMAVNS2Competitor t1;
					if (j == P3_CMAVNS_W)
						t1.cmavns_w(trainingSet[(probID+i)%trainingSet.size()], maxTimeInMilliseconds / trainingSet.size(), dim, EvDiscount, singleEval);
					else if (j == P2_CMAVNS_D)
						t1.cmavns_d(trainingSet[(probID+i)%trainingSet.size()], maxTimeInMilliseconds / trainingSet.size(), dim, EvDiscount, singleEval);
					else if (j == P5_ICMA)
						t1.icma(trainingSet[(probID+i)%trainingSet.size()], maxTimeInMilliseconds / trainingSet.size(), dim, EvDiscount, singleEval);
					if (trainingSet[0].getRemainingEvaluations() > 0)
						testMeans[j] = t1.TEST_RESULT;
				}
				else if (j == P4_CMAVNS_VARIANT)
				{
					CMAVNS2P4 emp;
					emp.run(trainingSet[(probID+i)%trainingSet.size()], maxTimeInMilliseconds / trainingSet.size(), dim, EvDiscount, singleEval, -1.0, 1.0, 1.0);
					if (trainingSet[0].getRemainingEvaluations() > 0)
						testMeans[j] = emp.TEST_RESULT;
				}
				else if (j == P1_CMAVNS)
				{
					CMAVNS2P1 emp;
					emp.run(trainingSet[(probID+i)%trainingSet.size()], maxTimeInMilliseconds / trainingSet.size(), dim, EvDiscount, singleEval);
					if (trainingSet[0].getRemainingEvaluations() > 0)
						testMeans[j] = emp.TEST_RESULT;
				}
			}
			if (trainingSet[0].getRemainingEvaluations() > 0)
			{
				for (int j = 0; j < HH_PROFILE_NUM; j++)
				{
				//	std::cout << testMeans[j] << " ";
					if (j == DEFAULT_PROFILE)
						continue;
					if (testMeans[j] > testMeans[DEFAULT_PROFILE] * 1.001)
						bestMeans[j] += 1.0;
					else if (testMeans[j] >= testMeans[DEFAULT_PROFILE])
						bestMeans[j] += 0.5;
				}
				//std::cout << std:: endl;
			}
		}
		double bestScore = -1.0;
		for (int j = 0; j < HH_PROFILE_NUM; j++)
		{
			if (bestMeans[j] > bestScore)
				bestScore = bestMeans[j];
		}
		if (bestScore > SAMPLE_SIZE * 0.5)
			for (int j = 0; j < HH_PROFILE_NUM; j++)
				if (bestMeans[(j+3)%HH_PROFILE_NUM] == bestScore)
				{
					LEARNED_PROFILE = (j+3)%HH_PROFILE_NUM;
					break;
				}
	}


	virtual void icma(ObjectiveFn& testCase, long long maxTimeMs, const int dim, const double EvDiscount, const int eval) 
	{
		srand(rand_seed++);
		const long long startTime = system_current_time_millis();
		evaluations = eval;
 		eval_offset = evaluations - testCase.getRemainingEvaluations();
		objfunc = &testCase;
		
		reset_mem(dim);
		
		double myEV = 1.0;
		double last_iter_sum = -1.0;
		long restarts = 0;
		do 
		{
			int mydim = dim - hof_bb_size;
			MIN_OBJV = 1.0E+8;
			MAX_OBJV = -1.0;
			icma_iter_sum = 0.0;
			std::vector<double> x0(mydim, 0.5);
			double lbounds[mydim],ubounds[mydim];
			for (int i=0;i<mydim;i++)
			{
				lbounds[i] = 0.0;
				ubounds[i] = 1.0;
			}
			GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,mydim); // genotype / phenotype transform associated to bounds.
			const double sigma = 0.5;
			CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0, sigma, PARAM_CMA_LAMDA, 0, gp); // -1 for automatically decided lambda
			cmaparams.set_seed(rand_seed++);
			cmaparams.set_max_fevals(remainingEvaluations());
			cmaparams.set_noisy();
			cmaparams.set_sep();
			cmaparams.set_algo(PARAM_CMA_ALG); // CMAES_DEFAULT, IPOP_CMAES, BIPOP_CMAES, aCMAES, aIPOP_CMAES, aBIPOP_CMAES, sepCMAES, sepIPOP_CMAES, sepBIPOP_CMAES, sepaCMAES, sepaIPOP_CMAES, sepaBIPOP_CMAES, VD_CMAES, VD_IPOP_CMAES, VD_BIPOP_CMAES 
			
			ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> optim(ifitfunc,cmaparams);
			optim.set_progress_func(CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>::_defaultPFunc);
				
			myEV*= PARAM_CMA_RUN_UNTIL_ICMA;
			while(!optim.stop() && remainingEvaluations() > evaluations * myEV)
			{
				dMat candidates = optim.ask();
				optim.eval(candidates);
				optim.tell();
				optim.inc_iter(); // important step: signals next iteration.
			}
			// followed by 1DS
			std::vector< bool > incumbent;
			double bestValue = 0.0;
			bool improved = false;
			for (auto hof : halloffame)
			{
				double tv = history.find(hof)->second;
				if (bestValue < tv && tv > (MAX_OBJV - MIN_OBJV) * 0.995 + MIN_OBJV && tabu.find(hof)==tabu.end())
				{
					incumbent = hof;
					bestValue = tv;
					improved = true;
				}
			}
			//std::cout << remainingEvaluations() << "_" << myEV << "_" << MAX_OBJV << " ";
			if (last_iter_sum == icma_iter_sum && !improved)
			{
				// double relax : jump from local optima
				double relaxed_size = dim * (1.0 - (2+restarts/10)*(1.0 - PARAM_HOF_BB_MAX_SIZE));
				while (hof_bb_size > relaxed_size)
				{
					size_t i = rand()%dim;
					if (hof_bb[i] >= 0)
					{
						hof_bb_size--;
						hof_bb[i] = -1;
					}
				}
				last_iter_sum = icma_iter_sum;
				restarts++;
				if (restarts >= 150)
					restarts = 10;
				continue;
			}
			last_iter_sum = icma_iter_sum;
			bestSoFar = bestValue;
			std::vector< std::vector< bool > > peers;
			while(improved && remainingEvaluations() > 0 )
			{
				double initV = bestValue;
				improved = false;
				tabu.insert(incumbent);
				std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent, false);
				for (size_t i=0; i<neighbors.size() && remainingEvaluations() > 0; i++)
				{
					std::vector< bool >& neighbor = neighbors[ i ];
					double value = getValue(neighbor);
					if(value > bestValue) 
					{
						// new best known
						improved = true;
						incumbent = neighbor;
						bestValue = value;
						if (value > bestSoFar)
						{
							bestSoFar = value;
							if (PARAM_HH_1DS_DEPTH_ICMA < SWAP1_HIGH)
								peers.clear();
						}
					}
					else if (value == bestValue && tabu.find(neighbor) == tabu.end())
						peers.push_back(neighbor);
					else if (PARAM_HH_1DS_DEPTH_ICMA > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
						peers.push_back(neighbor);
					if (value > (MAX_OBJV - MIN_OBJV) * 0.999 + MIN_OBJV)
						addHoF(neighbor);
				}
				neighbors.clear();
				
				neighbors = hamming1Neighbours(incumbent, true);
				for (size_t i=0; i<neighbors.size() && remainingEvaluations() > 0; i++)
				{
					std::vector< bool >& neighbor = neighbors[ i ];
					double value = getValue(neighbor);
					if(value > bestValue) 
					{
						// new best known
						improved = true;
						incumbent = neighbor;
						bestValue = value;
						if (value > bestSoFar)
						{
							bestSoFar = value;
							if (PARAM_HH_1DS_DEPTH_ICMA < SWAP1_HIGH)
								peers.clear();
						}
					}
					else if (value == bestValue && tabu.find(neighbor) == tabu.end())
						peers.push_back(neighbor);
					else if (PARAM_HH_1DS_DEPTH_ICMA > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
						peers.push_back(neighbor);
					if (value > (MAX_OBJV - MIN_OBJV) * 0.999 + MIN_OBJV)
						addHoF(neighbor);
				}
				neighbors.clear();
				if (!improved && peers.size() > 0)
				{
					int best_peer = best_in_vector(peers);
					if (best_peer >= 0)
					{
						incumbent = peers[best_peer];
						bestValue = history.find(incumbent)->second;
						peers.erase(peers.begin()+best_peer);
						improved = true;
					}
				}
			}
			hof_bb_size = find_hof_bb();
			
			for (size_t i = 0; i < halloffame.size(); i++)
				if(history.find(halloffame[i])->second > MAX_OBJV)
					MAX_OBJV = history.find(halloffame[i])->second;
			
			for (size_t i = 0; i < halloffame.size(); i++)
			{
				double min_req =  (MAX_OBJV- MIN_OBJV) *(remainingEvaluations() < evaluations/4 ? 0.99: 1.0) + MIN_OBJV;
				if(history.find(halloffame[i])->second < min_req)
					halloffame.erase(halloffame.begin()+(i--));
			}
			if (MAX_OBJV > TEST_RESULT)
				TEST_RESULT = MAX_OBJV;
		}
		while (remainingEvaluations() > 0);
	}
		
	virtual void cmavns_w(ObjectiveFn& testCase, long long maxTimeMs, const int dim, const double EvDiscount, const int eval) 
	{
		const long long startTime = system_current_time_millis();	
		// But the time limit is not tested because CMAVNS is very fast under g++ O3 param
		evaluations = eval;
 		eval_offset = evaluations - testCase.getRemainingEvaluations();
		reset_mem(dim);
		objfunc = &testCase;
		
		PARAM_CMA_LAMDA = (PARAM_CMA_LAMDA / 2) * 2;
				
		std::vector<double> x0(dim, 0.5);
		double lbounds[dim],ubounds[dim];
		for (int i=0;i<dim;i++)
		{
			lbounds[i] = 0.0;
			ubounds[i] = 1.0;
		}
		GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,dim); // genotype / phenotype transform associated to bounds.
		
		do 
		{
			const double sigma = 0.0;
			CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0, sigma, PARAM_CMA_LAMDA, 0, gp); // -1 for automatically decided lambda
			cmaparams.set_seed(2016+(rand_seed++));
			cmaparams.set_max_fevals(remainingEvaluations());
			cmaparams.set_noisy();
			cmaparams.set_sep();
			cmaparams.set_algo(PARAM_CMA_ALG); // CMAES_DEFAULT, IPOP_CMAES, BIPOP_CMAES, aCMAES, aIPOP_CMAES, aBIPOP_CMAES, sepCMAES, sepIPOP_CMAES, sepBIPOP_CMAES, sepaCMAES, sepaIPOP_CMAES, sepaBIPOP_CMAES, VD_CMAES, VD_IPOP_CMAES, VD_BIPOP_CMAES 
			
			ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> optim(fitfunc,cmaparams);
			optim.set_progress_func(CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>::_defaultPFunc);
				
			if (PARAM_CMA_RUN_UNTIL < 1.0)
			{
				double lastbest = -1;
				int thisiter = 0;
				while(!optim.stop() && (remainingEvaluations() > evaluations * PARAM_CMA_RUN_UNTIL || halloffame.size() < PARAM_HOF_MIN_SIZE))
				{
					dMat candidates = optim.ask();
					optim.eval(candidates);
					optim.tell();
					optim.inc_iter(); // important step: signals next iteration.
				}
				bestSoFar = MAX_OBJV;
				hof_bb_size = find_hof_bb();
			}
			if (MAX_OBJV > TEST_RESULT)
				TEST_RESULT = MAX_OBJV;
			if (PARAM_CMA_RUN_UNTIL > 0.0 && remainingEvaluations() > 0)
			{
				cma_hof_size = halloffame.size();
				int sumTests = remainingEvaluations();
				int lastRunPassed = 0;
				long eliteStarts = 0;
				std::vector< std::vector< bool > > peers;
				while (remainingEvaluations() > 0)
				{
					std::vector< bool > incumbent = random_bitvector( dim );
					double bestValue = 0;
					std::unordered_set<size_t> noImprov;
					
					if (halloffame.size() > 1)
						for( int i=0; i< dim; i++)
						{
							bool samev = true;
							for (int j = 0; j < halloffame.size() - 1 && j < cma_hof_size + 1; j++)
								if ( halloffame[j][i] !=  halloffame[j+1][i])
								{
									samev = false;
									break;
								}
							if (samev)
								noImprov.insert(i);
						}
					
					if (halloffame.size() < 2)
						bestValue = getValue( incumbent ); 
					else
					{
						for (int c = 0; c < PARAM_BB_CX_TRIALS; c++)
						{
							// dVec cand = candidates.col(i);
							std::vector< bool > neighbor = backbone_bitvector(PARAM_BB_CX_ALG, dim, halloffame);
							const bool hashed = history.end() != history.find( neighbor );
							double value = getValue( neighbor );
							if (!hashed)
							{
								if( value > bestValue ) 
								{
									incumbent = neighbor;
									bestValue = value;
								}
								if (value > bestSoFar)
									bestSoFar = value;
								if (bestSoFar > TEST_RESULT)
									TEST_RESULT = bestSoFar;
							}
						}
						eliteStarts ++;
					}
					if (bestValue == 0)
						bestValue = getValue( incumbent ); 
					
					tabu.insert(incumbent);
					bestSoFar = bestValue;
					
					lastRunPassed ++;
					bool improved = false;
					do
					{
						improved = false;
						std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent, noImprov, optim, PARAM_CMA_RUN_UNTIL < 1.0, eliteStarts >= 2 && PARAM_AA_TOLERANCE > 0.0);
						
						int cntNoImprov = noImprov.size();
						std::vector< bool > candForAA;
						double objVForAA = 0;
						double initV = bestValue;
						double peerV = -1;
						for (size_t i=0; i<neighbors.size() && remainingEvaluations() > 0; i++)
						{
							if (cntNoImprov > 0 && i < dim && noImprov.end() != noImprov.find( i ))
								continue;

							// improved 
							std::vector< bool >& neighbor = neighbors[ i ];
							const bool hashed = history.end() != history.find( neighbor );
							double value = getValue( neighbor );
							if (!hashed)
							{
								lastRunPassed = 0;
								if (PARAM_AA_ELITE <= 1.0 && value >= bestSoFar * PARAM_AA_ELITE 
									&& remainingEvaluations() < evaluations / 4 * PARAM_CMA_RUN_UNTIL)
								{
									peers.push_back(neighbor);
								}
							}
							if( value > bestValue && tabu.find(neighbor) == tabu.end()) 
							{
								// new best known
								improved = true;
								incumbent = neighbor;
								bestValue = value;
								tabu.insert(incumbent);
								if (value > bestSoFar)
								{
									bestSoFar = value;
									addHoF(neighbor, true);
									hof_bb_size = find_hof_bb();
									if (PARAM_HH_1DS_DEPTH < SWAP1_HIGH)
										peers.clear();
									if (bestSoFar > TEST_RESULT)
										TEST_RESULT = bestSoFar;
								}
								if (i >= dim)
									noImprov.clear();
							}
							else if (PARAM_HH_1DS_DEPTH > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
								peers.push_back(neighbor);
							// AA needs reconsideration
							if (!improved && i >= dim && eliteStarts >= 2 && PARAM_AA_TOLERANCE > 0.0 && tabu.find(neighbor) == tabu.end()
								&& (objVForAA == 0 || value > objVForAA && value > bestSoFar * (1.0 - PARAM_AA_TOLERANCE)))
							{
								objVForAA = value;
								candForAA = neighbor;
							}
							if (value < initV && i < dim)
								noImprov.insert(i);
						}
						neighbors.clear();
						
						// enclose of a 1DS trial :: UsingDoubleCheck
						if (!improved && noImprov.size() == dim)
						{
							// double check for missing items
							noImprov.clear();
							
							std::vector< std::vector< bool > > neighbors1 = hamming1Neighbours(incumbent);
							
							double initV1 = bestValue;
							for (size_t i=0; i<neighbors1.size() && remainingEvaluations() > 0; i++)
							{
								std::vector< bool >& neighbor = neighbors1[ i ];
								const bool hashed = history.end() != history.find( neighbor );
								double value = getValue(neighbor);
								if (!hashed)
									lastRunPassed = 0;
								if( value > bestValue && tabu.find(neighbor) == tabu.end()) 
								{
									improved = true;
									incumbent = neighbor;
									bestValue = value;
									tabu.insert(neighbor);
									if (value > bestSoFar)
									{
										bestSoFar = value;
										addHoF(neighbor, true);
										hof_bb_size = find_hof_bb();
										if (PARAM_HH_1DS_DEPTH < SWAP1_HIGH)
											peers.clear();
										if (bestSoFar > TEST_RESULT)
											TEST_RESULT = bestSoFar;
									}
									//freq1DS[mod[i]]++;
									noImprov.clear();
								}
								else if (PARAM_HH_1DS_DEPTH > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
									peers.push_back(neighbor);
								else if (value < initV1)
									noImprov.insert(i);
							}
							neighbors1.clear();
						}
						
						// clear noImprov if AA
						if (!improved && PARAM_AA_TOLERANCE > 0.0 && eliteStarts >= 2 && objVForAA > 0 && remainingEvaluations() < evaluations / 2 * PARAM_CMA_RUN_UNTIL )
						{
							tabu.insert(candForAA);
							improved = true;
							incumbent = candForAA;
							bestValue = objVForAA;
							noImprov.clear();
							lastRunPassed = 0;
						}
						if (!improved && peers.size() > 0)
						{
							int best_peer = best_in_vector(peers);
							if (best_peer >= 0)
							{
								incumbent = peers[best_peer];
								bestValue = history.find(incumbent)->second;
								peers.erase(peers.begin()+best_peer);
								tabu.insert(incumbent);
								improved = true;
								noImprov.clear();
								lastRunPassed = 0;
							}
						}
					} 
					while(improved && remainingEvaluations() > 0 );
					
					// HoF by LS
					addHoF(incumbent, true);
					noImprov.clear();
				}
			}
			halloffame.clear();
		}
		while (remainingEvaluations() > 0);
	}
	
	virtual void cmavns_d(ObjectiveFn& testCase, long long maxTimeMs, const int dim, const double EvDiscount, const int eval) 
	{
		objfunc = &testCase;
		evaluations = eval;
 		eval_offset = evaluations - testCase.getRemainingEvaluations();
		reset_mem(dim);
		
		bool FixedHoF = false;
		bool BB_restarts = false;
		bool AdaptiveAccept = false;
		bool AAhighCand = true;
		bool AAnearCand = false;
		
		double sigma = 0.5;
		int targetLambda = round(dim/ 8.0);
		int lambda = targetLambda ;
		double ESratio = 0.7;
		double AAratio = 0.5; 
		int BBtrials = 20;
		int cmaHoFsize = 0;
		if (EvDiscount > 2 && EvDiscount <= 8)
		{
			targetLambda = round(dim/ 8.0);
			lambda = targetLambda % 2 == 1 ? targetLambda : targetLambda - 1;
			ESratio = dim < 175 ? 0.65 : 0.6;
			BBtrials = dim < 195 ? 20 : 40;
			AdaptiveAccept = false;
		}
		else if (EvDiscount > 8)
		{
			targetLambda = round(dim/ 28.0);
			if (dim > 220)
				targetLambda = round(dim/ 26.0);
			else if (dim > 150)
				targetLambda = round(dim/ 22.0);
			if (targetLambda < 3)
				targetLambda = 3;
			lambda = targetLambda % 2 == 1 ? targetLambda : targetLambda - 1;
			
			ESratio = 0.75;
			if (dim > 220)
				ESratio = 0.7;
			else if (dim < 125)
				ESratio = 10.0;	// disabled
			
			BBtrials = 20;
			AdaptiveAccept = false;
		}
		else if (EvDiscount <= 2)
		{
			if (dim < 135)
				targetLambda = round(dim/ 4.0);
			else if (dim >= 280)
				targetLambda = round(dim/ 8.0);
			else
				targetLambda = round(dim/ 6.0);
			lambda = targetLambda % 2 == 1 ? targetLambda : targetLambda - 1;
			
			if (dim < 135)
				ESratio = 0.55;
			else if (dim < 145)
				ESratio = 0.6;
			else if (dim < 220)
				ESratio = 0.65;
			else 
				ESratio = 0.55;
			
			BBtrials = dim < 125 ? 40 : 80;
			AdaptiveAccept = true;
			AAhighCand = true;
			AAnearCand = false;
		}
		
		std::vector<double> x0(dim, 0.5);
		double lbounds[dim],ubounds[dim];
		for (int i=0;i<dim;i++)
		{
			lbounds[i] = 0.0;
			ubounds[i] = 1.0;
		}
		GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,dim); // genotype / phenotype transform associated to bounds.
		
		// CMAES_DEFAULT, IPOP_CMAES, BIPOP_CMAES, aCMAES, aIPOP_CMAES, aBIPOP_CMAES, sepCMAES, sepIPOP_CMAES, sepBIPOP_CMAES, sepaCMAES, sepaIPOP_CMAES, sepaBIPOP_CMAES, VD_CMAES, VD_IPOP_CMAES, VD_BIPOP_CMAES 
		do 
		{
			CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0,sigma,lambda,0,gp); // -1 for automatically decided lambda
			
			cmaparams.set_algo(sepaBIPOP_CMAES);
			cmaparams.set_seed(2015);
			cmaparams.set_max_fevals(remainingEvaluations());
			cmaparams.set_noisy();
			cmaparams.set_sep();
			
			ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> optim(fitfunc_2015,cmaparams);
			optim.set_progress_func(CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>::_defaultPFunc);
				
			if (ESratio < 1.0)
			{
				double lastbest = -1;
				int thisiter = 0;
				while(!optim.stop() && (remainingEvaluations() > evaluations * ESratio || halloffame.size() < (EvDiscount >= 8 ? 2 : 3)))
				{
					dMat candidates = optim.ask();
					optim.eval(candidates);
					optim.tell();
					optim.inc_iter(); // important step: signals next iteration.

					Candidate cmasol1 = optim.get_solutions().get_best_seen_candidate();
					
					double value = cmasol1.get_fvalue();
					if (((EvDiscount >= 8 && dim < 200)  || lastbest == value) && remainingEvaluations() < evaluations * (EvDiscount >= 8 ? 0.95 : 0.9) )
					{
						std::vector< bool > neighbor;
						for (int i = 0; i < dim; i++)
							neighbor.push_back(cmasol1.get_x_ptr()[i] >= 0.5);
						addHoF_2015(neighbor);
					}
					lastbest = value;
			
					if (lastbest > TEST_RESULT)
						TEST_RESULT = lastbest;
				}
			}
			if (ESratio > 0 && remainingEvaluations() > 0)
			{
				cmaHoFsize = halloffame.size();
				int sumTests = remainingEvaluations();
				int lastRunPassed = 0;
				long eliteStarts = 0;
				std::vector< std::vector< bool > > peers;
				while (remainingEvaluations() > 0)
				{
					std::vector< bool > incumbent = random_bitvector( testCase.getNumGenes() );
					double bestValue = 0;
					std::unordered_set<size_t> noImprov;
					
					if ((halloffame.size() >= 6 
						&& (history.find(halloffame[halloffame.size() - 1])->second 
							== history.find(halloffame[halloffame.size() - 6])->second)
						&& (history.find(halloffame[halloffame.size() - 1])->second 
							== history.find(halloffame[halloffame.size() - 2])->second)) || lastRunPassed > (EvDiscount > 8 ? 5 : 0))
					{
						while (halloffame.size() > 2 && halloffame.size() > cmaHoFsize)
							halloffame.pop_back();
					}
					
					if (halloffame.size() > 1)
					{
						for( int i=0; i< dim; ++i )
						{
							bool samev = true;
							for (int j = 0; j < halloffame.size() - 1 && j < cmaHoFsize + 1; j++)
								if ( halloffame[j][i] !=  halloffame[j+1][i])
								{
									samev = false;
									break;
								}
							if (samev)
								noImprov.insert(i);
						}
					}
					
					if (halloffame.size() < 2)
					{
						bestValue = testCase.value( incumbent ); 
						history.insert(std::make_pair<std::vector<bool>&,double&>(incumbent, bestValue));
					}
					else
					{
						for (int i = 0; i < BBtrials; i++)
						{
							// dVec cand = candidates.col(i);
							std::vector< bool > neighbor = backbone_bitvector_2015(testCase.getNumGenes(), halloffame);
							const bool hashed = history.end() != history.find( neighbor );
							double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
							if (!hashed)
								history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
							if( value > bestValue ) {
								incumbent = neighbor;
								bestValue = value;
							}
						}
						eliteStarts ++;
					}
					if (bestValue == 0)
					{
						bestValue = testCase.value( incumbent ); 
						history.insert(std::make_pair<std::vector<bool>&,double&>(incumbent, bestValue));
					}
					
					tabu.insert(incumbent);
					bestSoFar = bestValue;
					
					lastRunPassed ++;
					bool improved = false;
					int suspensions = 0;
					do
					{
						improved = false;
						std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent, noImprov, optim, ESratio < 1.0, eliteStarts >= 2 && AdaptiveAccept);
						
						int cntNoImprov = noImprov.size();
						std::vector< bool > candForAA;
						double objVForAA = 0;
						double initV = bestValue;
						long distToHoF = 1E+8; 
						for( size_t i=0; i<neighbors.size() && remainingEvaluations() > 0; ++i )
						{
							if (cntNoImprov > 0 && i < dim && noImprov.end() != noImprov.find( i ))
								continue;

							std::vector< bool >& neighbor = neighbors[ i ];
							const bool hashed = history.end() != history.find( neighbor );
							double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
							if (!hashed)
							{
								history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
								lastRunPassed = 0;
							}
							if( value > bestValue && tabu.find(neighbor) == tabu.end()) 
							{
								improved = true;
								incumbent = neighbor;
								bestValue = value;
								tabu.insert(incumbent);
								if (value > bestSoFar)
									bestSoFar = value;
								if (bestSoFar > TEST_RESULT)
									TEST_RESULT = bestSoFar;
								if (i >= dim)
									noImprov.clear();
								if (PARAM_HH_1DS_DEPTH_2015 < SWAP1_HIGH)
									peers.clear();
							}
							else if (PARAM_HH_1DS_DEPTH_2015 > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
								peers.push_back(neighbor);
							if (!improved && i >= testCase.getNumGenes() && eliteStarts >= 2 && AdaptiveAccept && tabu.find(neighbor) == tabu.end() && (objVForAA == 0 || !AAhighCand || value > objVForAA))
							{
								long mydist = 0;
								for (int h = ((cmaHoFsize > 2) ? cmaHoFsize-2 : 0); h < cmaHoFsize + 2 && h < halloffame.size(); h++)
									mydist += hamming(neighbor, halloffame[h]);
								if (!AAnearCand || mydist < distToHoF)
								{
									distToHoF = mydist;
									objVForAA = value;
									candForAA = neighbor;
								}
							}
							if (value < initV && i < dim)
								noImprov.insert(i);
						}
						
						// enclose of a trial
						if (!improved && noImprov.size() == testCase.getNumGenes())
						{
							// fine check
							noImprov.clear();
							
							std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent);
							
							double initV1 = bestValue;
							for( size_t i=0; i<neighbors.size() && remainingEvaluations() > 0; ++i )
							{
								std::vector< bool >& neighbor = neighbors[ i ];
								const bool hashed = history.end() != history.find( neighbor );
								double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
								if (!hashed)
								{
									history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
									lastRunPassed = 0;
								}
								if( value > bestValue && tabu.find(neighbor) == tabu.end()) {
									improved = true;
									incumbent = neighbor;
									bestValue = value;
									tabu.insert(neighbor);
									if (value > bestSoFar)
										bestSoFar = value;
									if (bestSoFar > TEST_RESULT)
										TEST_RESULT = bestSoFar;
									if (PARAM_HH_1DS_DEPTH_2015 < SWAP1_HIGH)
										peers.clear();
								}
								else if (PARAM_HH_1DS_DEPTH_2015 > SWAP1_LOW && value > initV && tabu.find(neighbor) == tabu.end())
									peers.push_back(neighbor);
								else if (value < initV1)
									noImprov.insert(i);
							}
						}
						
						if (!improved && AdaptiveAccept && eliteStarts >= 2 && objVForAA > 0 )
						{
							tabu.insert(candForAA);
							improved = true;
							incumbent = candForAA;
							bestValue = objVForAA;
							noImprov.clear();
							lastRunPassed = 0;
						}
						if (!improved && peers.size() > 0)
						{
							int best_peer = best_in_vector(peers);
							if (best_peer >= 0)
							{
								incumbent = peers[best_peer];
								bestValue = history.find(incumbent)->second;
								peers.erase(peers.begin()+best_peer);
								tabu.insert(incumbent);
								improved = true;
								noImprov.clear();
								lastRunPassed = 0;
							}
						}
					} while(( improved)&& remainingEvaluations() > 0 );
					
					addHoF_2015(incumbent);
					noImprov.clear();
				}
			}
		}
		while (remainingEvaluations() > 0);
	}

	void setp(int i, double vals)
	{
		param_preset = true;
		preset_params[i] = vals;
	}
	
	///////////////////////////////
	FitFunc fitfunc = [&](const double *x, const int N)
	{
		std::vector< bool > cand;
		for (int j = 0; j < N; j++)
			cand.push_back(x[j] >= 0.5);
		double value = 0.0;
		if (history.end() == history.find( cand ))
		{
			value = objfunc->value( cand );
			addHistory(cand, value);
			if (value >= (MAX_OBJV - MIN_OBJV) * 0.999 + MIN_OBJV)
				addHoF(cand, true);
		}
		else
			value = history.find( cand )->second;
		return - value;
	};
	
	FitFunc fitfunc_2015 = [&](const double *x, const int N)
	{
		std::vector< bool > cand;
		for (int j = 0; j < N; j++)
			cand.push_back(x[j] >= 0.5);
		double value = 0.0;
		if (history.end() == history.find( cand ))
		{
			value = objfunc->value( cand );
			addHistory(cand, value);
		}
		else
			value = history.find( cand )->second;
		return - value;
	};
	
	FitFunc ifitfunc = [&](const double *x, const int N)
	{
		std::vector< bool > cand;
		size_t idx = 0;
		for (size_t i = 0; i < hof_bb.size(); i++)
			if (hof_bb[i] >= 0)
				cand.push_back(hof_bb[i] == 1);
			else
				cand.push_back(x[idx++] >= 0.5);
		double value = 0.0;
		if (history.end() == history.find( cand ))
		{
			value = objfunc->value( cand );
			addHistory(cand, value);
		}
		else
		{
			value = history.find( cand )->second;
			if (value > MAX_OBJV)
				MAX_OBJV = value;
			if (value < MIN_OBJV)
				MIN_OBJV = value;
		}
		if (value >= (MAX_OBJV - MIN_OBJV) * 0.999 + MIN_OBJV)
			addHoF(cand, true);
		icma_iter_sum += value;
		return - value;
	};
	
	int find_hof_bb()
	{
		int dim = hof_bb.size();
		if (halloffame.size() < 2)
			return 0;
		std::vector<bool> marks(dim, false);
		int ret = 0;
		for (size_t i = 0; i < dim; i++)
		{
			bool samev = true;
			for (size_t j = 1; j < halloffame.size(); j++)
				if (halloffame[0][i] != halloffame[j][i])
				{
					samev = false;
					break;
				}
			marks[i] = samev;
			if (samev)
			{
				ret++;
				hof_bb[i] = halloffame[0][i];
			}
			else
			{
				hof_bb[i] = -1;
			}
		}
		while (ret > dim*PARAM_HOF_BB_MAX_SIZE)
		{
			size_t i = rand()%dim;
			if (hof_bb[i] >= 0)
			{
				ret--;
				hof_bb[i] = -1;
				marks[i] = false;
			}
		}
		for (long trials = 0; ret > 0 && tabu_hof_bb.find(marks)!=tabu_hof_bb.end(); trials++)
		{
			size_t i = 0;
			do
			{
				i = rand()%dim;
			}
			while (hof_bb[i] < 0);
			ret--;
			hof_bb[i] = -1;
			marks[i] = false;
		}
		tabu_hof_bb.insert(marks);
		return ret;
	}
	
	long remainingEvaluations()
	{
		return objfunc->getRemainingEvaluations() + eval_offset;
	}
	
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent) {
		std::vector< std::vector< bool > > result;
		for (size_t i=0; i<incumbent.size(); i++) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		return result;
	}
	
	
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent, bool hof_part) 
	{
		std::vector< std::vector< bool > > result;
		for (size_t i=0; i<incumbent.size(); i++) 
		{
			if (hof_part && hof_bb[i] == (incumbent[i] ? 1 : 0))
				continue;
			else if (!hof_part && hof_bb[i] != (incumbent[i] ? 1 : 0))
				continue;
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		return result;
	}
	
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent, std::unordered_set<size_t>& noImprov, ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> &optim, bool CMA, bool AA) {
		std::vector< std::vector< bool > > result;
		// N times 1-flip solutions
		for (size_t i=0; i<incumbent.size(); i++) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		// after some trials; where banned := noImprov
		if (noImprov.size() > incumbent.size()/2 || AA)
		{
			// 1 x N-flip solution
			if (hof_bb_size > 0)
			{
				std::vector< bool > neighbourNflip = incumbent;
				for (size_t i=0; i<incumbent.size(); i++)
				{
					if (hof_bb[i]<0)
						neighbourNflip[ i ] = !neighbourNflip[ i ];
				}
				if (history.end() == history.find( neighbourNflip ))
					result.push_back( neighbourNflip );
			}
			// 2~N+3 x flips in random
			if (incumbent.size() - noImprov.size() >= 2 && CMA)
			{
				dMat candidates;
				double extraN = (incumbent.size() - noImprov.size() + (AA ? 3 : 0)) * PARAM_VNS_SCALE;
				for (size_t j = 0; j< extraN; ++j )
				{
					std::vector< bool > neighbour = incumbent;
					size_t cnt = 0;
					// using CMA 
					if (CMA && PARAM_VNS_SCALE > 0.0)
					{
						if (j == 0)
							candidates = optim.ask();
						size_t use_cands = candidates.cols();
						if (use_cands < 2)
							use_cands = 2;
						if (j%use_cands == use_cands - 1)
							candidates = optim.ask();
						dVec cand = candidates.col(j % use_cands); //j % candidates.cols());
						for (size_t i=0; i<incumbent.size(); i++)
							if (hof_bb[i]<0)
								neighbour[i] =  cand.data()[i] >=0.5;
					}
					else
					{
						for (size_t i=0; i<incumbent.size(); i++)
							if (noImprov.end() == noImprov.find( i ) && rand() % 2 == 1)
								neighbour[ i ] = !neighbour[ i ];
					}
					if (history.end() == history.find( neighbour ))
						result.push_back( neighbour );
				}
			}
		}
		return result;
	}
	
	std::vector< std::vector< bool > >
	hamming1Neighbours2015( const std::vector< bool >& incumbent, std::unordered_set<size_t>& banned, ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> &optim, bool CMA, bool AA) {
		std::vector< std::vector< bool > > result;
		for (size_t i=0; i<incumbent.size(); ++i ) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		if (banned.size() > incumbent.size()/2 || AA)
		{
			// flip all cands
			{
				std::vector< bool > neighbour = incumbent;
				for (size_t i=0; i<incumbent.size(); ++i )
				{
					if (banned.end() == banned.find( i ))
						neighbour[ i ] = !neighbour[ i ];
				}
				if (history.end() == history.find( neighbour ))
					result.push_back( neighbour );
			}
			//  flip in random
			if (incumbent.size() - banned.size() >= 2 )
			{
				dMat candidates;
				// if (!optim.stop())
				for (size_t j = 0; j<(incumbent.size() - banned.size()) + (AA ? 3 : 0); ++j )
				{
					std::vector< bool > neighbour = incumbent;
					size_t cnt = 0;
					if (CMA && !optim.stop())
					{
						if (j == 0)
							candidates = optim.ask();
						size_t use_cands = candidates.cols()/5;
						if (use_cands < 2)
							use_cands = 2;
						if (j%use_cands == use_cands - 1)
							candidates = optim.ask();
						dVec cand = candidates.col(j % use_cands); //j % candidates.cols());
						for (size_t i=0; i<incumbent.size(); ++i )
							if (banned.end() == banned.find(i))
								neighbour[i] = cand.data()[i] >=0.5;
					}
					else
					{
						for (size_t i=0; i<incumbent.size(); ++i )
						{
							if (banned.end() == banned.find( i ))
								if (rand() % 2 == 1)
									neighbour[ i ] = !neighbour[ i ];
						}
					}
					if (history.end() == history.find( neighbour ))
						result.push_back( neighbour );
				}
			}
		}
		
		return result;
	}
	
	void reset_mem(int dim)
	{
		for (auto h : history)
		{
			auto s = h.first;
			s.clear();
		}
		history.clear();
		halloffame.clear();
		tabu.clear();
		hof_bb.clear();
		for (int i=0;i<dim;i++)
			hof_bb.push_back(-1);
		hof_bb_size = 0;
		cma_hof_size = 0;
		MIN_OBJV = 1.0E+8;
		MAX_OBJV = -1.0;
		bestSoFar = 0.0;
		TEST_RESULT = 0.0;
	}
	
	double getValue(std::vector<bool>& cand)
	{
		const bool hashed = history.end() != history.find( cand );
		double value = hashed ? history.find( cand )->second : objfunc->value( cand );
		if (!hashed)
			addHistory(cand, value);
		return value;
	}
	
	void addHoF(const std::vector< bool >& incumbent, bool cleanMinors = false)
	{
		int mindist = 1E+8;
		for (int j = 0; j < halloffame.size(); j++)
		{
			int dist = hamming(incumbent, halloffame[j]);
			if (dist < mindist)
			{
				mindist = dist;
				if (mindist == 0)
					break;
			}
		}
		if (mindist > 0)
		{
			halloffame.push_back(incumbent);
			if (cleanMinors)
			{
				double min_req = (MAX_OBJV - MIN_OBJV) * PARAM_HOF_SHRINK_THRESHOLD + MIN_OBJV;
				for (size_t i = 0; i < halloffame.size() /2 && halloffame.size() > 2; i++)
					if(history.find(halloffame[i])->second < min_req)
						halloffame.erase(halloffame.begin()+(i--));
			}
		}
	}
	
	
	void addHoF_2015(const std::vector< bool >& incumbent)
	{
		int mindist = 1E+5;
		for (int j = 0; j < halloffame.size(); j++)
		{
			int dist = hamming(incumbent, halloffame[j]);
			if (dist < mindist)
				mindist = dist;
		}
		if (mindist > incumbent.size() / 100)
			halloffame.push_back(incumbent);
	}
	
	void addHistory(std::vector< bool >& incumbent, double val)
	{
		history.insert(std::make_pair<std::vector<bool>&,double&>(incumbent, val));
		if (val > MAX_OBJV)
			MAX_OBJV = val;
		if (val < MIN_OBJV)
			MIN_OBJV = val;
	}
	
	int hamming(const std::vector< bool >& incumbent1, const std::vector< bool >& incumbent2)
	{
		int ret = 0;
		for (int i = 0; i < incumbent1.size(); i++)
			ret += incumbent1[i] == incumbent2[i] ? 0 : 1;
		return ret;
	}
	
	std::vector< bool > 
	backbone_bitvector(int PARAM_BB_CX_ALG, int length, std::vector<std::vector<bool>>& parents)
	{
		if (PARAM_BB_CX_ALG == 1)
			return free_crossover_random_HoF(length, parents);
		else if (PARAM_BB_CX_ALG == 2)
			return rand()%3 != 0 ? free_crossover_latest_HoF(length, parents) : free_crossover_random_HoF(length, parents);
		else if (PARAM_BB_CX_ALG == 3)
			return rand()%3 == 0 ? free_crossover_latest_HoF(length, parents) : free_crossover_random_HoF(length, parents);
		return free_crossover_latest_HoF(length, parents);
	}
	
	std::vector< bool >
	backbone_bitvector_2015( int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		for( int i=0; i<length; ++i )
		{
			result.push_back( parents[(parents.size() - 1) - (rand() % 2)][i] );
		}
		return result;
	}
	
	std::vector< bool >
	free_crossover_latest_HoF( int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		if (PARAM_BB_CX_NUM_POINTS <= 0)
			for( int i=0; i<length; i++)
				result.push_back( parents[(parents.size() - 1) - (rand() % 2)][i] );
		else
		{
			int p1 = parents.size() - 1;
			int p2 = p1;
			int hDist = 0;
			for (p2 = p1-1; p2 >= 0; p2 --)
			{
				if ((hDist = hamming(parents[p1], parents[p2])) >= PARAM_MIN_HAMMING_DIST)
					break;
			}
			if (hDist < PARAM_MIN_HAMMING_DIST)
				return free_crossover_random_HoF(length, parents);
			
			//	std::cout << hDist << std::endl;
			crosspoints.clear();
			for(int i = 0; i < PARAM_BB_CX_NUM_POINTS && i < hDist; i++)
				crosspoints.push_back(rand()%length+1);
			std::sort(crosspoints.begin(), crosspoints.end());
			//for( int i=0; i<crosspoints.size(); i++)
			//	std::cout<<crosspoints[i] <<" "; 
			for(int i=0; i<crosspoints.size(); i++)
				for (int j = (i==0 ? 0: crosspoints[i-1]); j < crosspoints[i]; j++)
					result.push_back( parents[(crosspoints[0]+i)%2==0 ? p1 : p2][j] );
			for (int j = crosspoints[crosspoints.size()-1]; j < length; j++)
				result.push_back( parents[(crosspoints[0]+crosspoints.size())%2==0 ? p1 : p2][j] );
			hDist = std::min(hamming(result, parents[p1]), hamming(result, parents[p2]));
			if (hDist < PARAM_MIN_HAMMING_DIST)
				for( int i=hDist; i<=PARAM_MIN_HAMMING_DIST; i++)
				{
					int idx = rand()%length;
					result[idx] = !result[idx];
				}
		}
		return result;
	}
	
	
	std::vector< bool >
	free_crossover_random_HoF(int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		for( int i=0; i<length; i++)
		{
			result.push_back( parents[rand() % parents.size()][i]);
		}
		return result;
	}
	
	void load_apriori_data()
	{
		// the 5 sample points - dim
		apriori_dim.push_back(50.0);
		apriori_dim.push_back(80.0);
		apriori_dim.push_back(130.0);
		apriori_dim.push_back(200.0);
		apriori_dim.push_back(300.0);
		// lamda 
		apriori_lamda_lc.push_back(20);	
		apriori_lamda_lc.push_back(22);	
		apriori_lamda_lc.push_back(32);
		apriori_lamda_lc.push_back(36);
		apriori_lamda_lc.push_back(45);
		apriori_lamda_hc.push_back(4);	
		apriori_lamda_hc.push_back(5); 
		apriori_lamda_hc.push_back(9);
		apriori_lamda_hc.push_back(16);
		apriori_lamda_hc.push_back(32);
		
		// backbone trials
		apriori_bbtrials_hc.push_back(10); 
		apriori_bbtrials_hc.push_back(10);
		apriori_bbtrials_hc.push_back(20);
		apriori_bbtrials_hc.push_back(20);
		apriori_bbtrials_hc.push_back(40);
		apriori_bbtrials_lc.push_back(10); 
		apriori_bbtrials_lc.push_back(30);
		apriori_bbtrials_lc.push_back(50);
		apriori_bbtrials_lc.push_back(70);
		apriori_bbtrials_lc.push_back(80);
		
		// CX num of points
		apriori_cxnum_lc.push_back(1); 
		apriori_cxnum_lc.push_back(1);
		apriori_cxnum_lc.push_back(5);
		apriori_cxnum_lc.push_back(20);
		apriori_cxnum_lc.push_back(15);
		apriori_cxnum_mc.push_back(3);
		apriori_cxnum_mc.push_back(2);  
		apriori_cxnum_mc.push_back(1);	
		apriori_cxnum_mc.push_back(1);	
		apriori_cxnum_mc.push_back(1);	
		
		// min hDist
		apriori_hdist_lc.push_back(5);
		apriori_hdist_lc.push_back(4);
		apriori_hdist_lc.push_back(4);
		apriori_hdist_lc.push_back(2);
		apriori_hdist_lc.push_back(3);
		apriori_hdist_mc.push_back(5);
		apriori_hdist_mc.push_back(5);
		apriori_hdist_mc.push_back(4);
		apriori_hdist_mc.push_back(2);
		apriori_hdist_mc.push_back(2);
		apriori_hdist_hc.push_back(3);
		apriori_hdist_hc.push_back(3);
		apriori_hdist_hc.push_back(4);
		apriori_hdist_hc.push_back(4);
		apriori_hdist_hc.push_back(5);
		
		// run_until
		apriori_rununtil_hc.push_back(90);
		apriori_rununtil_hc.push_back(55);
		apriori_rununtil_hc.push_back(50);
		apriori_rununtil_hc.push_back(50);
		apriori_rununtil_hc.push_back(50);
		
		apriori_rununtil_mc.push_back(80);
		apriori_rununtil_mc.push_back(55);
		apriori_rununtil_mc.push_back(55);
		apriori_rununtil_mc.push_back(50);
		apriori_rununtil_mc.push_back(50);
		
		apriori_rununtil_lc.push_back(50);
		apriori_rununtil_lc.push_back(65);
		apriori_rununtil_lc.push_back(60);
		apriori_rununtil_lc.push_back(50);
		apriori_rununtil_lc.push_back(50);
		
		apriori_rununtil_b_hc.push_back(80);
		apriori_rununtil_b_hc.push_back(75);
		apriori_rununtil_b_hc.push_back(65);
		apriori_rununtil_b_hc.push_back(55);
		apriori_rununtil_b_hc.push_back(60);
		
		apriori_rununtil_b_mc.push_back(90);
		apriori_rununtil_b_mc.push_back(50);
		apriori_rununtil_b_mc.push_back(50);
		apriori_rununtil_b_mc.push_back(55);
		apriori_rununtil_b_mc.push_back(50);
		
		apriori_rununtil_b_lc.push_back(65);
		apriori_rununtil_b_lc.push_back(65);
		apriori_rununtil_b_lc.push_back(60);
		apriori_rununtil_b_lc.push_back(50);
		apriori_rununtil_b_lc.push_back(50);
		
		apriori_rununtil_icma_hc.push_back(90);
		apriori_rununtil_icma_hc.push_back(80);
		apriori_rununtil_icma_hc.push_back(55);
		apriori_rununtil_icma_hc.push_back(50);
		apriori_rununtil_icma_hc.push_back(50);	
		
		apriori_rununtil_icma_mc.push_back(90);
		apriori_rununtil_icma_mc.push_back(50);
		apriori_rununtil_icma_mc.push_back(30);
		apriori_rununtil_icma_mc.push_back(50);
		apriori_rununtil_icma_mc.push_back(50);
		
		apriori_rununtil_icma_lc.push_back(60);
		apriori_rununtil_icma_lc.push_back(50);
		apriori_rununtil_icma_lc.push_back(50);
		apriori_rununtil_icma_lc.push_back(55);
		apriori_rununtil_icma_lc.push_back(55);
	}
	
	void setup_apriori_params(const int dim, const double EvDiscount, const bool highVV)
	{
		int LambdaLC = interpolation(apriori_lamda_lc, (double)dim);
		int LambdaHC = interpolation(apriori_lamda_hc, (double)dim);
		int BBHC = interpolation(apriori_bbtrials_hc, (double)dim);
		int BBLC = interpolation(apriori_bbtrials_lc, (double)dim);
		PARAM_CMA_ALG = 				dim <= 170 ? 2 : 11;			// BIPOP_CMAES : sepaBIPOP_CMAES
		PARAM_CMA_LAMDA = 				EvDiscount > 4 ? LambdaHC : (EvDiscount <= 1 ? LambdaLC : (LambdaLC + LambdaHC) / 2);
		PARAM_AA_ELITE = 				dim > 105 ? 0.995 : 1.0;
		PARAM_AA_TOLERANCE = 			EvDiscount > 4 && dim > 65 ? 0.5 : 0.0;
		PARAM_BB_CX_ALG = 				((EvDiscount > 4 || highVV) && dim > 250) || (dim > 165 && EvDiscount <= 1 && highVV) ? 3 : 0;
		PARAM_BB_CX_TRIALS = 			EvDiscount > 4 ? BBHC : BBLC ; 	// cluster size generated from BB
		
		if (EvDiscount <= 1)
			PARAM_VNS_SCALE = 			dim <= 165 ? 0.1 : 0.3;
		else if (EvDiscount <= 4)
			PARAM_VNS_SCALE = 			dim <= 105 ? 0.1 : 0.2;
		else 
			PARAM_VNS_SCALE = 			dim <= 165 ? 0.1 : 0.0;
		
		if (EvDiscount > 4)
			PARAM_BB_CX_NUM_POINTS = 	dim <= 165 ? 5 : 30;
		else if (EvDiscount <= 1)
			PARAM_BB_CX_NUM_POINTS = 	interpolation(apriori_cxnum_lc, (double)dim);
		else
			PARAM_BB_CX_NUM_POINTS = 	interpolation(apriori_cxnum_mc, (double)dim);
		
		if (highVV && EvDiscount > 1 && EvDiscount <= 4)
			PARAM_BB_CX_NUM_POINTS -= 1;
		
		if (EvDiscount > 4)
			PARAM_MIN_HAMMING_DIST = 	interpolation(apriori_hdist_hc, (double)dim);
		else if (EvDiscount <= 1)
			PARAM_MIN_HAMMING_DIST = 	interpolation(apriori_hdist_lc, (double)dim);
		else
			PARAM_MIN_HAMMING_DIST = 	interpolation(apriori_hdist_mc, (double)dim);
		
		if (EvDiscount <= 1.0)
			PARAM_HOF_SHRINK_THRESHOLD = 0.75;
		else if (EvDiscount > 4.0)
			PARAM_HOF_SHRINK_THRESHOLD = 0.95;
		else
			PARAM_HOF_SHRINK_THRESHOLD = dim <= 65 ? 0.95 : 0.8;
		
		if (EvDiscount > 1.0 && EvDiscount <= 4.0)
			PARAM_HOF_BB_MAX_SIZE = 	0.94;
		else
			PARAM_HOF_BB_MAX_SIZE = 	0.96;
		
		if (EvDiscount <= 1.0)
		{
			PARAM_CMA_RUN_UNTIL = 		interpolation(highVV ? apriori_rununtil_b_lc : apriori_rununtil_lc, (double)dim) * 0.01;
			PARAM_CMA_RUN_UNTIL_ICMA = 	interpolation(apriori_rununtil_icma_lc, (double)dim) * 0.01;
		}
		else if (EvDiscount <= 4.0)
		{
			PARAM_CMA_RUN_UNTIL = 		interpolation(highVV ? apriori_rununtil_b_mc : apriori_rununtil_mc, (double)dim) * 0.01;
			PARAM_CMA_RUN_UNTIL_ICMA = 	interpolation(apriori_rununtil_icma_mc, (double)dim) * 0.01;
		}
		else
		{
			PARAM_CMA_RUN_UNTIL = 		interpolation(highVV ? apriori_rununtil_b_hc : apriori_rununtil_hc, (double)dim) * 0.01;
			PARAM_CMA_RUN_UNTIL_ICMA = 	interpolation(apriori_rununtil_icma_hc, (double)dim) * 0.01;
		}
		PARAM_HH_1DS_DEPTH = 			dim < 80 && EvDiscount > 4 ? SWAP1_HIGH : SWAP1_MEDIUM;
		PARAM_HH_1DS_DEPTH_ICMA = 		SWAP1_MEDIUM;
		PARAM_HH_1DS_DEPTH_2015 = 		(EvDiscount > 1 && EvDiscount <= 4) || (EvDiscount <= 1 && dim > 250) ? SWAP1_MEDIUM : SWAP1_LOW;
		
		if (EvDiscount <= 1.0)
			PARAM_HH_TOP_PROFILE = 			dim > 220 ? P1_CMAVNS : (dim <= 65 && !highVV || dim > 100 ? P4_CMAVNS_VARIANT : (dim <= 125 || highVV ? P3_CMAVNS_W : P2_CMAVNS_D));
		else if (EvDiscount <= 4.0)
			PARAM_HH_TOP_PROFILE = 			dim > 240 ? P1_CMAVNS : (dim <= 100 || (dim <= 165 && highVV) || dim > 240 ? P4_CMAVNS_VARIANT : (dim <= 70 || dim > 250 ? P2_CMAVNS_D : P3_CMAVNS_W));
		else
			PARAM_HH_TOP_PROFILE = 			P4_CMAVNS_VARIANT; //dim <= 85 ? P2_CMAVNS_D : P3_CMAVNS_W;
	}
		
	int best_in_vector(std::vector< std::vector< bool > >& vec)
	{
		if (vec.size()==0)
			return -1;
		int ret = -1;
		double vbest = -1.0;
		for (size_t i = 0; i < vec.size(); i++)
		{
			if (tabu.find(vec[i]) != tabu.end())
				continue;
			double myv = history.find(vec[i])->second;
			if (myv > vbest)
			{
				ret = i;
				vbest = myv;
			}
		}
		return ret;
	}
	
	int last_in_vector(std::vector< std::vector< bool > >& vec)
	{
		if (vec.size()==0)
			return -1;
		for (int i = vec.size() - 1; i >=0 ; i--)
		{
			if (tabu.find(vec[i]) != tabu.end())
				continue;
			return i;
		}
		return -1;
	}
	
	int interpolation(std::vector< int >& y, double d)
	{
		for (int i = 0; i < apriori_dim.size(); i++)
		{
			if (apriori_dim[i] == d)
				return y[i];
			if (i < apriori_dim.size() - 1 && apriori_dim[i] < d && apriori_dim[i+1] > d)
				return y[i] + (int)std::round(1.0*(y[i+1]-y[i])*(d-apriori_dim[i])/(apriori_dim[i+1]-apriori_dim[i]));
		}
		return y[y.size() - 1];
	}
};

//////////////////////////////////////////////////////////////////////

#endif

// End ///////////////////////////////////////////////////////////////
