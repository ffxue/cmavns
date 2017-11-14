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
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * The authors appreciate the libcmaes <https://github.com/beniz/libcmaes>.
 *
 * This file is an independently redesigned variant of the 2015 version, 
 * involving 7 new components and 10+ new parameters. This class serves 
 * as one of the profiles for selection regarding the problem settings.
 *
 * This file implements the profile P4_CMAVNS_VARIANT.
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
 

#ifndef CMAVNS2_P4_HPP
#define CMAVNS2_P4_HPP

//////////////////////////////////////////////////////////////////////

#include "cbboc/CBBOCUtil.hpp"
#include "cbboc/Competitor.hpp"
#include "cbboc/ObjectiveFn.hpp"
#include "cbboc/ProblemClass.hpp"
#include "cbboc/TrainingCategory.hpp"
#include "cbboc/RNG.hpp"

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
 
class CMAVNS2P4 : public Competitor 
{
private:
	TrainingCategory myCategory = TrainingCategory::NONE;
	ObjectiveFn *objfunc;
	int evalOffset;
	std::unordered_map<std::vector<bool>, double> history;
	std::vector<std::vector<bool>> halloffame;
	std::unordered_set<std::vector<bool>> tabu;
	std::vector<double> apriori_dim;
	std::vector<int> apriori_lamda_hc;
	std::vector<int> apriori_lamda_lc;
	std::vector<int> apriori_bbtrials_hc;
	std::vector<int> apriori_bbtrials_lc;
	int cmaHoFsize;
	
	bool trainingMode = false;
	double learnedESratio = -1.0;
	double learnedLamdaCoeff = -1.0;
	double learnedTrialsCoeff = -1.0;
	bool learnedUsing2015 = false;
	
	//bool param_preset = false;
	//double* preset_params;
public:
	double TEST_RESULT = -1.0;
	double bestSoFar = 0.0;
	
	CMAVNS2P4() { objfunc = 0; //preset_params = new double[30]; 
		apriori_dim.push_back(50);	apriori_lamda_lc.push_back(20);	apriori_lamda_hc.push_back(4); apriori_bbtrials_hc.push_back(10); apriori_bbtrials_lc.push_back(10);
		apriori_dim.push_back(80);	apriori_lamda_lc.push_back(22);	apriori_lamda_hc.push_back(5); apriori_bbtrials_hc.push_back(10); apriori_bbtrials_lc.push_back(30);
		apriori_dim.push_back(130);	apriori_lamda_lc.push_back(32);	apriori_lamda_hc.push_back(9); apriori_bbtrials_hc.push_back(20); apriori_bbtrials_lc.push_back(50);
		apriori_dim.push_back(200);	apriori_lamda_lc.push_back(36);	apriori_lamda_hc.push_back(16); apriori_bbtrials_hc.push_back(20); apriori_bbtrials_lc.push_back(70);
		apriori_dim.push_back(300);	apriori_lamda_lc.push_back(45);	apriori_lamda_hc.push_back(32); apriori_bbtrials_hc.push_back(40); apriori_bbtrials_lc.push_back(80);
	}
	
	CMAVNS2P4(TrainingCategory category) : CMAVNS2P4() {myCategory = category;}
	
	CMAVNS2P4(bool selfTraining) : CMAVNS2P4() {trainingMode = selfTraining; }

	virtual ~CMAVNS2P4() 
	{ 
		for (auto h : history)
		{
			auto s = h.first;
			s.clear();
		}
		history.clear();
		halloffame.clear();
		tabu.clear();
	}

	///////////////////////////////

	virtual TrainingCategory getTrainingCategory() const { return myCategory; }

	virtual void train( std::vector<ObjectiveFn>& trainingSet, long long maxTimeInMilliseconds ) 
	{
	}

	virtual void test( ObjectiveFn& testCase, long long maxTimeInMilliseconds )
	{
		const int dim = testCase.getNumGenes();
		const double EvDiscount = 1.0 * dim * dim / testCase.getRemainingEvaluations();
		run(testCase, maxTimeInMilliseconds, dim, EvDiscount, testCase.getRemainingEvaluations(), -1.0, 1.0, 1.0);
	}

	virtual void run(ObjectiveFn& testCase, long long maxTimeMs, const int dim, const double EvDiscount, const int evaluations, const double trainingESratio, const double trainingLamdaCoeff, const double trainingTrialsCoeff) 
	{
		
		const long long startTime = system_current_time_millis();
 		evalOffset = evaluations - testCase.getRemainingEvaluations();
		
		for (auto h : history)
		{
			auto s = h.first;
			s.clear();
		}
		history.clear();
		halloffame.clear();
		tabu.clear();
		cmaHoFsize = 0;
		bestSoFar = 0.0;
		TEST_RESULT = -1.0;
		objfunc = &testCase;
		
		
		// params 
		int CMAAlgorithm = dim <= 170 ? 2 : 11;		// BIPOP_CMAES : sepaBIPOP_CMAES
		int LambdaLC = interpolation(apriori_dim, apriori_lamda_lc, dim);	// low constraints, EvDiscount <= 1
		int LambdaHC = interpolation(apriori_dim, apriori_lamda_hc, dim);	// high constraints, EvDiscount > 4
		int lambda = EvDiscount > 4 ? LambdaHC : (EvDiscount <= 1 ? LambdaLC : (LambdaLC + LambdaHC) / 2);
		double sigma = 0.0;
		double ESratio = 0.55;
		if ((dim <= 65 && EvDiscount > 1))
			ESratio = 1.0;
		else if (dim <= 105 && EvDiscount <= 1)
			ESratio = 0.65;
		double AAratio = EvDiscount <= 1 && dim > 80 ? 0.5 : 0.0; 		// tolarented gap to bestV in AA
		double AApeers = dim > 105 ? 0.995 : 1.0;
		int BBAlg = EvDiscount > 4 ? ( dim > 65 && dim <= 165 ? 0 : 3 ) : ( dim <= 250 ? 2 : 0 );
		int BBHC = interpolation(apriori_dim, apriori_bbtrials_hc, dim);
		int BBLC = interpolation(apriori_dim, apriori_bbtrials_lc, dim);
		int BBtrials = EvDiscount > 4 ? BBHC : BBLC ; // cluster size generated from BackBone each time; good scaling at 3-5
		bool UsingFreq1Hamming = false;	// using frequent changes of var
		int ESHoFDummySize = 3;		// Expected Minimum HoF size by CMA
		double CMAMatureRatio = 0.95 ;
		int ESFittingRuns = dim < 180 ? 0 : (EvDiscount > 4 ? 10 : 0) ;		// Minimum fitting runs for CMA
		double CMALSscale = 1.0;
		if (EvDiscount <= 1)
			CMALSscale = dim <= 65 ? 0.75 : 0.25;
		else if (EvDiscount > 4)
			CMALSscale = dim <= 65 ? 0.25 : 0.5;
		else 
			CMALSscale = dim <= 65 || dim > 250 ? 1.0 : 0.75;
		double CMALSlimit = 10.0;
		if (EvDiscount <= 1)
			CMALSlimit = dim <= 105 ? 0.25 : ( dim > 165 ? 0.05 : 0.025);
		else if (EvDiscount > 4)
			CMALSlimit = dim <= 65 ? 10.0 : 0.1;
		else 
			CMALSlimit = dim <= 105 ? 0.5 : 10.0;
		bool WatchHoFPremature = EvDiscount <= 1;
		bool autoAdjustWatch = true;
		
		if (trainingMode)	// training mode
		{
			if (trainingESratio > 0.0)
				ESratio = trainingESratio;
			if (trainingLamdaCoeff > 0.0 && trainingLamdaCoeff != 1.0)
				lambda = (int)(1.0 * lambda * trainingLamdaCoeff);
			if (trainingTrialsCoeff > 0.0 && trainingTrialsCoeff != 1.0)
				BBtrials = (int)(1.0 * BBtrials * trainingTrialsCoeff);
		}
		else if (myCategory != TrainingCategory::NONE)
		{
			if (learnedESratio > 0.0)
				ESratio = learnedESratio;
			if (learnedLamdaCoeff > 0.0)
				lambda = (int)(1.0 * lambda * learnedLamdaCoeff);
			if (learnedLamdaCoeff > 0.0)
				BBtrials = (int)(1.0 * BBtrials * learnedTrialsCoeff);
		}
		lambda = (lambda / 2) * 2;
		
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
			CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(x0,sigma,lambda,0,gp); // -1 for automatically decided lambda
			
			cmaparams.set_seed(2016);
			cmaparams.set_max_fevals(testCase.getRemainingEvaluations() + evalOffset);
			cmaparams.set_noisy();
			cmaparams.set_sep();
			cmaparams.set_algo(CMAAlgorithm); // CMAES_DEFAULT, IPOP_CMAES, BIPOP_CMAES, aCMAES, aIPOP_CMAES, aBIPOP_CMAES, sepCMAES, sepIPOP_CMAES, sepBIPOP_CMAES, sepaCMAES, sepaIPOP_CMAES, sepaBIPOP_CMAES, VD_CMAES, VD_IPOP_CMAES, VD_BIPOP_CMAES 
			
			ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> optim(fitfunc,cmaparams);
			optim.set_progress_func(CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>::_defaultPFunc);
				
			if (ESratio < 1.0)
			{
				double lastbest = -1;
				int thisiter = 0;
				while(!optim.stop() && (testCase.getRemainingEvaluations() + evalOffset > evaluations * ESratio || halloffame.size() < ESHoFDummySize))
				{
					dMat candidates = optim.ask();
					optim.eval(candidates);
					optim.tell();
					optim.inc_iter(); // important step: signals next iteration.

					Candidate cmasol1 = optim.get_solutions().get_best_seen_candidate();
					
					double value = cmasol1.get_fvalue();
					// HoF by CMA
					if (((EvDiscount > 4 && dim < 100)  || lastbest == value) && testCase.getRemainingEvaluations() + evalOffset < evaluations * CMAMatureRatio )
					{
						std::vector< bool > neighbor;
						for (int i = 0; i < dim; i++)
							neighbor.push_back(cmasol1.get_x_ptr()[i] >= 0.5);
						addHoF(neighbor);
					}
					lastbest = value;
					if (value > bestSoFar)
						bestSoFar = value;
				}
				if (autoAdjustWatch)
				{
					WatchHoFPremature = WatchHoFPremature || lastbest > 2.0 * dim;	// watch for higher constrainted
				}
			}
			
			if (ESratio > 0 && testCase.getRemainingEvaluations() + evalOffset > 0)
			{
				cmaHoFsize = halloffame.size();
				int sumTests = testCase.getRemainingEvaluations() + evalOffset;
				int lastRunPassed = 0;
				long eliteStarts = 0;
				while (testCase.getRemainingEvaluations() + evalOffset > 0)
				{
					std::vector< bool > incumbent = random_bitvector( testCase.getNumGenes() );
					double bestValue = 0;
					std::unordered_set<size_t> noImprov;
					
					if (WatchHoFPremature && ((halloffame.size() >= 6 
						&& (history.find(halloffame[halloffame.size() - 1])->second 
							== history.find(halloffame[halloffame.size() - 6])->second)
						&& (history.find(halloffame[halloffame.size() - 1])->second 
							== history.find(halloffame[halloffame.size() - 2])->second)) || lastRunPassed > ESFittingRuns))
					{
						while (halloffame.size() > 2 && halloffame.size() > cmaHoFsize)
						{
							if (history.find(halloffame[0])->second < history.find(halloffame[halloffame.size() - 1])->second)
								halloffame.erase(halloffame.begin());
							else
								halloffame.pop_back();
						}
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
						for (int c = 0; c < BBtrials; c++)
						{
							std::vector< bool > neighbor = backbone_bitvector(BBAlg, testCase.getNumGenes(), halloffame, BBtrials);
							const bool hashed = history.end() != history.find( neighbor );
							double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );						if (value > TEST_RESULT)
								TEST_RESULT = value;
							if (!hashed)
							{
								history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
								if( value > bestValue ) 
								{
									incumbent = neighbor;
									bestValue = value;
								}
								if (value > bestSoFar)
									bestSoFar = value;
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
					//if (bestValue > bestSoFar)
						bestSoFar = bestValue;
					
					lastRunPassed ++;
					bool improved = false;
					int suspensions = 0;
					std::vector< std::vector< bool > > peers;
					do
					{
						improved = false;
						std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent, noImprov, optim, ESratio < 1.0, eliteStarts >= 2 && AAratio > 0.0, CMALSscale, CMALSlimit);
						
						int cntNoImprov = noImprov.size();
						std::vector< bool > candForAA;
						double objVForAA = 0;
						double initV = bestValue;
						if (initV > TEST_RESULT)
							TEST_RESULT = initV;
						std::vector< bool >& peer = candForAA;
						double peerV = -1;
						for( size_t i=0; i<neighbors.size() && testCase.getRemainingEvaluations() + evalOffset > 0; ++i )
						{
							if (cntNoImprov > 0 && i < dim && noImprov.end() != noImprov.find( i ))
								continue;

							// improved 
							std::vector< bool >& neighbor = neighbors[ i ];
							const bool hashed = history.end() != history.find( neighbor );
							double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
							if (value > TEST_RESULT)
								TEST_RESULT = value;
							if (!hashed)
							{
								history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
								lastRunPassed = 0;
								if (AApeers <= 1.0 && value >= bestSoFar * AApeers && testCase.getRemainingEvaluations() + evalOffset < evaluations / 4 * ESratio)
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
								}
								if (i >= testCase.getNumGenes())
									noImprov.clear();
							}
							// AA needs reconsideration
							else if (!improved && i >= testCase.getNumGenes() && /* (objVForAA == 0 || value < objVForAA) &&*/ eliteStarts >= 2 && AAratio > 0.0 && tabu.find(neighbor) == tabu.end())
							{
								if (objVForAA == 0 || value > objVForAA && value > bestSoFar * (1.0 - AAratio))
								{
									objVForAA = value;
									candForAA = neighbor;
								}
							}
							if (value < initV)
							{
								if (i < testCase.getNumGenes())
									noImprov.insert(i);
							}
						}
						neighbors.clear();
						
						// enclose of a 1DS trial :: UsingDoubleCheck
						if (!improved && noImprov.size() == testCase.getNumGenes())
						{
							// double check for missing items
							noImprov.clear();
							
							std::vector< std::vector< bool > > neighbors1 = hamming1Neighbours(incumbent);
							
							double initV1 = bestValue;
							for( size_t i=0; i<neighbors1.size() && testCase.getRemainingEvaluations() + evalOffset > 0; ++i )
							{
								std::vector< bool >& neighbor = neighbors1[ i ];
								const bool hashed = history.end() != history.find( neighbor );
								double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
								if (value > TEST_RESULT)
									TEST_RESULT = value;
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
									{
										bestSoFar = value;
									}
									noImprov.clear();
								}
								else if (value < initV1)
								{
									noImprov.insert(i);
								}
							}
							neighbors1.clear();
						}
						
						
						// clear noImprov if AA
						if (!improved && AAratio > 0.0 && eliteStarts >= 2 && objVForAA > 0 && testCase.getRemainingEvaluations() + evalOffset < evaluations / 2 * ESratio )
						{
							tabu.insert(candForAA);
							improved = true;
							incumbent = candForAA;
							bestValue = objVForAA;
							noImprov.clear();
							lastRunPassed = 0;
						}
						
						
						if (!improved && AApeers <= 1.0 && peers.size() > 0)
						{
							incumbent = peers[peers.size()-1];
							peers.pop_back();
							bestValue = bestSoFar;
							tabu.insert(incumbent);
							improved = true;
							noImprov.clear();
							lastRunPassed = 0;
						}
						
						if (autoAdjustWatch)
						{
							WatchHoFPremature = bestSoFar > 2.0 * dim;	// watch for higher constrainted
						}
					} 
					while(improved && testCase.getRemainingEvaluations() + evalOffset > 0 );
					
					// HoF by LS
					addHoF(incumbent);
					noImprov.clear();
				}
			}
			halloffame.clear();
		}
		while (testCase.getRemainingEvaluations() + evalOffset > 0);
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
			history.insert(std::make_pair<std::vector<bool>&,double&>(cand, value));
		}
		else
			value = history.find( cand )->second;
		if (value > TEST_RESULT)
			TEST_RESULT = value;
		return - value;
	};
	
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent) {
		std::vector< std::vector< bool > > result;
		for( size_t i=0; i<incumbent.size(); ++i ) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		return result;
	}
	
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent, std::unordered_set<size_t>& noImprov, ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> &optim, bool CMA, bool AA, double cmaNscale, double cmaNlimit) {
		std::vector< std::vector< bool > > result;
		// N times 1-flip solutions
		for( size_t i=0; i<incumbent.size(); ++i ) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		// after some trials; where banned := noImprov
		if (noImprov.size() > incumbent.size()/2 || AA)
		{
			// 1 x N-flip solution
			std::vector< bool > neighbourNflip = incumbent;
			for( size_t i=0; i<incumbent.size(); ++i )
			{
				if (noImprov.end() == noImprov.find( i ))
					neighbourNflip[ i ] = !neighbourNflip[ i ];
			}
			if (history.end() == history.find( neighbourNflip ))
				result.push_back( neighbourNflip );
			// 2~N+3 x flips in random
			if (incumbent.size() - noImprov.size() >= 2 && CMA)
			{
				dMat candidates;
				// if (!optim.stop())
				double extraN = std::min(((incumbent.size() - noImprov.size()) + (AA ? 3 : 0)) * cmaNscale, incumbent.size() * cmaNlimit);
				for ( size_t j = 0; j< extraN; ++j )
				{
					std::vector< bool > neighbour = incumbent;
					size_t cnt = 0;
					// using CMA 
					if (CMA && !optim.stop())
					{
						if (j == 0)
							candidates = optim.ask();
						size_t use_cands = candidates.cols()/5;
						if (use_cands < 2)
							use_cands = 2;
						if (j%use_cands == use_cands - 1)
							candidates = optim.ask();
						dVec cand = candidates.col(j % use_cands); 
						for ( size_t i=0; i<incumbent.size(); ++i )
							if (noImprov.end() == noImprov.find(i))
								neighbour[i] = cand.data()[i] >=0.5;
					}
					else
					{
						for( size_t i=0; i<incumbent.size(); ++i )
						{
							if (noImprov.end() == noImprov.find( i ))
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
	
	
	void addHoF(const std::vector< bool >& incumbent)
	{
		int mindist = 1E+5;
		for (int j = 0; j < halloffame.size(); j++)
		{
			int dist = hamming(incumbent, halloffame[j]);
			if (dist < mindist)
				mindist = dist;
		}
		if (mindist > 0 && mindist > incumbent.size() / 100)
			halloffame.push_back(incumbent);
	}
	
	int hamming(const std::vector< bool >& incumbent1, const std::vector< bool >& incumbent2)
	{
		int ret = 0;
		for (int i = 0; i < incumbent1.size(); i++)
			ret += incumbent1[i] == incumbent2[i] ? 0 : 1;
		return ret;
	}
	
	std::vector< bool > 
	backbone_bitvector(int BBAlg, int length, std::vector<std::vector<bool>>& parents, int bb_trails)
	{
		if (BBAlg == 1)
			return free_crossover_random_HoF(length, parents);
		else if (BBAlg == 2)
			return rand()%3 != 0 ? free_crossover_latest_HoF(length, parents) : free_crossover_random_HoF(length, parents);
		else if (BBAlg == 3)
			return rand()%3 == 0 ? free_crossover_latest_HoF(length, parents) : free_crossover_random_HoF(length, parents);
		return free_crossover_latest_HoF(length, parents);
	}
	
	std::vector< bool >
	free_crossover_latest_HoF( int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		for( int i=0; i<length; ++i )
		{
			result.push_back( parents[(parents.size() - 1) - (rand() % 2)][i] );
		}
		return result;
	}
	
	
	std::vector< bool >
	free_crossover_random_HoF(int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		for( int i=0; i<length; ++i )
		{
			result.push_back( parents[rand() % parents.size()][i]);
		}
		return result;
	}
	
	int interpolation(std::vector< double >& x, std::vector< int >& y, double val)
	{
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] == val)
				return y[i];
			if (i < x.size() - 1 && x[i] < val && x[i+1] > val)
				return y[i] + (y[i+1]-y[i])*(val-x[i])/(x[i+1]-x[i]);
		}
		return y[y.size() - 1];
	}
};

//////////////////////////////////////////////////////////////////////

#endif

// End ///////////////////////////////////////////////////////////////
