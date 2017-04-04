/**
 * CMA-VNS: Covariance Matrix Adaptation Variable neighborhood search
 * Copyright (c) 2015 Hong Kong Polytechnic University
 * Author: Frank F. Xue <fan.xue@polyu.edu.hk; fanxue@outlook.com>
 *         Geoffrey Q.P. Shen <bsqpshen@polyu.edu.hk>
 * This file is for CBBOC2015 <http://web.mst.edu/~tauritzd/CBBOC/>.
 *
 * CMA-VNS is free software: you can redistribute it and/or modify
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
 */
 
#ifndef CMAVNS_COMPETITOR_HPP
#define CMAVNS_COMPETITOR_HPP

//////////////////////////////////////////////////////////////////////

// #include "cbboc2015/CBBOC2015.hpp"
#include "cbboc2015/CBBOC2015Util.hpp"
#include "cbboc2015/Competitor.hpp"
#include "cbboc2015/ObjectiveFn.hpp"
#include "cbboc2015/ProblemClass.hpp"
#include "cbboc2015/TrainingCategory.hpp"
#include "cbboc2015/RNG.hpp"

#include <algorithm>
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
 * CMA-VNS: Covariance Matrix Adaptation Variable neighborhood search
 *
 * CMA-VNS is a hyperheuristic which employs a CMA-ES evolution which
 * is followed by an iterated local search. The reason of employing 
 * the CMA-ES is the efficient estimation of variables in CMA-ES. Hence
 * the local search, variable neighborhood search in this context, can
 * be facilitated against expensive (and limited) evaluations.
 *
 * The parameters of the hyperheuristic were trained beforehand.
 *
 */
 
class CMAVNSCompetitor : public Competitor 
{
private:
	// ptr to objective function
	ObjectiveFn *objfunc;
	// search history
	std::unordered_map<std::vector<bool>, double> history;
	// hall of fame
	std::vector<std::vector<bool>> halloffame;
	int cmaHoFsize;
	// tabu list
	std::unordered_set<std::vector<bool>> tabu;
public:

	CMAVNSCompetitor() { objfunc = 0;}

	virtual ~CMAVNSCompetitor() {}

	///////////////////////////////

	virtual TrainingCategory getTrainingCategory() const { return TrainingCategory::NONE; }

	virtual void train( std::vector<ObjectiveFn>& trainingSet, long long maxTimeInMilliseconds ) 
	{
		// CMAVNS was trained beforehand, so we simply skip the online training by testing random solutions
		throw new std::logic_error( "Invalid call to CMAVNSCompetitor.train()" );
	}

	// main process for CBBOC call
	virtual void test( ObjectiveFn& testCase, long long maxTimeInMilliseconds ) 
	{
		bool FixedHoF = false;
		bool BB_restarts = false;
		bool AdaptiveAccept = false;
		bool AAhighCand = true;
		bool AAnearCand = false;
		for (auto h : history)
		{
			auto s = h.first;
			s.clear();
		}
		history.clear();
		halloffame.clear();
		tabu.clear();
		cmaHoFsize = 0;
 		int startEval = testCase.getRemainingEvaluations();
		objfunc = &testCase;		
		const int dim = testCase.getNumGenes();
		
		double sigma = 0.5;
		const double EvDiscount = 1.0 * dim * dim / startEval;
		int targetLambda = round(dim/ 8.0);
		int lambda = targetLambda ;
		double ESratio = 0.7;
		double AAratio = 0.5; 
		int BBtrials = 20;
		
		/**
		*
		* Here follows hard coded rules trained from millions of compuational experiments ...
		*
		* TODO: to be updated to some elegant version :)
		*/
		if (EvDiscount > 2 && EvDiscount <= 8)
		{
			// if (2 < discount <= 8) 	
			// (ODD) round(dim/8.0) or round(dim/8.0) -1 
			targetLambda = round(dim/ 8.0);
			lambda = targetLambda % 2 == 1 ? targetLambda : targetLambda - 1;
			// 0.65 small instances / 0.6 large instances
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
			cmaparams.set_max_fevals(testCase.getRemainingEvaluations());
			cmaparams.set_noisy();
			cmaparams.set_sep();
			
			ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> optim(fitfunc,cmaparams);
			optim.set_progress_func(CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>::_defaultPFunc);
				
			if (ESratio < 1.0)
			{
				
				double lastbest = -1;
				int thisiter = 0;
				while(!optim.stop() && (testCase.getRemainingEvaluations() > startEval * ESratio || halloffame.size() < (EvDiscount >= 8 ? 2 : 3)))
				{
					dMat candidates = optim.ask();
					optim.eval(candidates);
					optim.tell();
					optim.inc_iter(); // important step: signals next iteration.

					Candidate cmasol1 = optim.get_solutions().get_best_seen_candidate();
					// std::cout << (startEval - testCase.getRemainingEvaluations()) << "," << - cmasol1.get_fvalue() << ",CMA" << std::endl;
					
					double value = cmasol1.get_fvalue();
					if (((EvDiscount >= 8 && dim < 200)  || lastbest == value) && testCase.getRemainingEvaluations() < startEval * (EvDiscount >= 8 ? 0.95 : 0.9) )
					{
						std::vector< bool > neighbor;
						for (int i = 0; i < dim; i++)
							neighbor.push_back(cmasol1.get_x_ptr()[i] >= 0.5);
						addHoF(neighbor);
					}
					lastbest = value;
				}
			}
		
		
		
			if (ESratio > 0 && testCase.getRemainingEvaluations() > 0)
			{
				cmaHoFsize = halloffame.size();
				// std::cout << cmaHoFsize << " " << lambda << " " << startEval << " " << dim << " " << EvDiscount << " " << (1.0 * testCase.getRemainingEvaluations()/startEval) << std::endl;
				int sumTests = testCase.getRemainingEvaluations();
				int lastRunPassed = 0;
				long eliteStarts = 0;
				while (testCase.getRemainingEvaluations() > 0)
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
							std::vector< bool > neighbor = backbone_bitvector(testCase.getNumGenes(), halloffame);
							const bool hashed = history.end() != history.find( neighbor );
							double value = hashed ? history.find( neighbor )->second : testCase.value( neighbor );
							if (!hashed)
								history.insert(std::make_pair<std::vector<bool>&,double&>(neighbor, value));
							if( value > bestValue ) {
								incumbent = neighbor;
								bestValue = value;
								// if (value > history.find(halloffame[halloffame.size() - 1])->second)
									// break;
								// std::cout << (startEval - testCase.getRemainingEvaluations()) << "," << bestValue << ",ILS" << std::endl;
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
					double bestSoFar = bestValue;
					
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
						for( size_t i=0; i<neighbors.size() && testCase.getRemainingEvaluations() > 0; ++i )
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
								if (i >= testCase.getNumGenes())
									noImprov.clear();
								// std::cout << (startEval - testCase.getRemainingEvaluations()) << "," << bestValue << ",ILS" << eliteStarts << std::endl;
							}
							else if (!improved && i >= testCase.getNumGenes() && /* (objVForAA == 0 || value < objVForAA) &&*/ eliteStarts >= 2 && AdaptiveAccept && tabu.find(neighbor) == tabu.end())
							{
								// std::cout << "*";
								if (objVForAA == 0 || !AAhighCand || value > objVForAA)
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
							}
							if (value < initV)
							{
								if (i < testCase.getNumGenes())
									noImprov.insert(i);
							}
						}
						
						// enclose of a trial
						if (!improved && noImprov.size() == testCase.getNumGenes())
						{
							// fine check
							noImprov.clear();
							
							std::vector< std::vector< bool > > neighbors = hamming1Neighbours(incumbent);
							
							double initV1 = bestValue;
							for( size_t i=0; i<neighbors.size() && testCase.getRemainingEvaluations() > 0; ++i )
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
								}
								else if (value < initV1)
								{
									noImprov.insert(i);
								}
							}
						}
						
						if (!improved && AdaptiveAccept && eliteStarts >= 2 && objVForAA > 0 )
						{
							tabu.insert(candForAA);
							improved = true;
							incumbent = candForAA;
							bestValue = objVForAA;
							noImprov.clear();
							// std::cout << (startEval - testCase.getRemainingEvaluations()) << "," << bestValue << ",ILS-AA-" << tabu.size()<< "-" << history.size() << std::endl;
							lastRunPassed = 0;
						}
					} while(( improved)&& testCase.getRemainingEvaluations() > 0 );
					
					addHoF(incumbent);
					noImprov.clear();
				}
			}
			
		}
		while (testCase.getRemainingEvaluations() > 0);
	}

	///////////////////////////////
	// Lambda  function of wrapping fitness function for libcmaes
	FitFunc fitfunc = [&](const double *x, const int N)
	{
		std::vector< bool > cand;
		for (int j = 0; j < N; j++)
			cand.push_back(x[j] >= 0.5);
		/**
		*
		* Here is a known inefficient implementation. If a solution can hit the history 
		* record, there is absolutely no need to call objfunc->value()
		*
		* The 2016 version has revised this.
		*/
		double value = objfunc->value( cand );
		history.insert(std::make_pair<std::vector<bool>&,double&>(cand, value));
		return - value;
	};
	
	// simple neighborhood of 1-hamming distance
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent ) {
		std::vector< std::vector< bool > > result;
		for( size_t i=0; i<incumbent.size(); ++i ) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		
		return result;
	}
	
	// variable neighborhood consisting of 1-hamming distance, Adaptive acceptance, and CMA-ES guided solutions
	std::vector< std::vector< bool > >
	hamming1Neighbours( const std::vector< bool >& incumbent, std::unordered_set<size_t>& banned, ESOptimizer<BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>,CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions> &optim, bool CMA, bool AA) {
		std::vector< std::vector< bool > > result;
		for( size_t i=0; i<incumbent.size(); ++i ) {
			std::vector< bool > neighbour = incumbent;
			neighbour[ i ] = !neighbour[ i ];
			result.push_back( neighbour );
		}
		if (banned.size() > incumbent.size()/2 || AA)
		{
			// flip all cands
			{
				std::vector< bool > neighbour = incumbent;
				for( size_t i=0; i<incumbent.size(); ++i )
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
				for ( size_t j = 0; j<(incumbent.size() - banned.size()) + (AA ? 3 : 0); ++j )
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
						for ( size_t i=0; i<incumbent.size(); ++i )
							if (banned.end() == banned.find(i))
								neighbour[i] = cand.data()[i] >=0.5;
					}
					else
					{
						for( size_t i=0; i<incumbent.size(); ++i )
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
	
	// add a cool solution to "Hall of Fame"
	void addHoF(const std::vector< bool >& incumbent)
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
	
	// get the hamming distance between two solutions
	int hamming(const std::vector< bool >& incumbent1, const std::vector< bool >& incumbent2)
	{
		int ret = 0;
		for (int i = 0; i < incumbent1.size(); i++)
			ret += incumbent1[i] == incumbent2[i] ? 0 : 1;
		return ret;
	}
	
	// get the backbone vector of a given set
	std::vector< bool >
	backbone_bitvector( int length, std::vector<std::vector<bool>>& parents) 
	{
		std::vector< bool > result;
		for( int i=0; i<length; ++i )
		{
			result.push_back( parents[(parents.size() - 1) - (rand() % 2)][i] );
		}
		return result;
	}
	
	/* obsoleted
	
	std::vector< bool >
	backbone_random( int length, std::vector<std::vector<bool>>& parents) 
	{
		int p1 = rand() % parents.size();
		int p2 = rand() % parents.size();
		std::vector< bool > result;
		for( int i=0; i<length; ++i )
		{
			result.push_back( rand()%2 ? parents[p1][i] : parents[p2][i]);
		}
		return result;
	}
	*/
	
};

//////////////////////////////////////////////////////////////////////

#endif

// End ///////////////////////////////////////////////////////////////
