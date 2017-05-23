/*
 * NearestMeanClassifier.hh
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#ifndef NEARESTMEANCLASSIFIER_HH_
#define NEARESTMEANCLASSIFIER_HH_

#include <vector>
#include "Types.hh"

class Stump
{
private:
	u32 dimension_;
	u32 splitAttribute_;
	f32 splitValue_;
	u32 classLabelLeft_;
	u32 classLabelRight_;

    f32 weightedError(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel);
    f32 weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel);
    f32 entropy(const std::vector<Example>& data, const Vector &weights);
public:
	Stump();
	void initialize(u32 dimension);
	void train(const std::vector<Example>& data, const Vector& weights);
	u32 classify(const Vector& v);
	void classify(const std::vector<Example>& data, std::vector<u32>& classAssignments);
};

#endif /* NEARESTMEANCLASSIFIER_HH_ */
