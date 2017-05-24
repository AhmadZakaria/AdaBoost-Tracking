/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>
#include <climits>
#include <algorithm>
/*
 * Stump
 */

Stump::Stump() :
    dimension_(10),
    splitAttribute_(0),
    splitValue_(0),
    classLabelLeft_(0),
    classLabelRight_(0)
{
    rng.seed(std::random_device()());
}

void Stump::initialize(u32 dimension) {
    dimension_ = dimension;
}

f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel) {
    std::vector<Example> dataR;
    std::vector<Example> dataL;
    dataR.reserve(data.size());
    dataL.reserve(data.size());
    Vector weightsR;
    Vector weightsL;
    weightsR.reserve(data.size());
    weightsL.reserve(data.size());

    for (int d = 0; d < data.size(); ++d) {
        if(data[d].attributes[splitAttribute] < splitValue){
            dataL.push_back(data[d]);
            weightsL.push_back(weights[d]);
        } else {
            dataR.push_back(data[d]);
            weightsR.push_back(weights[d]);
        }
    }
    f32 entropyR = dataR.size() * entropy(dataR, weightsR);
    f32 entropyL = dataL.size() * entropy(dataL, weightsL);
    f32 entropyCurrent = entropy(data, weights);
    f32 gain = entropyCurrent - ((entropyL + entropyR) / data.size());

    u32 lVal = 1;
    f32 errL = weightedError(dataL,weightsL,splitAttribute,splitValue, lVal);
    resultingLeftLabel = (errL > 0) ? 0 : 1;

    return gain;
}


f32 Stump::weightedError(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel) {
    splitAttribute_=splitAttribute;
    splitValue_ = splitValue;
    classLabelLeft_ = resultingLeftLabel;
    classLabelRight_ = 1 - resultingLeftLabel;

    std::vector<u32> classAssignments;
    classAssignments.reserve(data.size());
    classify(data,classAssignments);


    f32 error = 0.0;
    for (int i = 0; i < data.size(); ++i) {
        error += weights[i] * abs(classAssignments[i] - data[i].label);
        //        std::cerr<< "\tWeighted Error: \t" << weights[i]  << std::endl;

    }
    //    std::cout<< "Weighted Error: \t" << error << std::endl;
    f32 weightsSum = std::accumulate(weights.begin(),weights.end(),0.0f);
    return error / weightsSum;
}

f32 Stump::entropy(const std::vector<Example> &data, const Vector& weights)
{
    f32 h = 0.0f;
    std::vector <f32> samplesPerClass(2);
    for (int d = 0; d < data.size(); ++d) {
        samplesPerClass[data[d].label]+= weights[d];
    }

    for (int i = 0; i < samplesPerClass.size(); ++i) {
        double spc = samplesPerClass[i] ;// / data.size();
        if(spc < 0.0000001) //zero (double comparison)
            continue;

        double tempH = spc * log2(spc);
        h += tempH; // Shannon's entropy
    }
    return -h;
}

void Stump::train(const std::vector<Example>& data, const Vector& weights) {
    if(data.empty()) return;

    initialize(data[0].attributes.size());

    // get min and max
    f32 minData = *std::min_element(data[0].attributes.begin(), data[0].attributes.end());
    f32 maxData = *std::max_element(data[0].attributes.begin(), data[0].attributes.end());

    for (u32 i = 0; i < data.size(); i++) {
        f32 minn = *std::min_element(data[i].attributes.begin(), data[i].attributes.end());
        f32 maxx = *std::max_element(data[i].attributes.begin(), data[i].attributes.end());

        minData = std::min(minn, minData);
        maxData = std::max(maxx, maxData);
    }


    std::uniform_int_distribution<std::mt19937::result_type> attrDist(0,dimension_-1);
    std::uniform_real_distribution<> valDist(minData, maxData);

    f32 bestError = std::numeric_limits<f32>::infinity();
    f32 bestGain = - std::numeric_limits<f32>::infinity();
    u32 bestAttr;
    f32 bestAttrVal;
    u32 bestLVal = 1;
    const int N_ITER = 2000;
    for (int iter = 0; (iter < N_ITER) /*|| (bestError > 0.4)*/; ++iter) {
        u32 splitAttribute = attrDist(rng);
        f32 splitValue = valDist(rng);
        u32 lVal;
        f32 gain = weightedGain(data, weights, splitAttribute, splitValue, lVal);
        classLabelLeft_ = lVal;
        classLabelRight_ = 1 - lVal;

        // a combination of error and gain optimization
        if (gain > bestGain){
            bestAttr = splitAttribute;
            bestAttrVal = splitValue;
            bestGain = gain;
            bestLVal = lVal;
        }
        f32 error = weightedError(data, weights, splitAttribute, splitValue, lVal);
        if(error < bestError){
            bestAttr = splitAttribute;
            bestAttrVal = splitValue;
            bestError = error;
        }

    }
    splitAttribute_ = bestAttr;
    splitValue_ = bestAttrVal;
    classLabelLeft_ = bestLVal;
    classLabelRight_ = 1 - bestLVal;
}

u32 Stump::classify(const Vector& v) {
    if(v[splitAttribute_] < splitValue_)
        return classLabelLeft_;
    return classLabelRight_;
}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {
    for (int i = 0; i < data.size(); ++i) {
        classAssignments[i]=classify(data[i].attributes);
    }
}
