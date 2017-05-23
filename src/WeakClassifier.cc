/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>
#include <random>
#include <climits>

/*
 * Stump
 */

Stump::Stump() :
    dimension_(10),
    splitAttribute_(0),
    splitValue_(0),
    classLabelLeft_(0),
    classLabelRight_(0)
{}

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


    // get how many ones
    int label1 = std::accumulate(dataL.begin(), dataL.end(),
                                 0, // start with first element
                                 [](int a, Example b) {
        return a + b.label;
    });
    int label0 = data.size() - label1;
    resultingLeftLabel = 0;//(label1 > label0)? 1 : 0;


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

    return error;
}

f32 Stump::entropy(const std::vector<Example> &data, const Vector& weights)
{
    f32 h = 0.0f;
    std::vector <f32> samplesPerClass(2);
    for (int d = 0; d < data.size(); ++d) {
        samplesPerClass[data[d].label]+= 1;//weights[d];
    }

    for (int i = 0; i < samplesPerClass.size(); ++i) {
        double spc = samplesPerClass[i] / data.size();
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

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> attrDist(0,dimension_-1);
    std::uniform_int_distribution<std::mt19937::result_type> valDist(1, 4);

    //    f32 bestError = std::numeric_limits<f32>::infinity();
    f32 bestGain = - std::numeric_limits<f32>::infinity();
    u32 bestAttr;
    f32 bestAttrVal;
    u32 bestLVal = 1;
    //    const int N_ITER = 10000;
    //    for (int iter = 0; iter < N_ITER; ++iter) {
    for (int splitattr = 0; splitattr < dimension_; ++splitattr)
        for (int val = 1; val < 5; ++val) {
            //        u32 splitAttribute = attrDist(rng);
            //        f32 splitValue = valDist(rng);
            u32 splitAttribute = splitattr;
            f32 splitValue = val;
            u32 lVal;
            f32 gain = weightedGain(data, weights, splitAttribute, splitValue, lVal);
            classLabelLeft_ = lVal;
            classLabelRight_ = 1 - lVal;
            if (gain > bestGain){
                bestAttr = splitAttribute;
                bestAttrVal = splitValue;
                bestGain = gain;
                bestLVal = lVal;
            }
            //        f32 error = weightedError(data, weights, splitAttribute, splitValue, lVal);
            //        if(error < bestError){
            //            bestAttr = splitAttribute;
            //            bestAttrVal = splitValue;
            //            bestError = error;
            //        }
            //        if (error < 0.5)
            //            break; // barely better than random
            //        std::cout<< "\t\tGain: " << gain << "\t Best: "<<bestGain<<std::endl;
        }
    std::cout<< "Best Gain: " << bestGain << std::endl;
    splitAttribute_ = bestAttr;
    splitValue_ = bestAttrVal;
    classLabelLeft_ = bestLVal;
    classLabelRight_ = 1 - bestLVal;


    //    std::cout<< "\t\tError: \t" << bestError << ", For "<< bestAttr<< ", "<< bestAttrVal<< std::endl;

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
