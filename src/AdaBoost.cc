/*
 * AdaBoost.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "AdaBoost.hh"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

AdaBoost::AdaBoost(u32 nIterations) :
    nIterations_(nIterations)
{}

void AdaBoost::normalizeWeights() {
    f32 sum = std::accumulate(weights_.begin(), weights_.end(), 0.0f);
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] /= sum;
    }
}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {

    for (int i = 0; i < data.size(); ++i) {
        u32 exponent = 1 - abs(classAssignments[i] - data[i].label);
        //        std::cout<<"exponent "<< pow(classifierWeights_[iteration], exponent)<<std::endl;
        weights_[i] = weights_[i] * pow(classifierWeights_[iteration], exponent);
    }
}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {
    f32 err = 0.0f;
    for (int n = 0; n < data.size(); ++n) {
        err += weights_[n] * abs(classAssignments[n] - data[n].label);
    }
    return err;
}

void AdaBoost::initialize(std::vector<Example>& data) {
    // initialize weak classifiers
    for (u32 iteration = 0; iteration < nIterations_; iteration++) {
        weakClassifier_.push_back(Stump());
    }
    // initialize classifier weights
    classifierWeights_.resize(nIterations_);
    // initialize weights
    weights_.resize(data.size());
    f32 init_weight = 1.0f / data.size();
    f32 minData = *std::min_element(data[0].attributes.begin(), data[0].attributes.end());
    f32 maxData = *std::max_element(data[0].attributes.begin(), data[0].attributes.end());

    for (u32 i = 0; i < data.size(); i++) {
        weights_.at(i) = init_weight;
        f32 minn = *std::min_element(data[i].attributes.begin(), data[i].attributes.end());
        f32 maxx = *std::max_element(data[i].attributes.begin(), data[i].attributes.end());

        minData = std::min(minn, minData);
        maxData = std::max(maxx, maxData);
    }
}

void AdaBoost::trainCascade(std::vector<Example>& data) {
    for (u32 iteration = 0; iteration < nIterations_; iteration++) {
        //        std::cout << "Training classifier\t" << iteration ;
        weakClassifier_[iteration].train(data, weights_);
        std::vector<u32> classAssignments;
        classAssignments.reserve(data.size());
        weakClassifier_[iteration].classify(data, classAssignments);

        f32 err = weightedErrorRate(data, classAssignments);
        classifierWeights_[iteration] =  err / (1.0f - err);
        //        std::cout << "\t Error: " << classifierWeights_[iteration] << std::endl;

        updateWeights(data, classAssignments, iteration);

        normalizeWeights();

    }
}

u32 AdaBoost::classify(const Vector& v) {
    f32 conf0 = confidence(v, 0);
    f32 conf1 = confidence(v, 1);
    if(conf0 > conf1) return 0;
    return 1;
}

f32 AdaBoost::confidence(const Vector& v, u32 k) {
    f32 conf = 0.0f;
    for (u32 iteration = 0; iteration < nIterations_; iteration++) {
        conf += log2(1.0f / classifierWeights_[iteration]) * (1 - abs(weakClassifier_[iteration].classify(v) - k));
    }
    return conf;
}
