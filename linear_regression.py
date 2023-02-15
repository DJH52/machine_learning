#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Name: linear_regression.py
Author: Dainel Holmes
Created: 2023-02-16
Last Modified: 2023-02-16
Version: 1.0

Description: Linear regression from scratch using MNIST dataset, a statistical method modelling the relationship between a dependent variable and one or more independent variables.
"""


import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk

def LinearRegression(n_features, learning_rate, n_iterations):
    
        #initialize the parameters
        slope, intercept = initialize_parameters(n_features)
    
        #calculate the predicted value
        y_pred = calculate_predicted_value(slope, intercept, x)
    
        #calculate the cost
        cost = calculate_cost(y, y_pred)
    
        #update the parameters
        slope, intercept = update_parameters(slope, intercept, x, y, y_pred, learning_rate)
    
        #calculate the predicted value
        y_pred = calculate_predicted_value(slope, intercept, x)
    
        #calculate the cost
        cost = calculate_cost(y, y_pred)
    
        #return the slope and intercept
        return slope, intercept

#Initialize the parameters: Initialize the slope and intercept of the regression line
def initialize_parameters(n_features):
    #initialize the slope and intercept to zero
    slope = np.zeros(n_features)
    intercept = 0

    #return the slope and intercept
    return slope, intercept

#Calculate the predicted value: Calculate the predicted value of the dependent variable using the slope and intercept of the regression line
def calculate_predicted_value(slope, intercept, x):
    #calculate the predicted value
    y_pred = np.dot(x, slope) + intercept

    #return the predicted value
    return y_pred

def calculate_cost(y, y_pred):
    #calculate the cost
    cost = np.mean(np.square(y - y_pred))

    #return the cost
    return cost

#Update the parameters: Update the slope and intercept of the regression line using the learning rate
def update_parameters(slope, intercept, x, y, y_pred, learning_rate):#
    #calculate the slope and intercept
    slope = slope - (2 * learning_rate * np.dot(x.T, (y_pred - y)) / len(x))
    intercept = intercept - (2 * learning_rate * np.sum(y_pred - y) / len(x))

    #return the slope and intercept
    return slope, intercept

#main function to run the script from
def main():
    
    #import the MNIST dataset from sklearn
    dataset = sk.datasets.load_digits()

    #split the dataset into training and testing data with 80% training and 20% testing data manually using the percentage
    #of the dataset size
    train_size = int(len(dataset.data) * 0.8)
    test_size = int(len(dataset.data) * 0.2)

    #split the dataset into training and testing data
    train_data = dataset.data[:train_size]
    train_labels = dataset.target[:train_size]

    test_data = dataset.data[train_size:]
    test_labels = dataset.target[train_size:]

    #create a linear regression model from scratch
    model = LinearRegression()





#run the main function
if __name__ == '__main__':
    main()