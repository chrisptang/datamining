package com.tangpeng.datamining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RegressionByDiscretization;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.IOException;

public class LogicalRegressionPoorPerformance {

    public static void main(String[] args) throws Exception {
        Instances trainingDataSet = getDataSet("training_set.arff");
        /** Classifier here is Linear Regression */

        Classifier classifier = new RegressionByDiscretization();
        /** */
        classifier.buildClassifier(trainingDataSet);
        /**
         * train the alogorithm with the training data and evaluate the
         * algorithm with testing data
         */
        Evaluation eval = new Evaluation(trainingDataSet);
        Instances testingDataSet = getDataSet("testing_set.arff");
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Linear Regression Evaluation with Datasets **");
        System.out.println(eval.toSummaryString(true));
        System.out.println("=====================================================================");
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(classifier);

        Instance predicationDataSet = getDataSet("prediction_set.arff").lastInstance();
        double value = classifier.classifyInstance(predicationDataSet);
        System.out.println("=======================Prediction Output===============================");
        /** Prediction Output */
        System.out.println(value);
    }

    public static Instances getDataSet(String fileName) throws IOException {
        /**
         * we can set the file i.e., loader.setFile("finename") to load the data
         */
        int classIdx = 1;
        /** the arffloader to load the arff file */
        ArffLoader loader = new ArffLoader();
        //loader.setFile(new File(fileName));
        /** load the traing data */
        loader.setSource(LogicalRegressionPoorPerformance.class.getResourceAsStream("/" + fileName));
        /**
         * we can also set the file like loader3.setFile(new
         * File("test-confused.arff"));
         */
        Instances dataSet = loader.getDataSet();
        /** set the index based on the data given in the arff files */
        dataSet.setClassIndex(classIdx);
        return dataSet;
    }
}
