package com.tangpeng.datamining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.IOException;

public class PoorPerformance {

    public static void main(String[] args) throws Exception {
        simpleKMeans();
    }

    private static void simpleKMeans() throws Exception {
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(10);
        kMeans.setPreserveInstancesOrder(true);
        kMeans.setDisplayStdDevs(true);
        kMeans.setSeed(20);

//        Instances testingDataSet = getDataSet("weather.arff");
        Instances testingDataSet = getDataSet("poor_perf_set.arff");
        kMeans.setPreserveInstancesOrder(true);
        kMeans.buildClusterer(testingDataSet);
        int[] assignments = kMeans.getAssignments();
        int i = 0;
        for (int clusterNum : assignments) {
            System.out.printf("Instance %d -> Cluster %d;\n", i, clusterNum);
            i++;
        }
        System.out.println(kMeans.toString());
    }

    private static void logicalRegression() throws Exception {
        Instances trainingDataSet = getDataSet("training_set.arff");
        /** Classifier here is Linear Regression */


        Evaluation eval = new Evaluation(trainingDataSet);
        Instances testingDataSet = getDataSet("testing_set.arff");
        Classifier classifier = new LinearRegression();
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Linear Regression Evaluation with Datasets **");
        System.out.println(eval.toSummaryString(true));
        System.out.println("=====================================================================");
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(classifier);

        Instance predicationDataSet = getDataSet("prediction_set.arff").firstInstance();
        double value = classifier.classifyInstance(predicationDataSet);
        System.out.println("=======================Prediction Output===============================");
        /** Prediction Output */
        System.out.println(value);
    }

    public static Instances getDataSet(String fileName) throws IOException {
        /**
         * we can set the file i.e., loader.setFile("finename") to load the data
         */
//        int classIdx = 1;
        /** the arffloader to load the arff file */
        ArffLoader loader = new ArffLoader();
        //loader.setFile(new File(fileName));
        /** load the traing data */
        loader.setSource(PoorPerformance.class.getResourceAsStream("/" + fileName));
        /**
         * we can also set the file like loader3.setFile(new
         * File("test-confused.arff"));
         */
        Instances dataSet = loader.getDataSet();
        /** set the index based on the data given in the arff files */
//        dataSet.setClassIndex(classIdx);
        return dataSet;
    }
}
