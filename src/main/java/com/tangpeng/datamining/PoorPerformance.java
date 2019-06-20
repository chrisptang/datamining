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
        int initialK = 3, k = initialK, size = 20;
        double[] squaredErrors = new double[size - k + 1];
        for (; k <= size; k++) {
            kMeans = new weka.clusterers.SimpleKMeans();
            kMeans.setNumClusters(k);
            kMeans.setPreserveInstancesOrder(true);
            kMeans.setDisplayStdDevs(true);
            kMeans.setSeed(2 * k);
            kMeans.buildClusterer(testingDataSet);
            PrintUtil.print(String.format("======================= K = %s ===============================", k));
            PrintUtil.print(String.format("K=%s, SquaredError:%s", k, kMeans.getSquaredError()));
            squaredErrors[k - initialK] = kMeans.getSquaredError();
            PrintUtil.print(kMeans.toString());
            PrintUtil.print(String.format("======================= End of K = %s ===============================", k));

//            int[] assignments = kMeans.getAssignments();
//            int i = 0;
//            for (int clusterNum : assignments) {
//                System.out.printf("Instance %d -> Cluster %d;\n", i, clusterNum);
//                i++;
//            }
        }

        PrintUtil.print(squaredErrors);
    }

    private static void logicalRegression() throws Exception {
        Instances trainingDataSet = getDataSet("training_set.arff");
        /** Classifier here is Linear Regression */


        Evaluation eval = new Evaluation(trainingDataSet);
        Instances testingDataSet = getDataSet("testing_set.arff");
        Classifier classifier = new LinearRegression();
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        PrintUtil.print("** Linear Regression Evaluation with Datasets **");
        PrintUtil.print(eval.toSummaryString(true));
        PrintUtil.print("=====================================================================");
        System.out.print(" the expression for the input data as per alogorithm is ");
        PrintUtil.print(classifier);

        Instance predicationDataSet = getDataSet("prediction_set.arff").firstInstance();
        double value = classifier.classifyInstance(predicationDataSet);
        PrintUtil.print("=======================Prediction Output===============================");
        /** Prediction Output */
        PrintUtil.print(value);
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
