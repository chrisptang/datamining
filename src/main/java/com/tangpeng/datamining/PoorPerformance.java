package com.tangpeng.datamining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static com.tangpeng.datamining.PrintUtil.print;

public class PoorPerformance {

    public static void main(String[] args) throws Exception {
        simpleKMeansFull();
    }

    private static void simpleKMeans() throws Exception {
        SimpleKMeans kMeans;

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
            print(String.format("======================= K = %s ===============================", k));
            print(String.format("K=%s, SquaredError:%s", k, kMeans.getSquaredError()));
            squaredErrors[k - initialK] = kMeans.getSquaredError();
            print(kMeans.toString());
            print(String.format("======================= End of K = %s ===============================", k));
        }

        print(squaredErrors);
    }

    private static void simpleKMeansFull() throws Exception {
        final int k = 18;
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(k);
        kMeans.setPreserveInstancesOrder(true);
        kMeans.setDisplayStdDevs(true);
        kMeans.setSeed(2 * k);

        Instances testingDataSet = getDataSet("poor_perf_set.arff");

        kMeans.buildClusterer(testingDataSet);
        print(String.format("======================= K = %s ===============================", k));
        print(String.format("K=%s, SquaredError:%s", k, kMeans.getSquaredError()));
        print(kMeans.toString());

        Instances clusterCentroids = kMeans.getClusterCentroids();
        Set<Integer> poorPerformanceClusters = new HashSet<Integer>();
        for (int cluster = 0; cluster < clusterCentroids.numInstances(); cluster++) {
            Instance clusterCentroid = clusterCentroids.instance(cluster);

            print(String.format("Cluster #%s, Centroids: %s\n", cluster, clusterCentroid.toString()));

            double roiClusterCentroids = clusterCentroid.value(clusterCentroid.numValues() - 1);
            if (roiClusterCentroids <= 3.0D) {
                print(String.format("====\nPoor performance Cluster #%s, roi:%s\n====\n", cluster, Utils.doubleToString(roiClusterCentroids, 4)));
                poorPerformanceClusters.add(cluster);
            }
        }
        int[] assignments = kMeans.getAssignments();
        int i = 0;
        for (int clusterNum : assignments) {
            if (poorPerformanceClusters.contains(clusterNum)) {
                print(String.format("Poor Performance Instance %s -> Cluster %s;", i, clusterNum));
            }
            i++;
        }
    }

    private static void logicalRegression() throws Exception {
        Instances trainingDataSet = getDataSet("training_set.arff");
        /** Classifier here is Linear Regression */


        Evaluation eval = new Evaluation(trainingDataSet);
        Instances testingDataSet = getDataSet("testing_set.arff");
        Classifier classifier = new LinearRegression();
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        print("** Linear Regression Evaluation with Datasets **");
        print(eval.toSummaryString(true));
        print("=====================================================================");
        print(" the expression for the input data as per alogorithm is ");
        print(classifier);

        Instance predicationDataSet = getDataSet("prediction_set.arff").firstInstance();
        double value = classifier.classifyInstance(predicationDataSet);
        print("=======================Prediction Output===============================");
        /** Prediction Output */
        print(value);
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
