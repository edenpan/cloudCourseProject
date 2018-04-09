package com.group.stock.predict;

import com.group.stock.model.RecurrentNetsModelConf;
import com.group.stock.representation.PriceCategory;
import com.group.stock.representation.StockDataSetIterator;
import com.group.stock.utils.PlotUtils;
import javafx.util.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
public class StockPricePrediction {

    private static final Logger log = LoggerFactory.getLogger(StockPricePrediction.class);
    private static int exampleLength = 22;

    public static void main(String[] args) throws IOException{
        SparkConf sparkConf = new SparkConf();
        //control whether running in the local or cluster
        boolean useSparkLocal = false;
        int averagingFrequency = 3;
        int batchSizePerWorker = 8;
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        //https://deeplearning4j.org/spark#kryo
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        sparkConf.setAppName("Stock prediction with LSTM");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        String file = "/stockPrice/prices-split-adjusted.csv";
        String symbol = "GOOG";
        int batchSize = 64;
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 100; // training epochs

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
        StockDataSetIterator iterator = new StockDataSetIterator(sc, file, symbol, batchSize, exampleLength, splitRatio, category);
        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        int examplesPerDataSetObject = 1;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(batchSizePerWorker)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .build();
        MultiLayerConfiguration conf = RecurrentNetsModelConf.buildLstmNetworksConf(iterator.inputColumns(), iterator.totalOutcomes());
        //change to spark distribute mode
        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, conf, tm);
        net.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1000)));
        //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring



        log.info("Training...");
        for (int i = 0; i < epochs; i++) {
            while (iterator.hasNext()) {
                DataSet trainData = iterator.next();
//                System.out.println("DataSet: " + trainData);
                List<DataSet> asList = trainData.asList();
                net.fit(sc.parallelize(asList));
            }
            iterator.reset(); // reset iterator
            net.getNetwork().rnnClearPreviousState(); // clear previous state
        }

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net.getNetwork(), locationToSave, true);

        log.info("Load model...");
        net.setNetwork(ModelSerializer.restoreMultiLayerNetwork(locationToSave));

        log.info("Testing...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net.getNetwork(), test, max, min);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net.getNetwork(), test, max, min, category);
        }
        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtils.plot(predicts, actuals, String.valueOf(category));
    }

    private static void predictPriceMultiple (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min) {
        // TODO
    }

    /** Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead */
    private static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\tActual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "\t" + actuals[i]);
        log.info("Plot...");
        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0: name = "Stock OPEN Price"; break;
                case 1: name = "Stock CLOSE Price"; break;
                case 2: name = "Stock LOW Price"; break;
                case 3: name = "Stock HIGH Price"; break;
                case 4: name = "Stock VOLUME Amount"; break;
                default: throw new NoSuchElementException();
            }
            PlotUtils.plot(pred, actu, name);
        }
    }


}
