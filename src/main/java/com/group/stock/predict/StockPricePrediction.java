package com.group.stock.predict;

import com.group.stock.model.RecurrentNetsModelConf;
import com.group.stock.representation.PriceCategory;
import com.group.stock.representation.StockDataSetIterator;
import com.group.stock.utils.PlotUtils;
import javafx.util.Pair;
import org.apache.avro.generic.GenericData;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.spark.earlystopping.SparkDataSetLossCalculator;
import org.deeplearning4j.spark.earlystopping.SparkEarlyStoppingTrainer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.TimeUnit;
public class StockPricePrediction {

    private static final Logger log = LoggerFactory.getLogger(StockPricePrediction.class);
    private static int exampleLength = 22;

    public static void main(String[] args) throws IOException{
        SparkConf sparkConf = new SparkConf();
        //control whether running in the local or cluster
        boolean useSparkLocal = false;
        int averagingFrequency = 10;
        int batchSizePerWorker = 64;
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
        int batchSize = -1;
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 1; // training epochs

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
        StockDataSetIterator iterator = new StockDataSetIterator(sc, file, symbol, batchSize, 1, exampleLength, splitRatio, category);
        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        int examplesPerDataSetObject = 1;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(10)    //Asynchronously prefetch up to 2 batches
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(batchSizePerWorker)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .build();
        MultiLayerConfiguration conf = RecurrentNetsModelConf.buildLstmNetworksConf(iterator.inputColumns(), iterator.totalOutcomes());
        //change to spark distribute mode

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
        //just get all the data and i just set the lenght of iterator is 1.

        List<DataSet> testData = new ArrayList<>();
        while(iterator.hasNext()){
            List<DataSet> temp = iterator.next().asList();
            testData.addAll(temp);
        }
        JavaRDD<DataSet> testRdd = (JavaRDD<DataSet>)sc.parallelize(testData);



        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                        .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(8.5))
                        .scoreCalculator(new SparkDataSetLossCalculator(testRdd, true, sc.sc()))
                        .modelSaver(saver).build();
        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(sc, tm, esConf, net, testRdd);
        log.info("Training...");
        EarlyStoppingResult result = trainer.fit();

        log.info("epoch number: " + result.getTotalEpochs());

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(result.getBestModel(), locationToSave, true);
//        result.getBestModel().fit();
        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);


        log.info("Testing...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net, test, max, min);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net, test, max, min, category);
        }
        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];

        for (int i = 0; i < testData.size(); i++) {
            //here is use the testdata key that is the previous 21days number to predict next day .and with some transform.
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }

        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
//        PlotUtils.plot(predicts, actuals, String.valueOf(category));
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
