package com.group.stock.utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.*;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PlotUtils {

    private static final Logger log = LoggerFactory.getLogger(PlotUtils.class);

    public static void plot(double[] predicts, double[] actuals, String name){
        double[] index = new double[predicts.length];
        for (int i = 0; i < predicts.length; i++)
            index[i] = i;

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("MM/dd HH:mm:ss");
        LocalDateTime now = LocalDateTime.now();
//        System.out.println(dtf.format(now));
        int min = minValue(predicts, actuals);
        int max = maxValue(predicts, actuals);
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, index, predicts, "predicts");
        addSeries(dataSet, index, actuals, "Actuals");
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Prediction Result",
                "Index",
                name,
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        XYPlot xyPlot = chart.getXYPlot();
        //X-axis
        final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
        domainAxis.setRange((int) index[0], (int)(index[index.length - 1]));
        domainAxis.setTickUnit(new NumberTickUnit(20));
        domainAxis.setVerticalTickLabels(true);

        //Y -axis
        final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
        rangeAxis.setRange(min, max);
        rangeAxis.setTickUnit(new NumberTickUnit(50));
        rangeAxis.setVerticalTickLabels(true);
        final ChartPanel panel = new ChartPanel(chart);
        try {

            OutputStream out = new FileOutputStream("src/main/resources/predict" + dtf.format(now) +  ".png");
            ChartUtilities.writeChartAsPNG(out,
                    chart,
                    panel.getWidth(),
                    panel.getHeight());

        } catch (IOException ex) {
            log.error(ex.getLocalizedMessage());
        }
//        final JFrame f = new JFrame();
//        f.add(panel);
//        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        f.pack();
//        f.setVisible(true);

    }

    private static void addSeries(final XYSeriesCollection dataSet, double[] x, double[] y, final String label){
        final XYSeries s = new XYSeries(label);
        for (int j = 0; j < x.length; j++)
            s.add(x[j], y[j]);
        dataSet.addSeries(s);
    }

    private static int minValue(double[] predicts, double[] actuals){
        double min = Integer.MAX_VALUE;
        for(int i = 0; i < predicts.length; i++){
            if(min > predicts[i]) min = predicts[i];
            if(min > actuals[i]) min = actuals[i];
        }

        return (int)(min * 0.98);

    }

    private static int maxValue(double[] predicts, double[] actuals){
        double max = Integer.MIN_VALUE;
        for (int i = 0; i < predicts.length; i++){
            if(max < predicts[i]) max = predicts[i];
            if(max < actuals[i]) max = actuals[i];
        }
        return (int)(max*1.02);

    }

}
