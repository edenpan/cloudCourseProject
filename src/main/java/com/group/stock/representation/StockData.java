package com.group.stock.representation;

public class StockData {

    private String date;
    private String symbol;

    private double open;
    private double close;
    private double low;
    private double high;
    private double volume;

    public StockData(){}

    public StockData(String date, String symbol, double open, double close, double low, double volumehigh, double volume){
        this.date = date;
        this.symbol = symbol;
        this.open = open;
        this.close = close;
        this.low = low;
        this.high = high;
        this.volume = volume;

    }

    public String getDate() {
        return date;
    }

    public String getSymbol() {
        return symbol;
    }

    public double getOpen() {
        return open;
    }

    public double getClose() {
        return close;
    }

    public double getLow() {
        return low;
    }

    public double getHigh() {
        return high;
    }

    public double getVolume() {
        return volume;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public void setOpen(double open) {
        this.open = open;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public void setLow(double low) {
        this.low = low;
    }

    public void setHigh(double high) {
        this.high = high;
    }

    public void setVolume(double volume) {
        this.volume = volume;
    }
}
