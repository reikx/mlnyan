package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class LSTMBlock{

    private double output;
    private double memory;
    private double lastMem;
    private double input;

    private int inSize;
    private int outSize;

    private PerceptronModel sigmoid1;
    private PerceptronModel sigmoid2;
    private PerceptronModel tanh;
    private PerceptronModel sigmoid3;

    private double pro1;
    private double pro2;
    private double pro3;
    private double pro4;
    private double learning;

    public LSTMBlock(PerceptronModel firstSig, PerceptronModel secondSig, PerceptronModel thirdSig, PerceptronModel tanh, double learning) {
        this.sigmoid1 = firstSig.clone();
        this.sigmoid2 = secondSig.clone();
        this.sigmoid3 = thirdSig.clone();
        this.tanh = tanh.clone();
        outSize = tanh.getOutputSize();
        this.learning = learning;
        this.lastMem = 0;
        inSize = tanh.getInputSize();
    }

    public void setLastMem(double lastMem) {
        this.lastMem = lastMem;
    }

    public double getMemory() {
        return memory;
    }

    public void setInput(double input) {
        this.input = input;
    }


    public double getOutput() {
        return this.output;
    }


    public double calc() {
        INDArray arr = Nd4j.create(new double[]{input});
        //  arr.put(outSize,input);
        sigmoid1.setInput(arr);
        sigmoid1.calc();
        sigmoid2.setInput(arr);
        sigmoid2.calc();
        tanh.setInput(arr);
        tanh.calc();
        sigmoid3.setInput(arr);
        sigmoid3.calc();
        this.memory = lastMem;
        this.pro1 = memory * sigmoid1.getOutput().getDouble(0,0);
        this.pro2 = sigmoid2.getOutput().getDouble(0,0) * tanh.getOutput().getDouble(0,0);
        this.pro3 = this.pro1 + this.pro2;
        this.pro4 = Math.tanh(this.pro3);
        this.memory = this.pro3;
        this.output = sigmoid3.getOutput().getDouble(0,0) * this.pro4;
        return this.output;
    }



    public double doBackPropagation(double target) {
        double diff = (output - target) * learning;
        return doBackPropagationByDiff(diff);
    }

    public LSTMBlock initial(){
        return new LSTMBlock(sigmoid1,sigmoid2,sigmoid3,tanh,learning);
    }


    public double doBackPropagationByDiff(double diff) {

        double s3d = sigmoid3.doBackPropagationByDiff(Nd4j.create(new double[]{this.pro4})).getDouble(0,0);
        double diff1 = (1 / Math.cosh(memory)) * diff * (this.pro2 / this.pro3);
        double diff2 = (1 / Math.cosh(memory)) * diff * (this.pro1 / this.pro3);
        double s1d = sigmoid1.doBackPropagationByDiff(Nd4j.create(new double[]{lastMem / diff2})).getDouble(0,0);
        double s2d = sigmoid2.doBackPropagationByDiff(Nd4j.create(new double[]{tanh.getOutput().getDouble(0,0) / diff1})).getDouble(0,0);
        double td = tanh.doBackPropagationByDiff(Nd4j.create(new double[]{sigmoid2.getOutput().getDouble(0,0) / diff1})).getDouble(0,0);

        return s3d + s1d + s2d + td;
    }


}