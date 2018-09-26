package mlnyan.ml.lstm;

import mlnyan.ml.MatrixUtil;
import mlnyan.ml.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


public class LSTMBlock{

    private INDArray output;
    private INDArray memory;
    private INDArray lastMem;
    private INDArray input;
    private int inSize;
    private int outSize;
    private Model sigmoid1;
    private Model sigmoid2;
    private Model tanh;
    private Model sigmoid3;

    private Model sigmoid1Dif;
    private Model sigmoid2Dif;
    private Model tanhDif;
    private Model sigmoid3Dif;

    private INDArray pro1;
    private INDArray pro2;
    private INDArray pro3;
    private INDArray pro4;
    private INDArray pro5;
    private double learning;

    public LSTMBlock(Model firstSig,Model secondSig,Model thirdSig,Model tanh,INDArray lastMem,double learning) {
        this.sigmoid1 = firstSig;
        this.sigmoid2 = secondSig;
        this.sigmoid3 = thirdSig;
        this.tanh = tanh;
        outSize = tanh.layers[tanh.layers.length - 1].perceptron.rows();
        this.output = Nd4j.zeros(outSize);
        this.memory = Nd4j.zeros(outSize);
        this.lastMem = lastMem.dup();
        this.learning = learning;
        inSize = tanh.layers[0].perceptron.rows();
    }


    @Override
    public void setInput(INDArray input) {
        this.input = input.dup();
    }

    @Override
    public INDArray getOutput() {
        return this.output.dup();
    }

    @Override
    public INDArray calc() {
        INDArray arr = Nd4j.zeros(inSize,1);
        for (int i = 0;i < outSize;++i){
            arr.put(i,0,output.getDouble(i,0));
        }
        for (int i = 0;i < input.rows();++i){
            arr.put(outSize + i,0,input.getDouble(i,0));
        }
        //  arr.put(outSize,input);
        sigmoid1.setInput(arr);
        sigmoid1.calc();
        sigmoid2.setInput(arr);
        sigmoid2.calc();
        tanh.setInput(arr);
        tanh.calc();
        sigmoid3.setInput(arr);
        sigmoid3.calc();
        this.memory = lastMem.dup();
        this.pro1 = memory.mul(sigmoid1.getOutput());
        this.pro2 = sigmoid2.getOutput().mul(tanh.getOutput());
        this.pro3 = this.pro1.add(this.pro2);
        this.pro4 = Transforms.tanh(this.pro3);
            /*   this.pro3 = this.pro1.add(this.pro2);
            this.pro4 = Transforms.tanh(this.pro3);
            this.pro5 = sigmoid3.getOutput().mul(this.pro4);
            this.memory = this.pro3.dup();
            this.output = this.pro5.dup();*/
        this.memory = this.pro1.add(this.pro2);
        this.output = sigmoid3.getOutput().mul(this.pro4);
        return this.output.dup();
    }


    @Override
    public void doBackPropagation(INDArray target) {
        INDArray diff = output.sub(target).mul(learning);
        doBackPropagationByDiff(diff);
    }

    @Override
    public void doBackPropagationByDiff(INDArray diff) {
        sigmoid1Dif = sigmoid1.clone();
        sigmoid2Dif = sigmoid2.clone();
        sigmoid3Dif = sigmoid3.clone();
        tanhDif = tanh.clone();

        sigmoid3Dif.doBackPropagationByDiff(this.pro4.mul(diff));
        INDArray diff1 = MatrixUtil.tanhDeriv(memory).mul(diff).mul(this.pro2.div(this.pro3));
        INDArray diff2 = MatrixUtil.tanhDeriv(memory).mul(diff).mul(this.pro1.div(this.pro3));
        sigmoid1Dif.doBackPropagationByDiff(lastMem.mul(diff2));
        sigmoid2Dif.doBackPropagationByDiff(tanh.getOutput().mul(diff1));
        tanhDif.doBackPropagationByDiff(sigmoid2Dif.getOutput().mul(diff1));
        for (int i = 0;i < sigmoid1.layers.length - 1;++i){
            sigmoid1Dif.layers[i].weight = sigmoid1Dif.layers[i].weight.sub(sigmoid1.layers[i].weight);

            sigmoid1Dif.layers[i].bias = sigmoid1Dif.layers[i].bias.sub(sigmoid1.layers[i].bias);
        }
        for (int i = 0;i < sigmoid2.layers.length - 1;++i){
            sigmoid2Dif.layers[i].weight = sigmoid2Dif.layers[i].weight.sub(sigmoid2.layers[i].weight);

            sigmoid2Dif.layers[i].bias = sigmoid2Dif.layers[i].bias.sub(sigmoid2.layers[i].bias);
        }
        for (int i = 0;i < sigmoid3.layers.length - 1;++i){
            sigmoid3Dif.layers[i].weight = sigmoid3Dif.layers[i].weight.sub(sigmoid3.layers[i].weight);

            sigmoid3Dif.layers[i].bias = sigmoid3Dif.layers[i].bias.sub(sigmoid3.layers[i].bias);
        }
        for (int i = 0;i < tanh.layers.length - 1;++i){
            tanhDif.layers[i].weight = tanhDif.layers[i].weight.sub(tanh.layers[i].weight);

            tanhDif.layers[i].bias = tanhDif.layers[i].bias.sub(tanh.layers[i].bias);
        }
    }


}