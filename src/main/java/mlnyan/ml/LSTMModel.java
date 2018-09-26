package mlnyan.ml;

import mlnyan.ml.ModelBase;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LSTMModel extends ModelBase {

    Layer[] layers;
    LSTMLayer[] lstmLayers;
    LSTMBlock block;
    private final double learning;
    public LSTMModel(LSTMBlock lstmBlock,double learning,Layer... layers){
        this.block = lstmBlock;
        this.layers = layers;
        this.learning = learning;
        this.lstmLayers = new LSTMLayer[this.layers.length - 1];
        for (int i = 0;i < lstmLayers.length;++i){
            lstmLayers[i] = new LSTMLayer(this.layers[i].getOutputSize(),lstmBlock);
        }
    }

    @Override
    public void init() {

    }


    @Override
    protected LSTMModel clone(){
        Layer[] ls = new Layer[layers.length];
        for (int j = 0;j < layers.length;++j){
            ls[j] = layers[j].clone();
        }
        LSTMLayer[] lls = new LSTMLayer[lstmLayers.length];
        for (int j = 0;j < lstmLayers.length;++j){
            lls[j] = lstmLayers[j].clone();
        }
        LSTMModel model = new LSTMModel(block,learning,ls);
        model.input = input;
        model.output = output;
        model.lstmLayers = lls;
        return model;
    }

    @Override
    public String convertToJson() {
        return null;
    }

    @Override
    public INDArray doBackPropagation(INDArray target) {
        return doBackPropagationByDiff(target.sub(output).mul(learning));
    }

    @Override
    public INDArray doBackPropagationByDiff(INDArray target) {
        INDArray res = layers[layers.length - 1].bp(target);
        for (int i = lstmLayers.length - 1;i >= 0;--i){
            res = lstmLayers[i].bp(res);
            res = layers[i].bp(res);
        }
        return res;
    }

    @Override
    public INDArray calc() {
        INDArray k = input;
        for (int i = 0;i < lstmLayers.length;++i){
            layers[i].setInput(k);
            k = layers[i].calc();
            lstmLayers[i].setInput(k);
            k = lstmLayers[i].calc();
        }
        layers[layers.length - 1].setInput(k);
        output = layers[layers.length - 1].calc();
        return output;
    }

    @Override
    public ModelBase initial() {
        return null;
    }

    @Override
    public int getInputSize() {
        return layers[0].perceptron.rows();
    }

    @Override
    public int getOutputSize() {
        return layers[layers.length - 1].perceptron.rows();
    }

    public LSTMModel next(){
        LSTMModel model = new LSTMModel(lstmLayers[0].blocks[0],learning,layers);
        for (int i = 0;i < lstmLayers.length;++i){
            model.lstmLayers[i] = lstmLayers[i].next();
        }
        return model;
    }


}
