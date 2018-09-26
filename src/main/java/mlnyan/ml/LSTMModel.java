package mlnyan.ml;

import mlnyan.ml.ModelBase;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LSTMModel extends ModelBase {

    Layer[] layers;
    LSTMLayer[] lstmLayers;
    public LSTMModel(LSTMBlock lstmBlock,Layer... layers){
        this.layers = layers;
        this.lstmLayers = new LSTMLayer[this.layers.length - 1];
        for (int i = 0;i < lstmLayers.length;++i){
            lstmLayers[i] = new LSTMLayer(this.layers[i].getOutputSize(),lstmBlock);
        }
    }

    @Override
    public void init() {

    }


    @Override
    public String convertToJson() {
        return null;
    }

    @Override
    public INDArray doBackPropagation(INDArray target) {
        return doBackPropagationByDiff(target.sub(layers[layers.length - 1].perceptron));
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
            lstmLayers[i].setInput(k);
            k = lstmLayers[i].calc();
            layers[i].setInput(k);
            k = layers[i].calc();
        }
        layers[layers.length - 1].setInput(k);
        return layers[layers.length - 1].calc();
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
}
