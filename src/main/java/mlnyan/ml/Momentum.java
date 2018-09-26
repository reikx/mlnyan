package mlnyan.ml;

import mlnyan.sys.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Momentum implements IBackPropagation{
    Layer momentumLayer;
    private final double move;
    private final double learning;

    public Momentum(double learning,double move){
        this.learning = learning;
        this.move = move;
    }

    @Override
    public INDArray calc(Layer layer, INDArray diff) {
        INDArray val = layer.weight.mmul(layer.perceptron).add(layer.bias);
        if(momentumLayer == null){
            momentumLayer = layer.initial();
        }
        switch (layer.getActivator()){
            case SOFTMAX:val = MatrixUtil.softmaxDeriv(val);break;
            case SIGMOID:val = MatrixUtil.sigmoidDeriv(val);break;
            case RELU:val = MatrixUtil.reluDeriv(val);break;
            case TANH:val = MatrixUtil.tanhDeriv(val);break;
            default:
                Logger.error("Illegal activator -> " + layer.getActivator());
                System.exit(1);
        }
        val = val.mul(diff);
        momentumLayer.weight = layer.perceptron.mmul(val.transpose()).transpose().add(momentumLayer.weight.mul(move));
        momentumLayer.bias = val.add(momentumLayer.bias.mul(move));
        layer.weight = layer.weight.sub(momentumLayer.weight);
        layer.bias = layer.bias.sub(momentumLayer.bias);
        return layer.weight.transpose().mmul(val);
    }

    @Override
    public Momentum initial() {
        Momentum momentum = new Momentum(learning,move);
        momentum.momentumLayer = momentumLayer.initial();
        return momentum;
    }

    @Override
    public Momentum clone(){
        Layer layer = momentumLayer.initial();
        layer.perceptron = momentumLayer.perceptron.dup();
        layer.bias = momentumLayer.bias.dup();
        layer.weight = momentumLayer.weight.dup();
        Momentum momentum = new Momentum(learning,move);
        momentum.momentumLayer = layer;
        return momentum;
    }
}
