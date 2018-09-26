package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layer extends LayerBase{
    private final Activator activator;
    private final IBackPropagation backPropagation;
    INDArray perceptron;
    INDArray bias;
    INDArray weight;
    public Layer(int perceptron,Activator activator,IBackPropagation backPropagation){
        this.perceptron = Nd4j.zeros(perceptron,1);
        this.bias = Nd4j.zeros(perceptron,1);
        this.weight = Nd4j.zeros(1,perceptron);
        this.activator = activator;
        this.backPropagation = backPropagation;
    }

    @Override
    protected Layer clone(){
        Layer layer = new Layer(perceptron.rows(),activator,backPropagation);
        layer.weight = this.weight.dup();
        layer.perceptron = this.perceptron.dup();
        layer.bias = this.bias.dup();
        return layer;
    }

    public Layer initial(){
        Layer layer = new Layer(perceptron.rows(),activator,backPropagation);
        if(this.perceptron != null)layer.perceptron = Nd4j.zeros(this.perceptron.rows(),this.perceptron.columns());
        if(this.weight != null)layer.weight = Nd4j.zeros(this.weight.rows(),this.weight.columns());
        if(this.bias != null)layer.bias = Nd4j.zeros(this.bias.rows(),this.bias.columns());
        return layer;
    }

    @Override
    public INDArray bp(INDArray diff) {
        return backPropagation.calc(this,diff);
    }

    Activator getActivator() {
        return activator;
    }
}
