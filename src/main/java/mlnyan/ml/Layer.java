package mlnyan.ml;

import mlnyan.sys.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Layer extends LayerBase{
    private final Activator activator;
    private final IBackPropagation backPropagation;
    INDArray perceptron;
    INDArray bias;
    INDArray weight;
    public Layer(int p1,int p2,Activator activator,IBackPropagation backPropagation){
        this.perceptron = Nd4j.zeros(p1,1);
        this.bias = Nd4j.zeros(p2,1);
        this.weight = Nd4j.rand(p2,p1).div(Math.sqrt((double)p1 / 2));
        this.activator = activator;
        this.backPropagation = backPropagation;
    }

    @Override
    protected Layer clone(){
        Layer layer = new Layer(getInputSize(),getOutputSize(),activator,backPropagation.clone());
        layer.weight = this.weight.dup();
        layer.perceptron = this.perceptron.dup();
        layer.bias = this.bias.dup();
        return layer;
    }

    @Override
    public int getInputSize() {
        return perceptron.rows();
    }

    @Override
    public int getOutputSize() {
        return weight.rows();
    }

    public Layer initial(){
        Layer layer = new Layer(getInputSize(),getOutputSize(),activator,backPropagation.clone());
        if(this.perceptron != null)layer.perceptron = Nd4j.zeros(this.perceptron.rows(),this.perceptron.columns());
        if(this.weight != null)layer.weight = Nd4j.zeros(this.weight.rows(),this.weight.columns());
        if(this.bias != null)layer.bias = Nd4j.zeros(this.bias.rows(),this.bias.columns());
        return layer;
    }

    @Override
    public INDArray calc() {
        INDArray indArray = weight.mmul(perceptron).add(bias);
        switch (activator){
            case SOFTMAX:indArray = Transforms.softmax(indArray);break;
            case RELU:indArray = Transforms.relu(indArray);break;
            case TANH:indArray = Transforms.tanh(indArray);break;
            case SIGMOID:indArray = Transforms.sigmoid(indArray);break;
            default:
                Logger.error("Illegal activator -> " + activator);
                System.exit(1);
        }
        return indArray;
    }

    @Override
    public void setInput(INDArray input) {
        if(input == null || this.perceptron.rows() != input.rows()){
            System.out.println(this.perceptron.rows() + "/" +input.rows() );
            Logger.error("Illegal input");
            return;
        }
        this.perceptron = input;
    }


    @Override
    public INDArray bp(INDArray diff) {
        return backPropagation.calc(this,diff);
    }

    Activator getActivator() {
        return activator;
    }
}
